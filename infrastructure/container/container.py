"""Montagem do grafo de dependencias.

**Este e o unico arquivo que instancia infraestrutura concreta.** Nenhuma outra
camada sabe que existe OWLv2, DINOv2 ou SIFT — todas falam com as portas. Trocar
qualquer adaptador e uma mudanca aqui e em mais lugar nenhum.

Tambem e o unico lugar que le variavel de ambiente, quando houver.

Typical usage:
    container = construir_container(AppConfig(), Path("indice"))
    saida = container.analisar_imagem.execute(comando)
"""

from dataclasses import dataclass
from pathlib import Path

from application.use_cases.analisar_imagem import AnalisarImagemUseCase
from application.use_cases.auditar_banco import AuditarBancoUseCase
from application.use_cases.construir_banco import ConstruirBancoUseCase
from config.settings import AppConfig
from domain.services.grupos_de_confusao import GruposDeConfusao
from domain.services.roteador_de_fila import Calibragem, PesosDeEvidencia, RoteadorDeFila
from infrastructure.codificacao.dinov2_codificador import Dinov2Codificador
from infrastructure.deteccao.owlv2_detector import Owlv2Detector
from infrastructure.geometria.sift_verificador import SiftVerificador
from infrastructure.imagem.pillow_fonte_imagens import PillowFonteImagens
from infrastructure.indice.banco_referencia_npz import (
    ARQUIVO_DE_METADADOS,
    ARQUIVO_DE_VETORES,
    BancoReferenciaNpz,
    GravadorBancoNpz,
)


@dataclass(frozen=True)
class Container:
    """Casos de uso prontos para uso, com as dependencias ja injetadas.

    Attributes:
        analisar_imagem: Pipeline completa para uma imagem. None quando o banco
            ainda nao existe — construir o banco nao exige banco.
        construir_banco: Pasta de referencias para indice vetorial.
        auditar_banco: Pares de marcas confundiveis. None sem banco.
        fonte_de_imagens: Exposto porque o entrypoint precisa listar pastas.
    """

    construir_banco: ConstruirBancoUseCase
    fonte_de_imagens: PillowFonteImagens
    analisar_imagem: AnalisarImagemUseCase | None = None
    auditar_banco: AuditarBancoUseCase | None = None


def construir_container(config: AppConfig, caminho_do_banco: Path) -> Container:
    """Monta o grafo completo de dependencias.

    O banco de referencia e carregado quando existe. Quando nao existe, os casos
    de uso que dependem dele ficam em None em vez de a montagem falhar — isso
    permite que `construir_banco` rode numa instalacao limpa, que e exatamente o
    primeiro comando que alguem executa.

    Args:
        config: Configuracao completa.
        caminho_do_banco: Pasta do indice vetorial.

    Returns:
        O container com os casos de uso disponiveis.

    Raises:
        CodificadorIncompativelError: Se o banco existir mas tiver sido
            construido com outro codificador.
    """
    fonte = PillowFonteImagens(lado_minimo_para_ampliar=config.codificador.lado_minimo_para_ampliar)
    codificador = Dinov2Codificador(
        config=config.codificador,
        dispositivo=config.dispositivo,
        precisao=config.precisao,
    )
    construir = ConstruirBancoUseCase(
        codificador=codificador, fonte=fonte, gravar=GravadorBancoNpz(), config=config
    )
    container = Container(construir_banco=construir, fonte_de_imagens=fonte)

    if not _banco_existe(caminho_do_banco):
        return container

    banco = BancoReferenciaNpz.carregar(caminho_do_banco, codificador.identificacao())
    analisar = AnalisarImagemUseCase(
        detector=Owlv2Detector(
            config=config.detector,
            dispositivo=config.dispositivo,
            precisao=config.precisao,
        ),
        codificador=codificador,
        verificador=SiftVerificador(config=config.geometria),
        fonte=fonte,
        banco=banco,
        roteador=_montar_roteador(config),
        config=config,
    )
    return Container(
        construir_banco=construir,
        fonte_de_imagens=fonte,
        analisar_imagem=analisar,
        auditar_banco=AuditarBancoUseCase(banco=banco, config=config),
    )


def _montar_roteador(config: AppConfig) -> RoteadorDeFila:
    """Constroi o servico de dominio que decide as filas.

    Args:
        config: Configuracao completa.

    Returns:
        O roteador pronto.

    Raises:
        ValueError: Se os pesos configurados nao somarem 1 ou se algum limiar
            estiver invertido.
    """
    roteamento = config.roteamento
    return RoteadorDeFila(
        pesos=PesosDeEvidencia(
            similaridade=roteamento.peso_similaridade,
            margem=roteamento.peso_margem,
            geometria=roteamento.peso_geometria,
            deteccao=roteamento.peso_deteccao,
        ),
        calibragem=Calibragem(
            similaridade_minima=roteamento.similaridade_minima,
            similaridade_maxima=roteamento.similaridade_maxima,
            margem_confiante=roteamento.margem_confiante,
            inliers_confiantes=roteamento.inliers_confiantes,
            aceite=roteamento.aceite,
            rejeicao=roteamento.rejeicao,
            orfao_inliers_minimos=roteamento.orfao_inliers_minimos,
            orfao_similaridade_maxima=roteamento.orfao_similaridade_maxima,
        ),
        grupos=GruposDeConfusao(
            grupos=config.confusao.grupos,
            negativas=config.confusao.negativas,
            margem_de_empate=config.confusao.margem_de_empate,
        ),
    )


def _banco_existe(caminho: Path) -> bool:
    """Verifica se ha um banco gravado no caminho.

    Args:
        caminho: Pasta do indice.

    Returns:
        True quando os dois arquivos do banco estao presentes.
    """
    return (caminho / ARQUIVO_DE_VETORES).exists() and (caminho / ARQUIVO_DE_METADADOS).exists()
