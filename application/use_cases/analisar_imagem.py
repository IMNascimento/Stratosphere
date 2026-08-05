"""Caso de uso que atravessa a pipeline inteira para uma imagem.

Orquestra, nao decide. Toda regra de negocio esta em `RoteadorDeFila`; aqui ha
apenas a sequencia das camadas e o cuidado de nao gastar camada cara com o que a
camada barata ja descartou.

A ordem importa e nao e arbitraria: cada camada e mais cara que a anterior, e so
ve o que a anterior deixou passar. A verificacao geometrica, em particular, so
roda nos melhores candidatos de cada regiao — se rodasse em tudo, seria a camada
dominante do custo.

Typical usage:
    caso = AnalisarImagemUseCase(detector, codificador, verificador, fonte, banco, roteador, config)
    saida = caso.execute(AnalisarImagemCommand(caminho))
"""

from application.dtos.analise import AnalisarImagemCommand, AnaliseOutput, RegiaoOutput
from application.ports.i_codificador import ICodificador
from application.ports.i_detector import IDetector
from application.ports.i_fonte_de_imagens import IFonteDeImagens, ImagemRgb
from application.ports.i_verificador_geometrico import IVerificadorGeometrico
from config.settings import AppConfig
from domain.entities.regiao_analisada import RegiaoAnalisada
from domain.repositories.i_banco_referencia import IBancoReferencia
from domain.services.roteador_de_fila import RoteadorDeFila


class AnalisarImagemUseCase:
    """Executa pre-filtro, deteccao, codificacao, busca, geometria e roteamento."""

    def __init__(
        self,
        detector: IDetector,
        codificador: ICodificador,
        verificador: IVerificadorGeometrico,
        fonte: IFonteDeImagens,
        banco: IBancoReferencia,
        roteador: RoteadorDeFila,
        config: AppConfig,
    ) -> None:
        """Recebe as portas e o servico de dominio ja construidos.

        Args:
            detector: Camada que encontra onde ha marca.
            codificador: Camada que transforma regiao em vetor.
            verificador: Camada que confirma se e o mesmo desenho.
            fonte: Acesso a imagem e recorte.
            banco: Banco de referencia consultado pela busca vetorial.
            roteador: Servico de dominio que decide a fila.
            config: Parametros de todas as camadas.
        """
        self._detector = detector
        self._codificador = codificador
        self._verificador = verificador
        self._fonte = fonte
        self._banco = banco
        self._roteador = roteador
        self._config = config

    def execute(self, command: AnalisarImagemCommand) -> AnaliseOutput:
        """Analisa uma imagem e devolve a decisao de cada regiao encontrada.

        Args:
            command: Pedido com o caminho da imagem.

        Returns:
            A analise completa. Quando o pre-filtro descarta a imagem,
            `descartada_por` explica o motivo e `regioes` vem vazio.

        Raises:
            OSError: Se a imagem nao puder ser lida.
        """
        imagem = self._fonte.carregar(command.caminho)

        motivo_do_descarte = self._descartar(imagem)
        if motivo_do_descarte is not None:
            return AnaliseOutput(caminho=command.caminho, descartada_por=motivo_do_descarte)

        deteccoes = self._detector.detectar(imagem)
        if not deteccoes:
            return AnaliseOutput(caminho=command.caminho, descartada_por=None)

        recortes = self._recortar(imagem, deteccoes)
        vetores = self._codificador.codificar(recortes)

        regioes: list[RegiaoOutput] = []
        for indice, (deteccao, recorte) in enumerate(zip(deteccoes, recortes, strict=True)):
            candidatos = self._banco.buscar(vetores[indice], self._config.busca.vizinhos)

            regiao = RegiaoAnalisada(
                identificador=f"{command.caminho.stem}-{indice:03d}",
                deteccao=deteccao,
            ).com_candidatos(candidatos)

            if self._config.geometria.ativa and candidatos:
                regiao = regiao.com_vereditos(self._verificador.verificar(recorte, candidatos))

            regioes.append(self._para_saida(regiao))

        return AnaliseOutput(caminho=command.caminho, descartada_por=None, regioes=tuple(regioes))

    def _descartar(self, imagem: ImagemRgb) -> str | None:
        """Decide se a imagem nao merece uma passada de detector.

        Deliberadamente permissivo: o recall do detector e o teto do sistema
        inteiro, e o que se descarta aqui nunca mais volta.

        Args:
            imagem: Imagem carregada.

        Returns:
            Motivo do descarte, ou None se a imagem deve ser processada.
        """
        config = self._config.pre_filtro
        if not config.ativo:
            return None

        largura, altura = self._fonte.dimensoes(imagem)
        if min(largura, altura) < config.lado_minimo:
            return f"lado menor que {config.lado_minimo}px — miniatura ou icone"

        densidade = self._fonte.densidade_de_bordas(imagem)
        if densidade < config.densidade_de_bordas_minima:
            return (
                f"densidade de bordas {densidade:.4f} abaixo de "
                f"{config.densidade_de_bordas_minima} — imagem sem estrutura"
            )
        return None

    def _recortar(self, imagem: ImagemRgb, deteccoes: tuple) -> list[ImagemRgb]:  # type: ignore[type-arg]
        """Recorta todas as regioes detectadas de uma vez.

        Args:
            imagem: Imagem de origem.
            deteccoes: Regioes encontradas pelo detector.

        Returns:
            Recortes na mesma ordem das deteccoes.
        """
        config = self._config.codificador
        return [
            self._fonte.recortar(
                imagem,
                deteccao.caixa,
                config.margem_do_recorte,
                config.lado_do_recorte,
            )
            for deteccao in deteccoes
        ]

    def _para_saida(self, regiao: RegiaoAnalisada) -> RegiaoOutput:
        """Converte a regiao analisada e sua decisao no DTO de saida.

        Args:
            regiao: Regiao com candidatos e vereditos ja reunidos.

        Returns:
            O DTO correspondente, com os sinais que justificam a decisao.
        """
        decisao = self._roteador.rotear(regiao)
        veredito = regiao.melhor_veredito
        caixa = regiao.deteccao.caixa
        return RegiaoOutput(
            identificador=regiao.identificador,
            caixa=(caixa.x1, caixa.y1, caixa.x2, caixa.y2),
            fila=decisao.fila,
            marca=decisao.marca,
            pontuacao=decisao.pontuacao,
            similaridade=regiao.similaridade_topo,
            margem=regiao.margem,
            inliers=veredito.inliers if veredito is not None else 0,
            motivos=decisao.motivos,
        )
