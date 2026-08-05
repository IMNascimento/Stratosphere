"""Caso de uso que transforma uma pasta de referencias no banco vetorial.

E este caso de uso que sustenta a afirmacao "marca nova em minutos": nao ha
treino, apenas codificacao de imagens novas e escrita de um arquivo.

O layout esperado e `<raiz>/<marca>/<variante>/arquivo`. A marca e o primeiro
nivel — e o rotulo que o sistema devolve. A variante e o segundo, e serve para
tornar visivel no `ls` qual tipo de aplicacao da marca esta sub-representado.

Typical usage:
    saida = caso.execute(ConstruirBancoCommand(pasta, destino))
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from application.dtos.analise import BancoOutput, ConstruirBancoCommand
from application.ports.i_codificador import ICodificador
from application.ports.i_fonte_de_imagens import IFonteDeImagens
from config.settings import AppConfig
from domain.exceptions.domain_exceptions import ReferenciasNaoEncontradasError
from domain.value_objects.caixa import Caixa

_PASTAS_IGNORADAS = (".", "_")


class ConstruirBancoUseCase:
    """Codifica as referencias de uma pasta e grava o banco vetorial."""

    def __init__(
        self,
        codificador: ICodificador,
        fonte: IFonteDeImagens,
        gravar: "GravadorDeBanco",
        config: AppConfig,
    ) -> None:
        """Recebe as portas e a funcao de gravacao ja construidas.

        Args:
            codificador: Camada que transforma referencia em vetor.
            fonte: Acesso a imagem e recorte.
            gravar: Objeto capaz de persistir o banco construido.
            config: Parametros de codificacao e de busca.
        """
        self._codificador = codificador
        self._fonte = fonte
        self._gravar = gravar
        self._config = config

    def execute(self, command: ConstruirBancoCommand) -> BancoOutput:
        """Constroi o banco a partir da pasta de referencias.

        Args:
            command: Pasta de origem e destino do banco.

        Returns:
            Resumo do que entrou, o que foi descartado por redundancia, e quais
            pares de marcas diferentes ficaram parecidos demais.

        Raises:
            ReferenciasNaoEncontradasError: Se a pasta nao contiver imagem
                utilizavel.
            FileNotFoundError: Se a pasta nao existir.
        """
        entradas = tuple(self._percorrer(command.pasta_de_referencias))
        if not entradas:
            raise ReferenciasNaoEncontradasError(
                f"nenhuma imagem em {command.pasta_de_referencias}. "
                f"Estrutura esperada: <marca>/<variante>/arquivo.jpg"
            )

        vetores = self._codificar(entradas)
        mantidas, descartadas = self._remover_redundantes(entradas, vetores)

        indices = [indice for indice, _ in mantidas]
        self._gravar.gravar(
            destino=command.destino,
            vetores=vetores[indices],
            marcas=tuple(marca for _, (marca, _, _) in mantidas),
            variantes=tuple(variante for _, (_, variante, _) in mantidas),
            caminhos=tuple(str(caminho) for _, (_, _, caminho) in mantidas),
            assinatura_do_codificador=self._codificador.identificacao(),
        )

        contagem: dict[str, int] = {}
        for _, (marca, _, _) in mantidas:
            contagem[marca] = contagem.get(marca, 0) + 1

        return BancoOutput(
            total_de_referencias=len(mantidas),
            referencias_por_marca=dict(sorted(contagem.items())),
            descartadas_por_redundancia=descartadas,
        )

    def _percorrer(self, raiz: Path) -> Iterable[tuple[str, str, Path]]:
        """Percorre a pasta de referencias produzindo `(marca, variante, caminho)`.

        Args:
            raiz: Pasta com o layout `<marca>/<variante>/arquivo`.

        Yields:
            Uma tupla por imagem encontrada, em ordem estavel.

        Raises:
            FileNotFoundError: Se a raiz nao existir.
        """
        if not raiz.exists():
            raise FileNotFoundError(f"pasta de referencias nao existe: {raiz}")

        for pasta_da_marca in sorted(p for p in raiz.iterdir() if p.is_dir()):
            if pasta_da_marca.name.startswith(_PASTAS_IGNORADAS):
                continue
            marca = pasta_da_marca.name.strip().lower()
            for caminho in self._fonte.listar(pasta_da_marca):
                relativo = caminho.relative_to(pasta_da_marca)
                if any(parte.startswith(_PASTAS_IGNORADAS) for parte in relativo.parts[:-1]):
                    continue
                variante = relativo.parts[0] if len(relativo.parts) > 1 else "raiz"
                yield marca, variante, caminho

    def _codificar(self, entradas: tuple[tuple[str, str, Path], ...]) -> NDArray[np.float32]:
        """Codifica todas as referencias em lote.

        As referencias sao preparadas do mesmo jeito que uma regiao de consulta
        sera preparada. Assimetria aqui — referencia esticada e consulta com
        letterbox — e uma fonte silenciosa de similaridade baixa em par que
        deveria casar.

        Args:
            entradas: Tuplas `(marca, variante, caminho)`.

        Returns:
            Matriz `(len(entradas), dimensao)` L2-normalizada.
        """
        lado = self._config.codificador.lado_do_recorte
        recortes = []
        for _, _, caminho in entradas:
            imagem = self._fonte.carregar(caminho)
            largura, altura = self._fonte.dimensoes(imagem)
            recortes.append(self._fonte.recortar(imagem, Caixa(0, 0, largura, altura), 0.0, lado))
        return self._codificador.codificar(recortes)

    def _remover_redundantes(
        self,
        entradas: tuple[tuple[str, str, Path], ...],
        vetores: NDArray[np.float32],
    ) -> tuple[list[tuple[int, tuple[str, str, Path]]], int]:
        """Descarta referencias quase identicas a outra da MESMA marca.

        O corte e por marca e **nunca entre marcas**: duas marcas parecidas
        acima do limiar sao exatamente o que a auditoria precisa enxergar, nao
        algo a colapsar. Dentro de uma marca, porem, trinta fotos do mesmo
        angulo sao um vetor — elas ocupam o topo da busca com copias e escondem
        os angulos que faltam.

        Args:
            entradas: Tuplas `(marca, variante, caminho)`.
            vetores: Vetores correspondentes, L2-normalizados.

        Returns:
            Tupla `(mantidas, quantidade_descartada)`, onde `mantidas` preserva
            o indice original de cada entrada.
        """
        limiar = self._config.busca.similaridade_de_redundancia
        por_marca: dict[str, list[int]] = {}
        for indice, (marca, _, _) in enumerate(entradas):
            por_marca.setdefault(marca, []).append(indice)

        mantidas: list[tuple[int, tuple[str, str, Path]]] = []
        descartadas = 0
        for indices in por_marca.values():
            aceitos: list[int] = []
            for indice in indices:
                similaridades = [float(vetores[indice] @ vetores[j]) for j in aceitos]
                if similaridades and max(similaridades) > limiar:
                    descartadas += 1
                    continue
                aceitos.append(indice)
                mantidas.append((indice, entradas[indice]))
        mantidas.sort(key=lambda par: par[0])
        return mantidas, descartadas


class GravadorDeBanco:
    """Contrato minimo de persistencia do banco, satisfeito pela infraestrutura.

    Nao e uma ABC formal para nao criar um port so por isto: a implementacao
    concreta e a mesma classe que le o banco, e o container injeta o objeto.
    """

    def gravar(
        self,
        destino: Path,
        vetores: NDArray[np.float32],
        marcas: tuple[str, ...],
        variantes: tuple[str, ...],
        caminhos: tuple[str, ...],
        assinatura_do_codificador: str,
    ) -> None:
        """Persiste o banco construido.

        Args:
            destino: Pasta onde gravar.
            vetores: Matriz L2-normalizada.
            marcas: Marca de cada linha.
            variantes: Variante de cada linha.
            caminhos: Arquivo de origem de cada linha.
            assinatura_do_codificador: Assinatura de quem produziu os vetores.

        Raises:
            NotImplementedError: Sempre — a implementacao vive em infrastructure.
        """
        raise NotImplementedError
