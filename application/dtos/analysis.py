"""Comandos e saidas dos casos de uso.

DTOs sao imutaveis e nao carregam regra de negocio — apenas transportam dados
entre a borda e a aplicacao.

Typical usage:
    command = AnalyzeImageCommand(path=Path("foto.jpg"))
    output = use_case.execute(command)
"""

from dataclasses import dataclass, field
from pathlib import Path

from domain.enums.queue import Queue


@dataclass(frozen=True)
class AnalyzeImageCommand:
    """Pedido de analise de uma imagem.

    Attributes:
        path: Arquivo a analisar.
    """

    path: Path


@dataclass(frozen=True)
class RegionOutput:
    """Resultado da analise de uma regiao.

    Attributes:
        identifier: Chave da regiao dentro da execucao.
        box: Coordenadas `(x1, y1, x2, y2)` na imagem original.
        queue: Destino operacional.
        brand: Marca afirmada, ou None.
        score: Evidencia agregada, entre 0 e 1.
        similarity: Semelhanca com a melhor referencia.
        margin: Vantagem sobre a rival mais proxima.
        inliers: Pontos coerentes da verificacao geometrica. Zero quando a
            camada nao opinou — o campo nao distingue os dois casos, e por isso
            `reasons` existe.
        reasons: Regras aplicadas, na ordem em que decidiram.
    """

    identifier: str
    box: tuple[int, int, int, int]
    queue: Queue
    brand: str | None
    score: float
    similarity: float
    margin: float
    inliers: int
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnalysisOutput:
    """Resultado da analise de uma imagem inteira.

    Attributes:
        path: Arquivo analisado.
        discarded_by: Motivo do descarte no pre-filtro, ou None se a imagem
            foi processada.
        regions: Uma entrada por regiao detectada.
    """

    path: Path
    discarded_by: str | None
    regions: tuple[RegionOutput, ...] = field(default_factory=tuple)

    def accepted_brands(self) -> tuple[str, ...]:
        """Retorna as marcas que a imagem entrega sem revisao humana.

        Returns:
            Marcas distintas em ordem alfabetica, apenas das regioes em
            auto-aceite. E a resposta que vai para o relatorio do cliente.
        """
        brands = {
            region.brand
            for region in self.regions
            if region.queue is Queue.AUTO_ACCEPT and region.brand
        }
        return tuple(sorted(brands))

    def count_by_queue(self) -> dict[str, int]:
        """Conta quantas regioes caíram em cada fila.

        Returns:
            Mapa nome da fila -> contagem, apenas com as filas presentes.
        """
        counts: dict[str, int] = {}
        for region in self.regions:
            counts[region.queue.value] = counts.get(region.queue.value, 0) + 1
        return counts


@dataclass(frozen=True)
class BuildDatabaseCommand:
    """Pedido de construcao do banco de referencia.

    Attributes:
        references_folder: Raiz com o layout `<marca>/<variante>/arquivo`.
        destination: Onde gravar o banco construido.
    """

    references_folder: Path
    destination: Path


@dataclass(frozen=True)
class DatabaseOutput:
    """Resultado da construcao do banco.

    Attributes:
        total_references: Quantas referencias entraram.
        references_by_brand: Contagem por marca.
        discarded_by_redundancy: Referencias que nao entraram por serem quase
            identicas a outra da mesma marca.
        confusable_pairs: Pares de marcas DIFERENTES cujas referencias se
            parecem demais. Cada um e um falso positivo agendado.
    """

    total_references: int
    references_by_brand: dict[str, int]
    discarded_by_redundancy: int
    confusable_pairs: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)
