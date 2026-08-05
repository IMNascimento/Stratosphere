"""Comandos e saidas dos casos de uso.

DTOs sao imutaveis e nao carregam regra de negocio — apenas transportam dados
entre a borda e a aplicacao.

Typical usage:
    comando = AnalisarImagemCommand(caminho=Path("foto.jpg"))
    saida = caso_de_uso.execute(comando)
"""

from dataclasses import dataclass, field
from pathlib import Path

from domain.enums.fila import Fila


@dataclass(frozen=True)
class AnalisarImagemCommand:
    """Pedido de analise de uma imagem.

    Attributes:
        caminho: Arquivo a analisar.
    """

    caminho: Path


@dataclass(frozen=True)
class RegiaoOutput:
    """Resultado da analise de uma regiao.

    Attributes:
        identificador: Chave da regiao dentro da execucao.
        caixa: Coordenadas `(x1, y1, x2, y2)` na imagem original.
        fila: Destino operacional.
        marca: Marca afirmada, ou None.
        pontuacao: Evidencia agregada, entre 0 e 1.
        similaridade: Semelhanca com a melhor referencia.
        margem: Vantagem sobre a rival mais proxima.
        inliers: Pontos coerentes da verificacao geometrica. Zero quando a
            camada nao opinou — o campo nao distingue os dois casos, e por isso
            `motivos` existe.
        motivos: Regras aplicadas, na ordem em que decidiram.
    """

    identificador: str
    caixa: tuple[int, int, int, int]
    fila: Fila
    marca: str | None
    pontuacao: float
    similaridade: float
    margem: float
    inliers: int
    motivos: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnaliseOutput:
    """Resultado da analise de uma imagem inteira.

    Attributes:
        caminho: Arquivo analisado.
        descartada_por: Motivo do descarte no pre-filtro, ou None se a imagem
            foi processada.
        regioes: Uma entrada por regiao detectada.
    """

    caminho: Path
    descartada_por: str | None
    regioes: tuple[RegiaoOutput, ...] = field(default_factory=tuple)

    def marcas_aceitas(self) -> tuple[str, ...]:
        """Retorna as marcas que a imagem entrega sem revisao humana.

        Returns:
            Marcas distintas em ordem alfabetica, apenas das regioes em
            auto-aceite. E a resposta que vai para o relatorio do cliente.
        """
        marcas = {
            regiao.marca
            for regiao in self.regioes
            if regiao.fila is Fila.AUTO_ACEITE and regiao.marca
        }
        return tuple(sorted(marcas))

    def contagem_por_fila(self) -> dict[str, int]:
        """Conta quantas regioes caíram em cada fila.

        Returns:
            Mapa nome da fila -> contagem, apenas com as filas presentes.
        """
        contagem: dict[str, int] = {}
        for regiao in self.regioes:
            contagem[regiao.fila.value] = contagem.get(regiao.fila.value, 0) + 1
        return contagem


@dataclass(frozen=True)
class ConstruirBancoCommand:
    """Pedido de construcao do banco de referencia.

    Attributes:
        pasta_de_referencias: Raiz com o layout `<marca>/<variante>/arquivo`.
        destino: Onde gravar o banco construido.
    """

    pasta_de_referencias: Path
    destino: Path


@dataclass(frozen=True)
class BancoOutput:
    """Resultado da construcao do banco.

    Attributes:
        total_de_referencias: Quantas referencias entraram.
        referencias_por_marca: Contagem por marca.
        descartadas_por_redundancia: Referencias que nao entraram por serem
            quase identicas a outra da mesma marca.
        pares_confundiveis: Pares de marcas DIFERENTES cujas referencias se
            parecem demais. Cada um e um falso positivo agendado.
    """

    total_de_referencias: int
    referencias_por_marca: dict[str, int]
    descartadas_por_redundancia: int
    pares_confundiveis: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)
