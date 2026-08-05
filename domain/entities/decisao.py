"""Entidade do veredito final sobre uma regiao analisada.

`motivos` e `contribuicoes` nao sao enfeite: sao o que permite responder "por que
esta regiao caiu nesta fila?" sem reexecutar a GPU. Numa pipeline em que cada
imagem gera dezenas de regioes, essa e a diferenca entre depurar em segundos e
depurar em horas.

Typical usage:
    decisao = Decisao(
        fila=Fila.AUTO_ACEITE,
        marca="nike",
        pontuacao=0.84,
        motivos=("geometria confirmou com 28 inliers",),
        contribuicoes={"similaridade": 0.36, "margem": 0.07},
    )
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.enums.fila import Fila
from domain.exceptions.domain_exceptions import DomainError


@dataclass(frozen=True)
class Decisao:
    """O que a pipeline decidiu sobre uma regiao, e por que.

    Attributes:
        fila: Destino operacional da regiao.
        marca: Marca afirmada, ou None quando nao ha marca a afirmar.
        pontuacao: Evidencia agregada, entre 0.0 e 1.0.
        motivos: Regras aplicadas, na ordem em que decidiram.
        contribuicoes: Quanto cada termo somou a pontuacao. A soma dos valores
            e a propria pontuacao quando a decisao veio pelos limiares.
    """

    fila: Fila
    marca: str | None
    pontuacao: float
    motivos: tuple[str, ...] = field(default_factory=tuple)
    contribuicoes: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida a pontuacao e a invariante de rejeicao sem marca.

        Raises:
            DomainError: Se a pontuacao estiver fora de [0, 1], ou se uma
                regiao rejeitada afirmar uma marca.
        """
        if not 0.0 <= self.pontuacao <= 1.0:
            raise DomainError(f"pontuacao fora de [0, 1]: {self.pontuacao}")
        if self.fila is Fila.AUTO_REJEICAO and self.marca is not None:
            raise DomainError(
                "regiao em AUTO_REJEICAO nao pode afirmar marca — " f"recebido: {self.marca!r}"
            )
