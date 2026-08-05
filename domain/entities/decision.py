"""Entidade do veredito final sobre uma regiao analisada.

`reasons` e `contributions` nao sao enfeite: sao o que permite responder "por que
esta regiao caiu nesta fila?" sem reexecutar a GPU. Numa pipeline em que cada
imagem gera dezenas de regioes, essa e a diferenca entre depurar em segundos e
depurar em horas.

Typical usage:
    decision = Decision(
        queue=Queue.AUTO_ACCEPT,
        brand="nike",
        score=0.84,
        reasons=("geometria confirmou com 28 inliers",),
        contributions={"similarity": 0.36, "margin": 0.07},
    )
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from domain.enums.queue import Queue
from domain.exceptions.domain_exceptions import DomainError


@dataclass(frozen=True)
class Decision:
    """O que a pipeline decidiu sobre uma regiao, e por que.

    Attributes:
        queue: Destino operacional da regiao.
        brand: Marca afirmada, ou None quando nao ha marca a afirmar.
        score: Evidencia agregada, entre 0.0 e 1.0.
        reasons: Regras aplicadas, na ordem em que decidiram.
        contributions: Quanto cada termo somou a pontuacao. A soma dos valores
            e a propria pontuacao quando a decisao veio pelos limiares.
    """

    queue: Queue
    brand: str | None
    score: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    contributions: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida a pontuacao e a invariante de rejeicao sem marca.

        Raises:
            DomainError: Se a pontuacao estiver fora de [0, 1], ou se uma
                regiao rejeitada afirmar uma marca.
        """
        if not 0.0 <= self.score <= 1.0:
            raise DomainError(f"pontuacao fora de [0, 1]: {self.score}")
        if self.queue is Queue.AUTO_REJECT and self.brand is not None:
            raise DomainError(
                "regiao em AUTO_REJECT nao pode afirmar marca — " f"recebido: {self.brand!r}"
            )
