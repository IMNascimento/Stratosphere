"""Value object do resultado da verificacao geometrica.

A verificacao geometrica responde uma pergunta diferente da busca vetorial:
"e literalmente o mesmo desenho, sob alguma transformacao coerente?" — nao
"parece com", que e o que o codificador responde.

As duas convivem porque **falham de formas diferentes**. A busca vetorial e
fraca justamente em mudanca de ponto de vista; casamento de pontos sob
homografia foi projetado para isso. Em compensacao, a verificacao geometrica
fica muda em logo chapado, pequeno ou vetorial demais para ter cantos.

Typical usage:
    veredito = VereditoGeometrico(
        marca="nike", inliers=28, correspondencias=31, confirma=True
    )
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import DomainError


@dataclass(frozen=True)
class VereditoGeometrico:
    """Resultado da verificacao de um par (regiao, referencia de uma marca).

    Attributes:
        marca: Marca da referencia comparada.
        inliers: Pontos que casam sob uma unica transformacao coerente. E o
            numero que mede a forca da evidencia.
        correspondencias: Pares de pontos que sobreviveram ao teste de razao,
            antes de exigir coerencia geometrica. Sempre >= inliers.
        confirma: Se a evidencia foi suficiente para afirmar que e o mesmo
            desenho. Falso nao significa "e outra marca" — significa "nao deu
            para confirmar", que e diferente e precisa ser tratado diferente.
        motivo: Explicacao curta de por que nao confirmou. Vazio quando confirma.
    """

    marca: str
    inliers: int
    correspondencias: int
    confirma: bool
    motivo: str = ""

    def __post_init__(self) -> None:
        """Valida contagens nao negativas e coerentes entre si.

        Raises:
            DomainError: Se alguma contagem for negativa ou se houver mais
                inliers que correspondencias.
        """
        if not self.marca:
            raise DomainError("marca do veredito nao pode ser vazia")
        if self.inliers < 0 or self.correspondencias < 0:
            raise DomainError(
                f"contagens nao podem ser negativas: inliers={self.inliers} "
                f"correspondencias={self.correspondencias}"
            )
        if self.inliers > self.correspondencias:
            raise DomainError(
                f"inliers nao pode exceder correspondencias: "
                f"{self.inliers} > {self.correspondencias}"
            )
