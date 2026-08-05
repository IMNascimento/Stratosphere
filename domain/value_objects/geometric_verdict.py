"""Value object do resultado da verificacao geometrica.

A verificacao geometrica responde uma pergunta diferente da busca vetorial:
"e literalmente o mesmo desenho, sob alguma transformacao coerente?" — nao
"parece com", que e o que o codificador responde.

As duas convivem porque **falham de formas diferentes**. A busca vetorial e
fraca justamente em mudanca de ponto de vista; casamento de pontos sob
homografia foi projetado para isso. Em compensacao, a verificacao geometrica
fica muda em logo chapado, pequeno ou vetorial demais para ter cantos.

Typical usage:
    verdict = GeometricVerdict(
        brand="nike", inliers=28, matches=31, confirms=True
    )
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import DomainError


@dataclass(frozen=True)
class GeometricVerdict:
    """Resultado da verificacao de um par (regiao, referencia de uma marca).

    Attributes:
        brand: Marca da referencia comparada.
        inliers: Pontos que casam sob uma unica transformacao coerente. E o
            numero que mede a forca da evidencia.
        matches: Pares de pontos que sobreviveram ao teste de razao,
            antes de exigir coerencia geometrica. Sempre >= inliers.
        confirms: Se a evidencia foi suficiente para afirmar que e o mesmo
            desenho. Falso nao significa "e outra marca" — significa "nao deu
            para confirmar", que e diferente e precisa ser tratado diferente.
        reason: Explicacao curta de por que nao confirmou. Vazio quando confirma.
    """

    brand: str
    inliers: int
    matches: int
    confirms: bool
    reason: str = ""

    def __post_init__(self) -> None:
        """Valida contagens nao negativas e coerentes entre si.

        Raises:
            DomainError: Se alguma contagem for negativa ou se houver mais
                inliers que correspondencias.
        """
        if not self.brand:
            raise DomainError("marca do veredito nao pode ser vazia")
        if self.inliers < 0 or self.matches < 0:
            raise DomainError(
                f"contagens nao podem ser negativas: inliers={self.inliers} "
                f"correspondencias={self.matches}"
            )
        if self.inliers > self.matches:
            raise DomainError(
                f"inliers nao pode exceder correspondencias: " f"{self.inliers} > {self.matches}"
            )
