"""Value object de resposta do banco de referencia.

Aqui existe marca. Em `Detection`, nao - e essa assimetria e o coracao da
arquitetura: o detector diz **onde**, o banco diz **qual**.

Typical usage:
    candidate = Candidate(brand="nike", similarity=0.91, reference="nike/01.jpg")
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import InvalidCandidateError

_MIN_SIMILARITY: float = -1.0
_MAX_SIMILARITY: float = 1.0


@dataclass(frozen=True)
class Candidate:
    """Uma marca proposta pelo banco de referencia para uma regiao.

    Attributes:
        brand: Identificador da marca. Nao vazio, minusculo por convencao.
        similarity: Cosseno entre o vetor da regiao e o da referencia.
            Entre -1.0 e 1.0. Valores tipicos de correspondencia real ficam
            bem acima de 0.5 - o valor absoluto so significa alguma coisa
            comparado a distribuicao do proprio banco.
        reference: Caminho da imagem de referencia que gerou o vetor. Serve
            para auditar por que uma referencia esta puxando falso positivo.
    """

    brand: str
    similarity: float
    reference: str

    def __post_init__(self) -> None:
        """Valida marca preenchida e similaridade no intervalo do cosseno.

        Raises:
            InvalidCandidateError: Se a marca for vazia ou a similaridade
                estiver fora de [-1.0, 1.0].
        """
        if not self.brand:
            raise InvalidCandidateError("marca nao pode ser vazia")
        if not _MIN_SIMILARITY <= self.similarity <= _MAX_SIMILARITY:
            raise InvalidCandidateError(f"similaridade fora de [-1, 1]: {self.similarity}")
