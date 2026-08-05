"""Value object de resposta do banco de referencia.

Aqui existe marca. Em `Deteccao`, nao — e essa assimetria e o coracao da
arquitetura: o detector diz **onde**, o banco diz **qual**.

Typical usage:
    candidato = Candidato(marca="nike", similaridade=0.91, referencia="nike/01.jpg")
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import CandidatoInvalidoError

_SIMILARIDADE_MINIMA: float = -1.0
_SIMILARIDADE_MAXIMA: float = 1.0


@dataclass(frozen=True)
class Candidato:
    """Uma marca proposta pelo banco de referencia para uma regiao.

    Attributes:
        marca: Identificador da marca. Nao vazio, minusculo por convencao.
        similaridade: Cosseno entre o vetor da regiao e o da referencia.
            Entre -1.0 e 1.0. Valores tipicos de correspondencia real ficam
            bem acima de 0.5 — o valor absoluto so significa alguma coisa
            comparado a distribuicao do proprio banco.
        referencia: Caminho da imagem de referencia que gerou o vetor. Serve
            para auditar por que uma referencia esta puxando falso positivo.
    """

    marca: str
    similaridade: float
    referencia: str

    def __post_init__(self) -> None:
        """Valida marca preenchida e similaridade no intervalo do cosseno.

        Raises:
            CandidatoInvalidoError: Se a marca for vazia ou a similaridade
                estiver fora de [-1.0, 1.0].
        """
        if not self.marca:
            raise CandidatoInvalidoError("marca nao pode ser vazia")
        if not _SIMILARIDADE_MINIMA <= self.similaridade <= _SIMILARIDADE_MAXIMA:
            raise CandidatoInvalidoError(f"similaridade fora de [-1, 1]: {self.similaridade}")
