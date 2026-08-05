"""Entidade de deteccao produzida pelo detector agnostico de marca.

**Esta entidade nao tem campo `marca`, e nunca tera.** Toda a propriedade "marca
nova sem retreino" depende disso: se o detector puder responder qual marca e,
marca nova volta a exigir retreino e a arquitetura inteira perde o sentido. A
ausencia do campo e o que impede o acoplamento de voltar por descuido.

Typical usage:
    deteccao = Deteccao(caixa=Caixa(10, 20, 110, 80), confianca=0.31)
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import DeteccaoInvalidaError
from domain.value_objects.caixa import Caixa


@dataclass(frozen=True)
class Deteccao:
    """Uma regiao onde o detector afirma haver marca grafica.

    Attributes:
        caixa: Regiao em coordenadas da imagem original.
        confianca: Confianca do detector, entre 0.0 e 1.0. **A escala e do
            detector, nao universal** — um valor de 0.3 pode ser alto num
            detector de vocabulario aberto e baixo num detector treinado.
            Comparar com limiar importado de outro modelo produz resultado sem
            sentido.
        conceito: Qual prompt de conceito disparou, quando o detector informa.
            Util para descobrir que um conceito puxa quase tudo e outro nunca
            dispara.
    """

    caixa: Caixa
    confianca: float
    conceito: str | None = None

    def __post_init__(self) -> None:
        """Valida a confianca no intervalo [0, 1].

        Raises:
            DeteccaoInvalidaError: Se a confianca estiver fora de [0.0, 1.0].
        """
        if not 0.0 <= self.confianca <= 1.0:
            raise DeteccaoInvalidaError(f"confianca fora de [0, 1]: {self.confianca}")
