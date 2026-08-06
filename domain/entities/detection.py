"""Entidade de deteccao produzida pelo detector agnostico de marca.

**Esta entidade nao tem campo `brand`, e nunca tera.** Toda a propriedade "marca
nova sem retreino" depende disso: se o detector puder responder qual marca e,
marca nova volta a exigir retreino e a arquitetura inteira perde o sentido. A
ausencia do campo e o que impede o acoplamento de voltar por descuido.

Typical usage:
    detection = Detection(box=Box(10, 20, 110, 80), confidence=0.31)
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import InvalidDetectionError
from domain.value_objects.box import Box


@dataclass(frozen=True)
class Detection:
    """Uma regiao onde o detector afirma haver marca grafica.

    Attributes:
        box: Regiao em coordenadas da imagem original.
        confidence: Confianca do detector, entre 0.0 e 1.0. **A escala e do
            detector, nao universal** - um valor de 0.3 pode ser alto num
            detector de vocabulario aberto e baixo num detector treinado.
            Comparar com limiar importado de outro modelo produz resultado sem
            sentido.
        concept: Qual prompt de conceito disparou, quando o detector informa.
            Util para descobrir que um conceito puxa quase tudo e outro nunca
            dispara.
    """

    box: Box
    confidence: float
    concept: str | None = None

    def __post_init__(self) -> None:
        """Valida a confianca no intervalo [0, 1].

        Raises:
            InvalidDetectionError: Se a confianca estiver fora de [0.0, 1.0].
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise InvalidDetectionError(f"confianca fora de [0, 1]: {self.confidence}")
