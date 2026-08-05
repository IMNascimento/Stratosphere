"""Contrato do detector agnostico de marca — a camada que responde ONDE.

**A regra que nao pode ser quebrada:** uma implementacao desta porta responde
apenas *onde ha marca grafica*, nunca *qual marca e*. `Detection` nao tem campo
`brand`, e e essa ausencia que sustenta a propriedade "marca nova em minutos,
sem retreino".

Consequencia pratica para detectores de vocabulario aberto: os prompts sao
**conceitos** (`logo`, `emblem`, `wordmark`), jamais nomes (`nike`, `itau`).
Prompt com nome de marca faz marca nova voltar a exigir mudanca no detector.

Typical usage:
    detections = detector.detect(image)
"""

from abc import ABC, abstractmethod

from application.ports.i_image_source import RgbImage
from domain.entities.detection import Detection


class IDetector(ABC):
    """Detector de regioes que contem marca grafica, agnostico de marca."""

    @abstractmethod
    def detect(self, image: RgbImage) -> tuple[Detection, ...]:
        """Encontra as regioes da imagem que contem marca grafica.

        Args:
            image: Imagem completa, ja carregada em RGB.

        Returns:
            Deteccoes em coordenadas da imagem **original**, ja reescaladas da
            resolucao de inferencia, filtradas por area e lado minimos, e
            passadas por supressao agnostica de conceito. Ordenadas por
            confianca decrescente. Tupla vazia quando nao ha nada.
        """
        ...

    @abstractmethod
    def prepare(self) -> None:
        """Carrega os pesos do modelo. Idempotente.

        Separado da construcao porque carregar pesos e caro e a instancia e
        reutilizada entre imagens.
        """
        ...
