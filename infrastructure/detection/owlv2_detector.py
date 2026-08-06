"""Detector de vocabulario aberto - encontra ONDE ha marca grafica.

Os prompts sao **conceitos** (`logo`, `emblem`, `wordmark`), nunca nomes de
marca. Nome de marca no prompt reintroduz exatamente o acoplamento que a
arquitetura existe para eliminar: marca nova voltaria a exigir mudanca aqui.

Duas caracteristicas deste tipo de detector, medidas e nao supostas, mudam como
ele deve ser configurado:

**A escala de confianca e comprimida e nao separa bem.** A distribuicao de
confianca em regiao com logo e em regiao sem logo se sobrepoe quase inteiramente.
Consequencia pratica: subir o limiar corta logo real na mesma taxa que corta
ruido - ele nao e um bom filtro de qualidade, so um controle de volume.

**Por isso o limiar padrao e baixo.** O detector so precisa acertar *onde*; a
busca vetorial e a verificacao geometrica filtram *quem*. Limiar alto troca
recall - irrecuperavel, porque a regiao nunca chega ao codificador - por
precisao, que e recuperavel adiante.

Typical usage:
    detector = Owlv2Detector(config, device="cuda:0", precision="float16")
    detector.prepare()
    detections = detector.detect(image)
"""

from typing import Any

from application.ports.i_detector import IDetector
from application.ports.i_image_source import RgbImage
from config.settings import DetectorConfig
from domain.entities.detection import Detection
from domain.exceptions.domain_exceptions import InvalidBoxError
from domain.value_objects.box import Box


class Owlv2Detector(IDetector):
    """Detector agnostico de marca sobre um modelo de vocabulario aberto."""

    def __init__(self, config: DetectorConfig, device: str, precision: str) -> None:
        """Guarda a configuracao sem carregar pesos.

        Args:
            config: Parametros de deteccao.
            device: Onde o modelo roda, por exemplo `cuda:0` ou `cpu`.
            precision: Precisao numerica, por exemplo `float16`.
        """
        self._config = config
        self._device = device
        self._precision = precision
        self._model: Any = None
        self._processor: Any = None
        # Modelos de vocabulario aberto respondem melhor a frase que a palavra solta.
        self._queries = [f"a photo of a {concept}" for concept in config.concepts]

    def prepare(self) -> None:
        """Carrega o modelo e o processador. Idempotente."""
        if self._model is not None:
            return

        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._processor = Owlv2Processor.from_pretrained(self._config.identifier)
        model: Any = Owlv2ForObjectDetection.from_pretrained(
            self._config.identifier, dtype=getattr(torch, self._precision)
        )
        model = model.to(self._device)
        self._model = model.eval()
        self._torch = torch

    def detect(self, image: RgbImage) -> tuple[Detection, ...]:
        """Encontra as regioes da imagem que contem marca grafica.

        Args:
            image: Imagem completa em RGB.

        Returns:
            Deteccoes em coordenadas da imagem original, filtradas e suprimidas,
            em ordem decrescente de confianca.
        """
        self.prepare()

        width, height = image.size
        shrunk, scale = self._shrink(image)
        raw = self._infer(shrunk)

        if scale != 1.0:
            inverse = 1.0 / scale
            raw = [
                (self._rescale(box, inverse), confidence, concept)
                for box, confidence, concept in raw
            ]

        return self._post_process(raw, width, height)

    # -- etapas internas ---------------------------------------------------

    def _shrink(self, image: RgbImage) -> tuple[RgbImage, float]:
        """Reduz a imagem para a resolucao de inferencia.

        Reduzir demais faz simbolos pequenos deixarem de existir antes de
        qualquer modelo opinar - por isso o padrao e alto.

        Args:
            image: Imagem original.

        Returns:
            Tupla `(imagem_reduzida, escala)`. A escala traz as caixas de volta
            para as coordenadas originais.
        """
        from PIL import Image

        width, height = image.size
        longest_side = max(width, height)
        target = self._config.inference_side
        if longest_side <= target:
            return image, 1.0

        scale = target / longest_side
        shrunk = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BICUBIC,
        )
        return shrunk, scale

    def _infer(
        self, image: RgbImage
    ) -> list[tuple[tuple[float, float, float, float], float, str | None]]:
        """Roda o modelo e devolve caixas brutas na escala da imagem recebida.

        Args:
            image: Imagem ja reduzida.

        Returns:
            Lista de `(caixa_xyxy, confianca, conceito)`.
        """
        inputs = self._processor(text=[self._queries], images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)

        width, height = image.size
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=self._torch.tensor([[height, width]], device=self._device),
            threshold=self._config.confidence_threshold,
        )[0]

        raw = []
        for box, confidence, label in zip(
            results["boxes"], results["scores"], results["labels"], strict=True
        ):
            coordinates = tuple(float(value) for value in box.tolist())
            index = int(label)
            concept = self._config.concepts[index] if index < len(self._config.concepts) else None
            raw.append((coordinates, float(confidence), concept))
        return raw  # type: ignore[return-value]

    @staticmethod
    def _rescale(
        box: tuple[float, float, float, float], factor: float
    ) -> tuple[float, float, float, float]:
        """Multiplica as coordenadas pelo fator de volta a escala original.

        Args:
            box: Coordenadas xyxy na escala de inferencia.
            factor: Inverso da escala aplicada na reducao.

        Returns:
            Coordenadas na escala da imagem original.
        """
        return (box[0] * factor, box[1] * factor, box[2] * factor, box[3] * factor)

    def _post_process(
        self,
        raw: list[tuple[tuple[float, float, float, float], float, str | None]],
        width: int,
        height: int,
    ) -> tuple[Detection, ...]:
        """Filtra por area e lado, suprime sobreposicoes e limita a quantidade.

        Args:
            raw: Caixas na escala original.
            width: Largura da imagem.
            height: Altura da imagem.

        Returns:
            Deteccoes validas em ordem decrescente de confianca.
        """
        image_area = float(width * height)
        accepted: list[Detection] = []

        for coordinates, confidence, concept in raw:
            box = self._clamp(coordinates, width, height)
            if box is None:
                continue
            fraction = box.area / max(1.0, image_area)
            if fraction < self._config.min_relative_area:
                continue
            if fraction > self._config.max_relative_area:
                continue
            if box.shorter_side < self._config.min_side:
                continue
            accepted.append(Detection(box=box, confidence=min(1.0, confidence), concept=concept))

        accepted.sort(key=lambda detection: detection.confidence, reverse=True)
        suppressed = self._suppress(accepted)
        return tuple(suppressed[: self._config.max_regions])

    @staticmethod
    def _clamp(
        coordinates: tuple[float, float, float, float], width: int, height: int
    ) -> Box | None:
        """Arredonda e prende as coordenadas aos limites da imagem.

        Args:
            coordinates: Caixa xyxy, possivelmente fora dos limites.
            width: Largura da imagem.
            height: Altura da imagem.

        Returns:
            A caixa valida, ou None se ela degenerar apos o corte.
        """
        x1, y1, x2, y2 = coordinates
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        try:
            return Box(
                x1=int(max(0, min(width - 1, round(x1)))),
                y1=int(max(0, min(height - 1, round(y1)))),
                x2=int(max(1, min(width, round(x2)))),
                y2=int(max(1, min(height, round(y2)))),
            )
        except InvalidBoxError:
            return None

    def _suppress(self, detections: list[Detection]) -> list[Detection]:
        """Remove caixas sobrepostas, **sem olhar o conceito que as gerou**.

        A supressao e agnostica de proposito: modelos de vocabulario aberto
        disparam varios conceitos sobre o mesmo pixel - `logo` e `wordmark` no
        mesmo simbolo, por exemplo. Suprimir por conceito deixaria as duas
        caixas passarem, e a mesma regiao seria codificada e cobrada duas vezes.

        Args:
            detections: Deteccoes ja ordenadas por confianca decrescente.

        Returns:
            As que sobreviveram, na mesma ordem.
        """
        if len(detections) <= 1:
            return detections

        kept: list[Detection] = []
        for candidate in detections:
            overlaps = any(
                candidate.box.intersection_over_union(keeper.box) > self._config.suppression_iou
                for keeper in kept
            )
            if not overlaps:
                kept.append(candidate)
        return kept
