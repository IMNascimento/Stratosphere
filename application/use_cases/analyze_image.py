"""Caso de uso que atravessa a pipeline inteira para uma imagem.

Orquestra, nao decide. Toda regra de negocio esta em `QueueRouter`; aqui ha
apenas a sequencia das camadas e o cuidado de nao gastar camada cara com o que a
camada barata ja descartou.

A ordem importa e nao e arbitraria: cada camada e mais cara que a anterior, e so
ve o que a anterior deixou passar. A verificacao geometrica, em particular, so
roda nos melhores candidatos de cada regiao — se rodasse em tudo, seria a camada
dominante do custo.

Typical usage:
    use_case = AnalyzeImageUseCase(detector, encoder, verifier, source, database, router, config)
    output = use_case.execute(AnalyzeImageCommand(path))
"""

from application.dtos.analysis import AnalysisOutput, AnalyzeImageCommand, RegionOutput
from application.ports.i_detector import IDetector
from application.ports.i_encoder import IEncoder
from application.ports.i_geometric_verifier import IGeometricVerifier
from application.ports.i_image_source import IImageSource, RgbImage
from config.settings import AppConfig
from domain.entities.analyzed_region import AnalyzedRegion
from domain.repositories.i_reference_database import IReferenceDatabase
from domain.services.queue_router import QueueRouter


class AnalyzeImageUseCase:
    """Executa pre-filtro, deteccao, codificacao, busca, geometria e roteamento."""

    def __init__(
        self,
        detector: IDetector,
        encoder: IEncoder,
        verifier: IGeometricVerifier,
        source: IImageSource,
        database: IReferenceDatabase,
        router: QueueRouter,
        config: AppConfig,
    ) -> None:
        """Recebe as portas e o servico de dominio ja construidos.

        Args:
            detector: Camada que encontra onde ha marca.
            encoder: Camada que transforma regiao em vetor.
            verifier: Camada que confirma se e o mesmo desenho.
            source: Acesso a imagem e recorte.
            database: Banco de referencia consultado pela busca vetorial.
            router: Servico de dominio que decide a fila.
            config: Parametros de todas as camadas.
        """
        self._detector = detector
        self._encoder = encoder
        self._verifier = verifier
        self._source = source
        self._database = database
        self._router = router
        self._config = config

    def execute(self, command: AnalyzeImageCommand) -> AnalysisOutput:
        """Analisa uma imagem e devolve a decisao de cada regiao encontrada.

        Args:
            command: Pedido com o caminho da imagem.

        Returns:
            A analise completa. Quando o pre-filtro descarta a imagem,
            `discarded_by` explica o motivo e `regions` vem vazio.

        Raises:
            OSError: Se a imagem nao puder ser lida.
        """
        image = self._source.load(command.path)

        discard_reason = self._discard(image)
        if discard_reason is not None:
            return AnalysisOutput(path=command.path, discarded_by=discard_reason)

        detections = self._detector.detect(image)
        if not detections:
            return AnalysisOutput(path=command.path, discarded_by=None)

        crops = self._crop_all(image, detections)
        vectors = self._encoder.encode(crops)

        regions: list[RegionOutput] = []
        for index, (detection, crop) in enumerate(zip(detections, crops, strict=True)):
            candidates = self._database.search(vectors[index], self._config.search.neighbors)

            region = AnalyzedRegion(
                identifier=f"{command.path.stem}-{index:03d}",
                detection=detection,
            ).with_candidates(candidates)

            if self._config.geometry.enabled and candidates:
                region = region.with_verdicts(self._verifier.verify(crop, candidates))

            regions.append(self._to_output(region))

        return AnalysisOutput(path=command.path, discarded_by=None, regions=tuple(regions))

    def _discard(self, image: RgbImage) -> str | None:
        """Decide se a imagem nao merece uma passada de detector.

        Deliberadamente permissivo: o recall do detector e o teto do sistema
        inteiro, e o que se descarta aqui nunca mais volta.

        Args:
            image: Imagem carregada.

        Returns:
            Motivo do descarte, ou None se a imagem deve ser processada.
        """
        config = self._config.pre_filter
        if not config.enabled:
            return None

        width, height = self._source.dimensions(image)
        if min(width, height) < config.min_side:
            return f"lado menor que {config.min_side}px — miniatura ou icone"

        density = self._source.edge_density(image)
        if density < config.min_edge_density:
            return (
                f"densidade de bordas {density:.4f} abaixo de "
                f"{config.min_edge_density} — imagem sem estrutura"
            )
        return None

    def _crop_all(self, image: RgbImage, detections: tuple) -> list[RgbImage]:  # type: ignore[type-arg]
        """Recorta todas as regioes detectadas de uma vez.

        Args:
            image: Imagem de origem.
            detections: Regioes encontradas pelo detector.

        Returns:
            Recortes na mesma ordem das deteccoes.
        """
        config = self._config.encoder
        return [
            self._source.crop(
                image,
                detection.box,
                config.crop_margin,
                config.crop_side,
            )
            for detection in detections
        ]

    def _to_output(self, region: AnalyzedRegion) -> RegionOutput:
        """Converte a regiao analisada e sua decisao no DTO de saida.

        Args:
            region: Regiao com candidatos e vereditos ja reunidos.

        Returns:
            O DTO correspondente, com os sinais que justificam a decisao.
        """
        decision = self._router.route(region)
        verdict = region.best_verdict
        box = region.detection.box
        return RegionOutput(
            identifier=region.identifier,
            box=(box.x1, box.y1, box.x2, box.y2),
            queue=decision.queue,
            brand=decision.brand,
            score=decision.score,
            similarity=region.top_similarity,
            margin=region.margin,
            inliers=verdict.inliers if verdict is not None else 0,
            reasons=decision.reasons,
        )
