"""Montagem do grafo de dependencias.

**Este e o unico arquivo que instancia infraestrutura concreta.** Nenhuma outra
camada sabe que existe OWLv2, DINOv2 ou SIFT — todas falam com as portas. Trocar
qualquer adaptador e uma mudanca aqui e em mais lugar nenhum.

Tambem e o unico lugar que le variavel de ambiente, quando houver.

Typical usage:
    container = build_container(AppConfig(), Path("indice"))
    output = container.analyze_image.execute(command)
"""

from dataclasses import dataclass
from pathlib import Path

from application.ports.i_encoder import IEncoder
from application.ports.i_geometric_verifier import IGeometricVerifier
from application.use_cases.analyze_image import AnalyzeImageUseCase
from application.use_cases.audit_database import AuditDatabaseUseCase
from application.use_cases.build_database import BuildDatabaseUseCase
from config.settings import AppConfig
from domain.services.brand_corroboration_resolver import BrandCorroborationResolver
from domain.services.confusion_groups import ConfusionGroups
from domain.services.nested_region_resolver import NestedRegionResolver
from domain.services.queue_router import Calibration, EvidenceWeights, QueueRouter
from infrastructure.detection.owlv2_detector import Owlv2Detector
from infrastructure.encoding.dinov2_encoder import Dinov2Encoder
from infrastructure.encoding.siglip_encoder import SiglipEncoder
from infrastructure.geometry.lightglue_verifier import LightGlueVerifier
from infrastructure.geometry.sift_verifier import SiftVerifier
from infrastructure.image.pillow_annotator import PillowAnnotator
from infrastructure.image.pillow_image_source import PillowImageSource
from infrastructure.index.npz_reference_database import (
    METADATA_FILE,
    VECTORS_FILE,
    NpzDatabaseWriter,
    NpzReferenceDatabase,
)
from infrastructure.judge.qwen_judge import QwenJudge


@dataclass(frozen=True)
class Container:
    """Casos de uso prontos para uso, com as dependencias ja injetadas.

    Attributes:
        analyze_image: Pipeline completa para uma imagem. None quando o banco
            ainda nao existe — construir o banco nao exige banco.
        build_database: Pasta de referencias para indice vetorial.
        audit_database: Pares de marcas confundiveis. None sem banco.
        image_source: Exposto porque o entrypoint precisa listar pastas.
        annotator: Desenho das caixas sobre a imagem, para conferencia no olho.
            E saida de diagnostico e nao participa da pipeline.
    """

    build_database: BuildDatabaseUseCase
    image_source: PillowImageSource
    annotator: PillowAnnotator
    analyze_image: AnalyzeImageUseCase | None = None
    audit_database: AuditDatabaseUseCase | None = None


def build_container(config: AppConfig, database_path: Path) -> Container:
    """Monta o grafo completo de dependencias.

    O banco de referencia e carregado quando existe. Quando nao existe, os casos
    de uso que dependem dele ficam em None em vez de a montagem falhar — isso
    permite que `build_database` rode numa instalacao limpa, que e exatamente o
    primeiro comando que alguem executa.

    Args:
        config: Configuracao completa.
        database_path: Pasta do indice vetorial.

    Returns:
        O container com os casos de uso disponiveis.

    Raises:
        IncompatibleEncoderError: Se o banco existir mas tiver sido construido
            com outro codificador.
    """
    source = PillowImageSource(min_side_to_upscale=config.encoder.min_side_to_upscale)
    encoder = _build_encoder(config)
    build = BuildDatabaseUseCase(
        encoder=encoder, source=source, writer=NpzDatabaseWriter(), config=config
    )
    annotator = PillowAnnotator(source=source)
    container = Container(build_database=build, image_source=source, annotator=annotator)

    if not _database_exists(database_path):
        return container

    database = NpzReferenceDatabase.load(database_path, encoder.signature())
    analyze = AnalyzeImageUseCase(
        detector=Owlv2Detector(
            config=config.detector,
            device=config.device,
            precision=config.precision,
        ),
        encoder=encoder,
        verifier=_build_verifier(config),
        source=source,
        database=database,
        router=_build_router(config),
        resolver=NestedRegionResolver(
            containment=config.routing.nested_containment,
            same_brand_containment=config.routing.nested_same_brand_containment,
        ),
        corroborator=BrandCorroborationResolver(
            min_similarity=config.routing.corroboration_min_similarity,
            enabled=config.routing.corroboration_enabled,
        ),
        config=config,
        judge=(
            QwenJudge(config.judge, device=config.device, precision=config.precision)
            if config.judge.enabled
            else None
        ),
    )
    return Container(
        build_database=build,
        image_source=source,
        annotator=annotator,
        analyze_image=analyze,
        audit_database=AuditDatabaseUseCase(database=database, config=config),
    )


def _build_encoder(config: AppConfig) -> IEncoder:
    """Escolhe a implementacao do codificador.

    O default e o modelo com supervisao de texto: medido com o proprio banco,
    ele separa "mesma marca" de "mesma foto" com AUC 0.812 contra 0.515 do
    DINOv2, que codificava a superficie em vez da marca.

    Args:
        config: Configuracao completa.

    Returns:
        O codificador configurado em `EncoderConfig.backend`.

    Raises:
        ValueError: Se o nome do backend nao existir.
    """
    if config.encoder.backend == "dinov2":
        return Dinov2Encoder(
            config=config.encoder, device=config.device, precision=config.precision
        )
    if config.encoder.backend == "siglip":
        return SiglipEncoder(
            config=config.encoder, device=config.device, precision=config.precision
        )
    raise ValueError(
        f"backend {config.encoder.backend!r} desconhecido. Validos: 'siglip', 'dinov2'"
    )


def _build_verifier(config: AppConfig) -> IGeometricVerifier:
    """Escolhe a implementacao da verificacao geometrica.

    O default e o matcher aprendido: medido em 60 pares certos contra 60
    errados, o SIFT confirma **0%** deles sem deixar passar um errado, contra
    63% do LightGlue. O SIFT fica disponivel para comparacao, e porque nao
    depende de peso baixado.

    Args:
        config: Configuracao completa.

    Returns:
        O verificador configurado em `GeometryConfig.matcher`.

    Raises:
        ValueError: Se o nome do matcher nao existir.
    """
    if config.geometry.matcher == "sift":
        return SiftVerifier(config=config.geometry)
    if config.geometry.matcher == "lightglue":
        return LightGlueVerifier(config=config.geometry, device=config.device)
    raise ValueError(
        f"matcher {config.geometry.matcher!r} desconhecido. Validos: 'lightglue', 'sift'"
    )


def _build_router(config: AppConfig) -> QueueRouter:
    """Constroi o servico de dominio que decide as filas.

    Args:
        config: Configuracao completa.

    Returns:
        O roteador pronto.

    Raises:
        ValueError: Se os pesos configurados nao somarem 1 ou se algum limiar
            estiver invertido.
    """
    routing = config.routing
    return QueueRouter(
        weights=EvidenceWeights(
            similarity=routing.similarity_weight,
            consensus=routing.consensus_weight,
            margin=routing.margin_weight,
            geometry=routing.geometry_weight,
            detection=routing.detection_weight,
        ),
        calibration=Calibration(
            min_similarity=routing.min_similarity,
            max_similarity=routing.max_similarity,
            confident_margin=routing.confident_margin,
            confident_inliers=routing.confident_inliers,
            accept=routing.accept,
            reject=routing.reject,
            orphan_min_inliers=routing.orphan_min_inliers,
            orphan_max_similarity=routing.orphan_max_similarity,
            consensus_accept=routing.consensus_accept,
            consensus_min_agreeing=routing.consensus_min_agreeing,
            geometry_accept_inliers=routing.geometry_accept_inliers,
            informative_inliers=routing.informative_inliers,
        ),
        groups=ConfusionGroups(
            groups=config.confusion.groups,
            negatives=config.confusion.negatives,
            tie_margin=config.confusion.tie_margin,
        ),
    )


def _database_exists(path: Path) -> bool:
    """Verifica se ha um banco gravado no caminho.

    Args:
        path: Pasta do indice.

    Returns:
        True quando os dois arquivos do banco estao presentes.
    """
    return (path / VECTORS_FILE).exists() and (path / METADATA_FILE).exists()
