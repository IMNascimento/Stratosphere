"""Mede onde os limiares deveriam estar, usando imagens com marca conhecida.

--------------------------------------------------------------------------
POR QUE ISTO EXISTE
--------------------------------------------------------------------------
`config/settings.py` avisa que os limiares sao ponto de partida e precisam ser
recalibrados com dado proprio. Enquanto ninguem mede, quatro numeros decidem
tudo sem nunca terem visto a base:

    min_similarity 0.68   onde a similaridade comeca a valer
    max_similarity 0.97   onde ela satura
    confident_inliers 25  quantos pontos sao "geometria confirmada"
    orphan_max_similarity 0.88   abaixo disto vira orfao

O efeito medido de nao calibrar: uma regiao com similaridade 0.953 e o top-25
inteiro da mesma marca foi para revisao humana, porque a geometria rendeu 15
inliers em vez de 25.

--------------------------------------------------------------------------
COMO O ROTULO E OBTIDO
--------------------------------------------------------------------------
Pela pasta: `<raiz>/<marca>/arquivo.jpg` significa "esta imagem contem esta
marca". E rotulo por imagem, nao por caixa - entao uma regiao cuja marca do topo
bate com a pasta conta como **acerto provavel**, e uma que aponta outra marca
conta como **erro provavel**.

O "provavel" e honesto e importa: a imagem pode conter mais de uma marca, e
nesse caso uma regiao correta de outra marca entra como erro. Isso empurra a
distribuicao dos erros para cima, ou seja, os limiares sugeridos aqui sao
**conservadores** por construcao.

Typical usage:
    poetry run python tools/calibrate_thresholds.py --rotuladas dataset/contexto --banco indice
"""

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from application.ports.i_geometric_verifier import IGeometricVerifier  # noqa: E402
from config.settings import AppConfig  # noqa: E402
from domain.entities.analyzed_region import AnalyzedRegion  # noqa: E402
from infrastructure.container.container import _build_verifier  # noqa: E402
from infrastructure.detection.owlv2_detector import Owlv2Detector  # noqa: E402
from infrastructure.encoding.dinov2_encoder import Dinov2Encoder  # noqa: E402
from infrastructure.image.pillow_image_source import (  # noqa: E402
    ACCEPTED_EXTENSIONS,
    PillowImageSource,
)
from infrastructure.index.npz_reference_database import NpzReferenceDatabase  # noqa: E402
from shared.logging.logger import configure_logging, get_logger  # noqa: E402

log = get_logger("calibrate")

# Percentil da distribuicao dos ERROS usado para sugerir `min_similarity`:
# abaixo dele a evidencia do banco deveria contar como nula.
_NOISE_PERCENTILE = 90

# Percentil da distribuicao dos ACERTOS usado para sugerir `max_similarity` e
# `confident_inliers`: a partir dele a evidencia ja e maxima.
_SIGNAL_PERCENTILE = 75

# Percentil baixo dos ACERTOS usado para sugerir `orphan_max_similarity`: o
# orfao precisa disparar no rabo de baixo, e nao no meio da faixa normal.
_ORPHAN_PERCENTILE = 10


@dataclass
class Sample:
    """Uma regiao medida, ja rotulada como acerto ou erro provavel.

    Attributes:
        similarity: Similaridade do melhor candidato.
        consensus: Fracao do top-k que concorda com a marca do topo.
        margin: Vantagem sobre a rival mais proxima.
        inliers: Inliers do melhor veredito, ou -1 quando a geometria nao opinou.
        hit: Se a marca do topo bate com a pasta.
    """

    similarity: float
    consensus: float
    margin: float
    inliers: int
    hit: bool


@dataclass
class Collected:
    """Amostras acumuladas.

    Attributes:
        samples: Todas as regioes medidas.
        images: Quantas imagens foram processadas.
        skipped: Imagens descartadas pelo pre-filtro ou sem deteccao.
    """

    samples: list[Sample] = field(default_factory=list)
    images: int = 0
    skipped: int = 0


def main(arguments: list[str] | None = None) -> int:
    """Roda a medicao e imprime os limiares sugeridos.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 em sucesso, 1 quando a pasta rotulada nao existe.
    """
    options = _parser().parse_args(arguments)
    configure_logging("INFO")

    root = Path(options.rotuladas)
    if not root.is_dir():
        log.error("pasta rotulada nao existe: %s", root)
        return 1

    config = AppConfig()
    source = PillowImageSource(min_side_to_upscale=config.encoder.min_side_to_upscale)
    encoder = Dinov2Encoder(config.encoder, options.dispositivo, options.precisao)
    detector = Owlv2Detector(config.detector, options.dispositivo, options.precisao)
    # Mede o verificador que a pipeline realmente usa: a escala de inliers
    # muda com o matcher, e calibrar contra outro produziria limiar errado.
    verifier = _build_verifier(replace(config, device=options.dispositivo))
    database = NpzReferenceDatabase.load(Path(options.banco), encoder.signature())
    references_by_brand = database.references_by_brand()

    collected = Collected()
    files = sorted(_walk(root))
    log.info("%d imagens rotuladas em %s", len(files), root)

    for position, (path, brand) in enumerate(files, start=1):
        _measure(
            path,
            brand,
            config,
            source,
            encoder,
            detector,
            verifier,
            database,
            references_by_brand,
            collected,
        )
        if position % 10 == 0:
            log.info("  %d/%d", position, len(files))

    _report(collected, config)
    return 0


def _measure(  # noqa: PLR0913 - e um script de medicao, os componentes vem todos de fora
    path: Path,
    brand: str,
    config: AppConfig,
    source: PillowImageSource,
    encoder: Dinov2Encoder,
    detector: Owlv2Detector,
    verifier: IGeometricVerifier,
    database: NpzReferenceDatabase,
    references_by_brand: dict[str, int],
    collected: Collected,
) -> None:
    """Mede todas as regioes de uma imagem rotulada.

    Args:
        path: Arquivo a medir.
        brand: Marca esperada, vinda do nome da pasta.
        config: Configuracao completa.
        source: Acesso a imagem.
        encoder: Codificador.
        detector: Detector agnostico.
        verifier: Verificacao geometrica.
        database: Banco de referencia.
        references_by_brand: Contagem por marca, para o teto do consenso.
        collected: Acumulador, alterado no lugar.
    """
    try:
        image = source.load(path)
    except OSError as error:
        log.warning("ilegivel: %s (%s)", path.name, error)
        return

    detections = detector.detect(image)
    collected.images += 1
    if not detections:
        collected.skipped += 1
        return

    crops = [
        source.crop(image, detection.box, config.encoder.crop_margin, config.encoder.crop_side)
        for detection in detections
    ]
    vectors = encoder.encode(crops)

    for index, (detection, crop) in enumerate(zip(detections, crops, strict=True)):
        candidates = database.search(vectors[index], config.search.neighbors)
        region = AnalyzedRegion(identifier=f"{path.stem}-{index}", detection=detection)
        region = region.with_candidates(candidates)
        region = region.with_brand_references(references_by_brand.get(region.top_brand or "", 0))
        if config.geometry.enabled and candidates:
            region = region.with_verdicts(verifier.verify(crop, candidates))

        verdict = region.best_verdict
        collected.samples.append(
            Sample(
                similarity=region.top_similarity,
                consensus=region.brand_consensus,
                margin=region.margin,
                inliers=verdict.inliers if verdict is not None else -1,
                hit=region.top_brand == brand,
            )
        )


def _report(collected: Collected, config: AppConfig) -> None:
    """Imprime as distribuicoes e os limiares sugeridos.

    Args:
        collected: Amostras acumuladas.
        config: Configuracao atual, para comparar com o sugerido.
    """
    hits = [s for s in collected.samples if s.hit]
    misses = [s for s in collected.samples if not s.hit]
    print(f"\n{collected.images} imagens, {len(collected.samples)} regioes")
    print(f"  {len(hits)} acertos provaveis, {len(misses)} erros provaveis")
    if not hits or not misses:
        print("\nsem amostras suficientes dos dois lados - nada a sugerir")
        return

    print("\n--- SIMILARIDADE ---")
    _distribution("acertos", [s.similarity for s in hits])
    _distribution("erros  ", [s.similarity for s in misses])

    print("\n--- CONSENSO ---")
    _distribution("acertos", [s.consensus for s in hits])
    _distribution("erros  ", [s.consensus for s in misses])

    print("\n--- INLIERS (so onde a geometria opinou) ---")
    hit_inliers = [float(s.inliers) for s in hits if s.inliers >= 0]
    miss_inliers = [float(s.inliers) for s in misses if s.inliers >= 0]
    mute = sum(1 for s in hits if s.inliers < 0)
    if hit_inliers:
        _distribution("acertos", hit_inliers)
    if miss_inliers:
        _distribution("erros  ", miss_inliers)
    print(f"  geometria ficou muda em {mute} de {len(hits)} acertos")

    print("\n--- CONSENSO COMO REGRA SOZINHA ---")
    for cut in (0.60, 0.70, 0.80, 0.90, 1.00):
        acima_h = sum(1 for s in hits if s.consensus >= cut)
        acima_m = sum(1 for s in misses if s.consensus >= cut)
        total = acima_h + acima_m
        precisao = acima_h / total if total else 0.0
        cobertura = acima_h / len(hits)
        print(
            f"  consenso >= {cut:.2f}: {total:>4} regioes, "
            f"{precisao * 100:>5.1f}% certas, cobre {cobertura * 100:>5.1f}% dos acertos"
        )

    print("\n--- SUGESTAO ---")
    _suggest(
        "min_similarity",
        config.routing.min_similarity,
        float(np.percentile([s.similarity for s in misses], _NOISE_PERCENTILE)),
        f"percentil {_NOISE_PERCENTILE} dos erros",
    )
    _suggest(
        "max_similarity",
        config.routing.max_similarity,
        float(np.percentile([s.similarity for s in hits], _SIGNAL_PERCENTILE)),
        f"percentil {_SIGNAL_PERCENTILE} dos acertos",
    )
    if hit_inliers:
        _suggest(
            "confident_inliers",
            config.routing.confident_inliers,
            float(np.percentile(hit_inliers, _SIGNAL_PERCENTILE)),
            f"percentil {_SIGNAL_PERCENTILE} dos acertos",
        )
    _suggest(
        "orphan_max_similarity",
        config.routing.orphan_max_similarity,
        float(np.percentile([s.similarity for s in hits], _ORPHAN_PERCENTILE)),
        f"percentil {_ORPHAN_PERCENTILE} dos acertos",
    )


def _distribution(label: str, values: list[float]) -> None:
    """Imprime os percentis de uma distribuicao.

    Args:
        label: Nome da serie.
        values: Valores medidos.
    """
    p = np.percentile(values, [10, 25, 50, 75, 90])
    print(
        f"  {label}  n={len(values):<5} "
        f"p10={p[0]:.3f} p25={p[1]:.3f} mediana={p[2]:.3f} p75={p[3]:.3f} p90={p[4]:.3f}"
    )


def _suggest(name: str, current: float, suggested: float, basis: str) -> None:
    """Imprime a comparacao entre o valor atual e o sugerido.

    Args:
        name: Nome do parametro.
        current: Valor em uso.
        suggested: Valor que os dados indicam.
        basis: De onde saiu a sugestao.
    """
    arrow = "=" if abs(current - suggested) < 1e-3 else ("v" if suggested < current else "^")
    print(f"  {name:<24} hoje={current:<7.3f} {arrow}  sugerido={suggested:<7.3f} ({basis})")


def _walk(root: Path) -> Iterator[tuple[Path, str]]:
    """Percorre a arvore rotulada.

    Args:
        root: Raiz com o layout `<marca>/arquivo`.

    Yields:
        Tuplas `(caminho, marca)`.
    """
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        brand = folder.name.strip().lower()
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in ACCEPTED_EXTENSIONS:
                yield path, brand


def _parser() -> argparse.ArgumentParser:
    """Monta o analisador de argumentos.

    Returns:
        O analisador configurado.
    """
    parser = argparse.ArgumentParser(
        prog="calibrate_thresholds",
        description="Mede onde os limiares deveriam estar, com imagens de marca conhecida.",
    )
    parser.add_argument("--rotuladas", required=True, help="pasta <marca>/arquivo")
    parser.add_argument("--banco", default="indice", help="pasta do indice vetorial")
    parser.add_argument("--dispositivo", default="cuda:0", help="cuda:0 ou cpu")
    parser.add_argument("--precisao", default="float16", help="float16 ou float32")
    return parser


if __name__ == "__main__":
    sys.exit(main())
