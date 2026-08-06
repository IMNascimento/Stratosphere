"""Roda as camadas caras uma vez e grava os sinais crus de cada regiao.

**Existe para tornar a iteracao barata.** Detectar, codificar, buscar e verificar
custa dezenas de minutos de GPU por passada; mudar um peso do roteador custa
nada. Misturar as duas coisas num script so significa pagar a GPU de novo a cada
limiar que se quer experimentar.

Entao aqui a parte cara roda uma vez e vira um arquivo. Depois o
`sweep_routing.py` explora quantas configuracoes de roteamento quiser, em
segundos, sobre esse mesmo arquivo - e usando o `QueueRouter` de verdade, nao
uma reimplementacao dele.

Trocar o matcher geometrico **exige** um dump novo: os inliers sao produzidos
aqui. Trocar peso, limiar ou regra de fila nao exige.

O rotulo vem da pasta: `<raiz>/<marca>/arquivo.jpg` significa "esta imagem
contem esta marca". E rotulo por imagem, nao por caixa - ver `sweep_routing.py`
para o que isso implica na leitura das metricas.

Typical usage:
    poetry run python tools/dump_signals.py --rotuladas dataset/por_marca --saida sinais.jsonl
"""

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from config.settings import AppConfig
from infrastructure.container.container import _build_verifier
from infrastructure.detection.owlv2_detector import Owlv2Detector
from infrastructure.encoding.dinov2_encoder import Dinov2Encoder
from infrastructure.environment.env_settings import apply_env_overrides, load_env_file
from infrastructure.image.pillow_image_source import PillowImageSource
from infrastructure.index.npz_reference_database import NpzReferenceDatabase
from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere.dump_signals")

_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def main(arguments: list[str] | None = None) -> int:
    """Grava os sinais de cada regiao das imagens rotuladas.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 em sucesso, 1 quando a pasta rotulada nao existe.
    """
    options = _parser().parse_args(arguments)
    configure_logging("INFO")

    root = Path(options.rotuladas)
    if not root.is_dir():
        log.error("pasta de imagens nao existe: %s", root)
        return 1

    load_env_file()
    config = apply_env_overrides(AppConfig())
    config = replace(config, device=options.dispositivo, precision=options.precisao)
    if options.matcher:
        config = replace(config, geometry=replace(config.geometry, matcher=options.matcher))
    if options.lado:
        config = replace(config, geometry=replace(config.geometry, match_side=options.lado))

    source = PillowImageSource(min_side_to_upscale=config.encoder.min_side_to_upscale)
    encoder = Dinov2Encoder(config.encoder, config.device, config.precision)
    detector = Owlv2Detector(config.detector, config.device, config.precision)
    verifier = _build_verifier(config)
    database = NpzReferenceDatabase.load(Path(options.banco), encoder.signature())
    references_by_brand = database.references_by_brand()

    files = sorted(_walk(root))
    if options.limite and options.limite < len(files):
        # Passo constante, e nao os primeiros N: a lista vem ordenada por marca,
        # entao cortar pelo comeco mediria tres marcas e chamaria de amostra.
        step = len(files) / options.limite
        files = [files[int(i * step)] for i in range(options.limite)]

    destination = Path(options.saida)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "%d imagens, matcher %s, lado %d -> %s",
        len(files),
        config.geometry.matcher,
        config.geometry.match_side,
        destination,
    )

    regions = 0
    with destination.open("w", encoding="utf-8") as stream:
        for position, (path, brand) in enumerate(files, start=1):
            for record in _measure(
                path, brand, config, source, encoder, detector, verifier, database,
                references_by_brand,
            ):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                regions += 1
            if position % 20 == 0:
                log.info("  %d/%d imagens, %d regioes", position, len(files), regions)

    log.info("%d regioes gravadas em %s", regions, destination)
    return 0


def _measure(  # noqa: PLR0913 - e um script de medicao, os componentes vem todos de fora
    path: Path,
    brand: str,
    config: AppConfig,
    source: PillowImageSource,
    encoder: Dinov2Encoder,
    detector: Owlv2Detector,
    verifier: object,
    database: NpzReferenceDatabase,
    references_by_brand: dict[str, int],
) -> Iterator[dict[str, object]]:
    """Produz um registro por regiao de uma imagem rotulada.

    Args:
        path: Arquivo a medir.
        brand: Marca esperada, vinda do nome da pasta.
        config: Configuracao completa.
        source: Acesso a imagem.
        encoder: Codificador.
        detector: Detector agnostico.
        verifier: Verificacao geometrica ja escolhida.
        database: Banco de referencia.
        references_by_brand: Contagem por marca, para o teto do consenso.

    Yields:
        Dicionarios prontos para serializar.
    """
    try:
        image = source.load(path)
    except OSError as error:
        log.warning("ilegivel: %s (%s)", path.name, error)
        return

    detections = detector.detect(image)
    if not detections:
        return

    crops = [
        source.crop(image, detection.box, config.encoder.crop_margin, config.encoder.crop_side)
        for detection in detections
    ]
    vectors = encoder.encode(crops)

    for index, (detection, crop) in enumerate(zip(detections, crops, strict=True)):
        candidates = database.search(vectors[index], config.search.neighbors)
        verdicts = verifier.verify(crop, candidates) if candidates else ()  # type: ignore[attr-defined]
        top = candidates[0].brand if candidates else None
        yield {
            "image": str(path),
            "expected": brand,
            "index": index,
            "detection_score": detection.confidence,
            "box": [detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2],
            "brand_references": references_by_brand.get(top or "", 0),
            "candidates": [
                {"brand": c.brand, "similarity": c.similarity, "reference": c.reference}
                for c in candidates
            ],
            "verdicts": [
                {"brand": v.brand, "inliers": v.inliers, "matches": v.matches,
                 "confirms": v.confirms, "reason": v.reason}
                for v in verdicts
            ],
        }


def _walk(root: Path) -> Iterator[tuple[Path, str]]:
    """Percorre as imagens, com ou sem rotulo por pasta.

    Duas formas de organizacao, porque o rotulo pode vir de dois lugares. Com
    subpastas, `<raiz>/<marca>/arquivo` da o rotulo por imagem. Sem subpastas, o
    rotulo vem do COCO e e resolvido depois, por caixa - que e melhor, e por
    isso o campo fica vazio aqui em vez de receber um palpite.

    Args:
        root: Pasta com uma subpasta por marca, ou pasta plana de imagens.

    Yields:
        Pares `(arquivo, marca)`. Marca vazia quando nao ha rotulo por pasta.
    """
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _EXTENSIONS:
                yield path, ""
        return

    for folder in folders:
        brand = folder.name.strip().lower()
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in _EXTENSIONS:
                yield path, brand


def _parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos.

    Returns:
        O parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="grava os sinais crus de cada regiao para iterar o roteamento sem GPU"
    )
    parser.add_argument("--rotuladas", required=True, help="pasta <marca>/arquivo")
    parser.add_argument("--banco", default="indice", help="pasta do indice vetorial")
    parser.add_argument("--saida", required=True, help="arquivo .jsonl de destino")
    parser.add_argument("--matcher", default="", help="lightglue ou sift; vazio usa o default")
    parser.add_argument("--lado", type=int, default=0, help="match_side; 0 usa o default")
    parser.add_argument("--dispositivo", default="cuda:0")
    parser.add_argument("--precisao", default="float16")
    parser.add_argument("--limite", type=int, default=0, help="maximo de imagens, 0 = todas")
    return parser


if __name__ == "__main__":
    sys.exit(main())
