"""Mede o juiz visual sobre dados rotulados e sugere os cortes de decisao.

O juiz nao devolve "sim" ou "nao": devolve a probabilidade que o modelo deu a
resposta "sim". **Essa probabilidade nao vem calibrada.** Modelo de instrucao
tende a concordar com o que se pergunta, entao um corte ingenuo em 0.5 aprova
tudo — medido: 0.606 a 0.673 em regiao que era acerto e em regiao que era erro,
com o mesmo "sim" nas duas.

O que tem sinal e a **ordem**, nao o valor. Este script mede onde estao as duas
distribuicoes — acerto e erro — e sugere os dois cortes que o juiz usa:

    p >= confirm_above   -> confirma, vai para aceite automatico
    p <= deny_below      -> nega, vai para descarte
    entre os dois        -> se abstem, e uma pessoa decide

Os cortes saem de percentil, e nao de arredondamento bonito: `confirm_above` fica
acima de quase todo erro, e `deny_below` abaixo de quase todo acerto. O meio e
grande de proposito — ele e a fila humana, e encolhe-lo sem medir e trocar
revisao por erro silencioso.

O rotulo vem da pasta, igual ao `calibrate_thresholds.py`:
`<raiz>/<marca>/arquivo.jpg` significa "esta imagem contem esta marca".

Typical usage:
    poetry run python tools/calibrate_judge.py --rotuladas dataset/por_marca --banco indice
"""

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import AppConfig
from domain.entities.analyzed_region import AnalyzedRegion
from infrastructure.detection.owlv2_detector import Owlv2Detector
from infrastructure.encoding.dinov2_encoder import Dinov2Encoder
from infrastructure.environment.env_settings import apply_env_overrides, load_env_file
from infrastructure.image.pillow_image_source import PillowImageSource
from infrastructure.index.npz_reference_database import NpzReferenceDatabase
from infrastructure.judge.qwen_judge import QwenJudge
from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere.calibrate_judge")

_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass
class Collected:
    """Probabilidades medidas, separadas por rotulo.

    Attributes:
        hits: Probabilidade de "sim" onde a marca do topo batia com a pasta.
        errors: Probabilidade de "sim" onde a marca do topo era outra.
        images: Imagens processadas.
    """

    hits: list[float] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    images: int = 0


def main(arguments: list[str] | None = None) -> int:
    """Roda a medicao e imprime os cortes sugeridos.

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

    load_env_file()
    config = apply_env_overrides(AppConfig())
    source = PillowImageSource(min_side_to_upscale=config.encoder.min_side_to_upscale)
    encoder = Dinov2Encoder(config.encoder, options.dispositivo, options.precisao)
    detector = Owlv2Detector(config.detector, options.dispositivo, options.precisao)
    database = NpzReferenceDatabase.load(Path(options.banco), encoder.signature())
    judge = QwenJudge(config.judge, options.dispositivo, options.precisao)

    collected = Collected()
    files = sorted(_walk(root))
    if options.limite and options.limite < len(files):
        # Passo constante, e nao os primeiros N: a lista vem ordenada por marca,
        # entao cortar pelo comeco mediria tres marcas e chamaria de amostra.
        step = len(files) / options.limite
        files = [files[int(i * step)] for i in range(options.limite)]
    log.info("%d imagens rotuladas em %s", len(files), root)

    for position, (path, brand) in enumerate(files, start=1):
        _measure(path, brand, config, source, encoder, detector, database, judge, collected)
        if position % 10 == 0:
            log.info("  %d/%d — %d acertos, %d erros", position, len(files),
                     len(collected.hits), len(collected.errors))

    _report(collected, config)
    return 0


def _measure(  # noqa: PLR0913 - e um script de medicao, os componentes vem todos de fora
    path: Path,
    brand: str,
    config: AppConfig,
    source: PillowImageSource,
    encoder: Dinov2Encoder,
    detector: Owlv2Detector,
    database: NpzReferenceDatabase,
    judge: QwenJudge,
    collected: Collected,
) -> None:
    """Pede o parecer do juiz sobre cada regiao de uma imagem rotulada.

    Args:
        path: Arquivo a medir.
        brand: Marca esperada, vinda do nome da pasta.
        config: Configuracao completa.
        source: Acesso a imagem.
        encoder: Codificador.
        detector: Detector agnostico.
        database: Banco de referencia.
        judge: Juiz visual.
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
        return

    crops = [
        source.crop(image, detection.box, config.encoder.crop_margin, config.encoder.crop_side)
        for detection in detections
    ]
    vectors = encoder.encode(crops)

    for index, (detection, crop) in enumerate(zip(detections, crops, strict=True)):
        candidates = database.search(vectors[index], config.search.neighbors)
        if not candidates:
            continue
        region = AnalyzedRegion(identifier=f"{path.stem}-{index}", detection=detection)
        region = region.with_candidates(candidates)
        top = region.candidates[0]

        try:
            reference = source.load(Path(top.reference))
            judge._prepare()  # noqa: SLF001 - medicao usa a probabilidade crua, nao o parecer
            probability = judge._probability_of_yes(crop, reference, top.brand)  # noqa: SLF001
        except (OSError, RuntimeError) as error:
            log.warning("juiz falhou em %s-%d: %s", path.stem, index, error)
            continue

        target = collected.hits if top.brand == brand else collected.errors
        target.append(probability)


def _report(collected: Collected, config: AppConfig) -> None:
    """Imprime as distribuicoes e os cortes sugeridos.

    Args:
        collected: Medicoes acumuladas.
        config: Configuracao, para mostrar o valor em uso hoje.
    """
    print(f"\n=== juiz {config.judge.model} ===")
    print(f"  {collected.images} imagens, {len(collected.hits)} acertos, "
          f"{len(collected.errors)} erros")
    if not collected.hits or not collected.errors:
        print("  amostra insuficiente — sem acerto ou sem erro, nao da para calibrar")
        return

    _distribution("acertos", collected.hits)
    _distribution("erros  ", collected.errors)
    print(f"\n  separacao das medias: {_mean(collected.hits) - _mean(collected.errors):+.4f}")
    print(f"  AUC (chance de ordenar certo): {_auc(collected.hits, collected.errors):.3f}")

    print("\n  precisao por corte de confirmacao:")
    for cut in (0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95):
        approved_hits = sum(1 for v in collected.hits if v >= cut)
        approved_errors = sum(1 for v in collected.errors if v >= cut)
        total = approved_hits + approved_errors
        if total == 0:
            continue
        share = approved_hits / len(collected.hits)
        print(f"    p >= {cut:.2f}   {approved_hits:>3} acertos, {approved_errors:>3} erros"
              f"   precisao {approved_hits / total:.3f}   cobre {share:.1%} dos acertos")

    print("\n  precisao por corte de negacao:")
    for cut in (0.30, 0.40, 0.50, 0.60, 0.65, 0.70):
        denied_errors = sum(1 for v in collected.errors if v <= cut)
        denied_hits = sum(1 for v in collected.hits if v <= cut)
        total = denied_hits + denied_errors
        if total == 0:
            continue
        share = denied_errors / len(collected.errors)
        print(f"    p <= {cut:.2f}   {denied_errors:>3} erros, {denied_hits:>3} acertos"
              f"   precisao {denied_errors / total:.3f}   cobre {share:.1%} dos erros")

    print(f"\n  sugestao: confirm_above = {_percentile(collected.errors, 0.98):.3f} "
          f"(p98 dos erros)")
    print(f"            deny_below    = {_percentile(collected.hits, 0.02):.3f} "
          f"(p2 dos acertos)")


def _distribution(label: str, values: list[float]) -> None:
    """Imprime uma serie em percentis.

    Args:
        label: Nome da serie.
        values: Valores medidos.
    """
    print(f"  {label}  n={len(values):<4} min={min(values):.3f} p10={_percentile(values, 0.10):.3f} "
          f"mediana={_percentile(values, 0.50):.3f} p90={_percentile(values, 0.90):.3f} "
          f"max={max(values):.3f}")


def _mean(values: list[float]) -> float:
    """Media simples.

    Args:
        values: Valores.

    Returns:
        A media.
    """
    return sum(values) / len(values)


def _percentile(values: list[float], fraction: float) -> float:
    """Percentil por posicao, sem interpolacao.

    Args:
        values: Valores, em qualquer ordem.
        fraction: Fracao entre 0.0 e 1.0.

    Returns:
        O valor naquela posicao.
    """
    ordered = sorted(values)
    position = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[position]


def _auc(hits: list[float], errors: list[float]) -> float:
    """Chance de um acerto sorteado pontuar acima de um erro sorteado.

    E a metrica honesta aqui: ela mede a ordem, e nao depende de onde o corte
    esta. 0.5 e moeda; 1.0 e separacao perfeita.

    Args:
        hits: Probabilidades das regioes que eram acerto.
        errors: Probabilidades das regioes que eram erro.

    Returns:
        Area sob a curva ROC.
    """
    above = sum(1 for h in hits for e in errors if h > e)
    tied = sum(1 for h in hits for e in errors if h == e)
    return (above + 0.5 * tied) / (len(hits) * len(errors))


def _walk(root: Path) -> Iterator[tuple[Path, str]]:
    """Percorre a arvore rotulada.

    Args:
        root: Pasta com uma subpasta por marca.

    Yields:
        Pares `(arquivo, marca)`.
    """
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
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
        description="mede o juiz visual e sugere os cortes de confirmacao e negacao"
    )
    parser.add_argument("--rotuladas", required=True, help="pasta <marca>/arquivo")
    parser.add_argument("--banco", default="indice", help="pasta do indice vetorial")
    parser.add_argument("--dispositivo", default="cuda:0")
    parser.add_argument("--precisao", default="float16")
    parser.add_argument("--limite", type=int, default=0, help="maximo de imagens, 0 = todas")
    return parser


if __name__ == "__main__":
    sys.exit(main())
