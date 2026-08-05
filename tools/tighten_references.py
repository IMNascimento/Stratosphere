"""Reenquadra as referencias no logo, usando o proprio detector da pipeline.

--------------------------------------------------------------------------
POR QUE ISTO EXISTE
--------------------------------------------------------------------------
Referencia que inclui a placa em volta ensina a placa, nao a marca. Medido no
banco real, no par `amazon x azul` recortado da MESMA foto de backdrop:

    recorte como esta (placa + moldura + fundo) ... 0.903
    tirando o fundo e a moldura ................... 0.801
    so o wordmark ................................. 0.630

Duas marcas diferentes a 0.903 nao e erro de rotulo: e o codificador
descrevendo *placa clara arredondada com wordmark ao centro sobre fundo
escuro*. Cor quase nao entra nessa conta — a mesma imagem em tons de cinza fica
a 0.929 do original, e com o matiz girado de laranja para azul, a 0.965. O que
separa marca de marca e a forma do que sobra no quadro.

**E ha um piso.** Cortando a 40% a similaridade do par SOBE de novo, para 0.699:
o corte comeu letras, e fragmento mutilado volta a parecer com outro fragmento
mutilado. Por isso este script nao corta por fracao fixa — ele usa a caixa do
detector, que segue a marca inteira, e ainda acrescenta uma folga pequena.

--------------------------------------------------------------------------
POR QUE O DETECTOR, E NAO UM CORTE CENTRAL
--------------------------------------------------------------------------
E a mesma peca que roda em producao. Reenquadrar a referencia com ela deixa o
enquadramento da referencia igual ao da consulta por construcao — e assimetria
entre os dois lados e fonte silenciosa de similaridade baixa em par que deveria
casar.

Referencia em que o detector nao acha nada e **copiada como esta**. Perder uma
referencia e pior que manter uma folgada.

Typical usage:
    poetry run python tools/tighten_references.py --origem brands --destino brands_cortado
"""

import argparse
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import AppConfig  # noqa: E402
from domain.entities.detection import Detection  # noqa: E402
from domain.value_objects.box import Box  # noqa: E402
from infrastructure.detection.owlv2_detector import Owlv2Detector  # noqa: E402
from infrastructure.image.pillow_image_source import (  # noqa: E402
    ACCEPTED_EXTENSIONS,
    PillowImageSource,
)
from shared.logging.logger import configure_logging, get_logger  # noqa: E402

log = get_logger("tighten")

# Folga em volta da caixa do detector. Pequena de proposito: o corte a 40% do
# teste mostrou que invadir a marca piora. Melhor sobrar um fio que faltar letra.
_MARGIN = 0.06

# Abaixo disto a caixa provavelmente pegou um detalhe dentro do logo — uma letra
# solta, um simbolo interno — e nao a marca. Referencia ja e um recorte: a marca
# domina o quadro, entao caixa minuscula e sinal de que o detector se perdeu.
_MIN_AREA_SHARE = 0.15

# Acima disto a caixa cobre a imagem toda e reenquadrar nao muda nada.
_MAX_AREA_SHARE = 0.98


@dataclass
class Report:
    """Contagem do que aconteceu com cada referencia.

    Attributes:
        tightened: Referencias reenquadradas.
        kept: Copiadas como estavam, por falta de caixa utilizavel.
        failed: Arquivos que nao abriram.
        area_before: Soma das areas originais, em pixels.
        area_after: Soma das areas finais, em pixels.
    """

    tightened: int = 0
    kept: int = 0
    failed: int = 0
    area_before: int = 0
    area_after: int = 0


def main(arguments: list[str] | None = None) -> int:
    """Reenquadra uma arvore de referencias inteira.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 em sucesso, 1 se a origem nao existir.
    """
    options = _parser().parse_args(arguments)
    configure_logging("INFO")

    origin = Path(options.origem)
    destination = Path(options.destino)
    if not origin.is_dir():
        log.error("pasta de origem nao existe: %s", origin)
        return 1

    config = AppConfig()
    source = PillowImageSource(min_side_to_upscale=config.encoder.min_side_to_upscale)
    detector = Owlv2Detector(
        config=config.detector, device=options.dispositivo, precision=options.precisao
    )
    detector.prepare()

    files = sorted(_walk(origin))
    log.info("%d referencias em %s", len(files), origin)

    report = Report()
    for position, path in enumerate(files, start=1):
        _process(path, origin, destination, source, detector, report)
        if position % 50 == 0:
            log.info("  %d/%d", position, len(files))

    _print_report(report, destination)
    return 0


def _process(
    path: Path,
    origin: Path,
    destination: Path,
    source: PillowImageSource,
    detector: Owlv2Detector,
    report: Report,
) -> None:
    """Reenquadra uma referencia e grava no destino.

    Args:
        path: Arquivo de origem.
        origin: Raiz da arvore de origem, para espelhar a estrutura.
        destination: Raiz da arvore de saida.
        source: Acesso a imagem.
        detector: Detector agnostico de marca.
        report: Contagem acumulada, alterada no lugar.
    """
    target = destination / path.relative_to(origin)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        image = source.load(path)
    except OSError as error:
        log.warning("ilegivel, pulando: %s (%s)", path.name, error)
        report.failed += 1
        return

    width, height = source.dimensions(image)
    report.area_before += width * height

    box = _best_box(detector.detect(image), width, height)
    if box is None:
        # Copia o arquivo original, e nao a imagem decodificada: preserva o
        # formato, o perfil de cor e a transparencia sem reencodar.
        shutil.copy2(path, target)
        report.kept += 1
        report.area_after += width * height
        return

    enlarged = box.with_margin(_MARGIN, width, height)
    crop = image.crop((enlarged.x1, enlarged.y1, enlarged.x2, enlarged.y2))
    crop.save(target if target.suffix.lower() != ".png" else target.with_suffix(".png"))
    report.tightened += 1
    report.area_after += enlarged.area


def _best_box(detections: tuple[Detection, ...], width: int, height: int) -> Box | None:
    """Escolhe a caixa que melhor representa a marca na referencia.

    A confianca do detector sozinha nao serve: ela dispara em detalhe interno do
    logo com a mesma folga que na marca inteira. Como referencia ja e um
    recorte, a marca domina o quadro — entao o filtro de area vem primeiro, e a
    confianca so desempata o que sobrou.

    Args:
        detections: Deteccoes na imagem de referencia.
        width: Largura da referencia.
        height: Altura da referencia.

    Returns:
        A caixa escolhida, ou None quando nenhuma serve e o arquivo deve ser
        mantido como esta.
    """
    if not detections:
        return None
    total = float(width * height)
    usable = [
        detection
        for detection in detections
        if _MIN_AREA_SHARE <= detection.box.area / total <= _MAX_AREA_SHARE
    ]
    if not usable:
        return None
    return max(usable, key=lambda detection: detection.confidence).box


def _walk(root: Path) -> Iterator[Path]:
    """Percorre a arvore de referencias.

    Args:
        root: Raiz da busca.

    Yields:
        Cada arquivo de imagem encontrado.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ACCEPTED_EXTENSIONS:
            yield path


def _print_report(report: Report, destination: Path) -> None:
    """Imprime o resumo do reenquadramento.

    Args:
        report: Contagem acumulada.
        destination: Pasta de saida, para o proximo comando.
    """
    total = report.tightened + report.kept
    print(f"\n{total} referencias gravadas em {destination}")
    print(f"  {report.tightened:>5} reenquadradas")
    print(f"  {report.kept:>5} mantidas como estavam (detector nao achou caixa utilizavel)")
    if report.failed:
        print(f"  {report.failed:>5} ilegiveis")
    if report.area_before:
        reduction = 1 - report.area_after / report.area_before
        print(f"\narea media {reduction * 100:.1f}% menor — e o fundo que saiu do quadro")
    print("\nProximo passo:")
    print(f"  poetry run stratosphere banco --referencias {destination} --destino indice")


def _parser() -> argparse.ArgumentParser:
    """Monta o analisador de argumentos.

    Returns:
        O analisador configurado.
    """
    parser = argparse.ArgumentParser(
        prog="tighten_references",
        description="Reenquadra referencias no logo usando o detector da pipeline.",
    )
    parser.add_argument("--origem", required=True, help="pasta <marca>/<variante>/arquivo")
    parser.add_argument("--destino", required=True, help="onde gravar a arvore reenquadrada")
    parser.add_argument("--dispositivo", default="cuda:0", help="cuda:0 ou cpu")
    parser.add_argument("--precisao", default="float16", help="float16 ou float32")
    return parser


if __name__ == "__main__":
    sys.exit(main())
