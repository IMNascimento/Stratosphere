"""Prepara uma imagem de logo para entrar no banco - e diz se ela deve entrar.

--------------------------------------------------------------------------
POR QUE ISTO EXISTE
--------------------------------------------------------------------------
Referencia nova nao e sempre ganho. Ela pode ser:

- **redundante** - quase igual a uma que ja esta la. Nao acrescenta cobertura,
  ocupa o topo da busca com copia e ainda desloca vizinho util do top-k, que e o
  que alimenta o consenso.
- **perigosa** - parecida demais com referencia de OUTRA marca. Cada par assim e
  um falso positivo agendado. Medido no banco antigo: eram 238 pares acima do
  limiar de alerta, e eles explicavam sozinhos `itau` sendo lido como `sadia`.
- **mal enquadrada** - com a placa, a moldura e o fundo em volta. Isso ensina a
  superficie, nao a marca.
- **rotulada errado** - acontece, e o banco nao tem como saber.

Esta ferramenta responde as quatro perguntas antes de a imagem entrar, e grava
uma copia ja preparada. Ela **nao** escreve no banco: a decisao final e humana, e
o relatorio existe para essa decisao ser informada em vez de otimista.

--------------------------------------------------------------------------
O QUE ELA FAZ COM A IMAGEM
--------------------------------------------------------------------------
1. Compoe transparencia sobre o fundo neutro do recorte. PNG com alpha perdido
   vira retangulo preto - quatro referencias do banco original eram exatamente
   isso, e uma delas colocava `cimed x nike` a 0.951 de similaridade.
2. Reenquadra com o **proprio detector da pipeline**, nao com corte fixo. Assim o
   enquadramento da referencia fica igual ao da consulta por construcao, e
   assimetria entre os dois lados e fonte silenciosa de similaridade baixa em par
   que deveria casar.
3. Amplia o que for pequeno demais para render sinal.

Typical usage:
    poetry run python tools/prepare_reference.py --entrada logo.png --marca nike \\
        --variante backdrop --destino referencias_v3
"""

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from application.ports.i_image_source import RgbImage
from config.settings import AppConfig
from domain.entities.detection import Detection
from domain.value_objects.box import Box
from infrastructure.container.container import _build_encoder
from infrastructure.detection.owlv2_detector import Owlv2Detector
from infrastructure.environment.env_settings import apply_env_overrides, load_env_file
from infrastructure.image.pillow_image_source import PillowImageSource
from infrastructure.index.npz_reference_database import NpzReferenceDatabase
from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere.prepare_reference")

_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# Folga ao redor da caixa do detector. Pequena de proposito: cortar rente demais
# come letra, e fragmento mutilado volta a parecer com outro fragmento mutilado.
_MARGIN = 0.06

# Abaixo disto o recorte nao tem pixel suficiente para o codificador ver forma.
_MIN_SIDE = 48


@dataclass
class Verdict:
    """O que se descobriu sobre uma imagem candidata.

    Attributes:
        path: Arquivo avaliado.
        brand: Marca declarada por quem chamou.
        nearest_same: Maior similaridade contra referencia da MESMA marca.
        nearest_same_path: Qual referencia foi essa.
        nearest_other: Maior similaridade contra referencia de OUTRA marca.
        nearest_other_brand: De qual marca.
        top_brand: Marca que o banco daria a esta imagem hoje.
        top_similarity: Similaridade dessa marca.
        side: Menor lado do recorte final.
        reframed: Se o detector reenquadrou.
        problems: Motivos para nao aceitar.
        notes: Observacoes que nao impedem.
    """

    path: Path
    brand: str
    nearest_same: float = 0.0
    nearest_same_path: str = ""
    nearest_other: float = 0.0
    nearest_other_brand: str = ""
    top_brand: str = ""
    top_similarity: float = 0.0
    side: int = 0
    reframed: bool = False
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Se a imagem deve entrar no banco.

        Returns:
            True quando nenhum problema bloqueante foi encontrado.
        """
        return not self.problems


def main(arguments: list[str] | None = None) -> int:
    """Avalia e prepara as imagens candidatas.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 quando ao menos uma candidata foi aprovada, 1 quando nenhuma.
    """
    options = _parser().parse_args(arguments)
    configure_logging("WARNING")

    entry = Path(options.entrada)
    if not entry.exists():
        log.error("entrada nao existe: %s", entry)
        return 1

    load_env_file()
    config = apply_env_overrides(AppConfig())
    source = PillowImageSource(min_side_to_upscale=config.encoder.min_side_to_upscale)
    encoder = _build_encoder(config)
    detector = Owlv2Detector(config.detector, config.device, config.precision)
    database = NpzReferenceDatabase.load(Path(options.banco), encoder.signature())

    candidates = [entry] if entry.is_file() else sorted(_walk(entry))
    if not candidates:
        log.error("nenhuma imagem em %s", entry)
        return 1

    brand = options.marca.strip().lower()
    destination = Path(options.destino) / brand / options.variante.strip().lower()
    verdicts = [
        _evaluate(path, brand, config, source, encoder, detector, database, options)
        for path in candidates
    ]

    approved = [v for v in verdicts if v.accepted]
    if approved and not options.so_relatorio:
        destination.mkdir(parents=True, exist_ok=True)
        for verdict in approved:
            _write(verdict, config, source, detector, destination, options)

    _report(verdicts, destination, options)
    return 0 if approved else 1


def _evaluate(  # noqa: PLR0913 - e uma ferramenta de diagnostico, tudo vem de fora
    path: Path,
    brand: str,
    config: AppConfig,
    source: PillowImageSource,
    encoder: object,
    detector: Owlv2Detector,
    database: NpzReferenceDatabase,
    options: argparse.Namespace,
) -> Verdict:
    """Mede uma candidata contra o banco.

    Args:
        path: Imagem candidata.
        brand: Marca declarada.
        config: Configuracao completa.
        source: Acesso a imagem.
        encoder: Codificador da pipeline.
        detector: Detector agnostico.
        database: Banco de referencia.
        options: Opcoes de linha de comando.

    Returns:
        O parecer sobre esta imagem.
    """
    verdict = Verdict(path=path, brand=brand)
    try:
        prepared, reframed = _prepare(path, config, source, detector)
    except OSError as error:
        verdict.problems.append(f"ilegivel: {error}")
        return verdict

    verdict.reframed = reframed
    verdict.side = min(prepared.size)
    if verdict.side < _MIN_SIDE:
        verdict.problems.append(
            f"pequena demais: menor lado {verdict.side}px, minimo {_MIN_SIDE}px"
        )
        return verdict
    if not reframed:
        verdict.notes.append("o detector nao achou logo aqui - confira o enquadramento")

    # Codifica a preparada E a original. O reenquadramento muda o vetor o
    # bastante para uma copia exata deixar de bater consigo mesma - medido: uma
    # referencia do proprio banco, reenquadrada, ficou a 0.905 da vizinha mais
    # proxima em vez de 0.99 de si. Sem os dois lados, a checagem de redundancia
    # deixaria duplicata entrar.
    original = source.load(path)
    vectors = encoder.encode(  # type: ignore[attr-defined]
        [
            source.crop(img, Box(0, 0, img.size[0], img.size[1]), 0.0, config.encoder.crop_side)
            for img in (prepared, original)
        ]
    )

    neighbours = database.search(vectors[0], 60)
    originais = database.search(vectors[1], 60)
    if not neighbours:
        verdict.notes.append("banco vazio - nada com que comparar")
        return verdict

    verdict.top_brand = neighbours[0].brand
    verdict.top_similarity = neighbours[0].similarity

    same = next((c for c in neighbours if c.brand == brand), None)
    other = next((c for c in neighbours if c.brand != brand), None)
    if same is not None:
        verdict.nearest_same = same.similarity
        verdict.nearest_same_path = Path(same.reference).name
    if other is not None:
        verdict.nearest_other = other.similarity
        verdict.nearest_other_brand = other.brand

    # A original tambem conta para redundancia: e ela que revela duplicata.
    same_original = next((c for c in originais if c.brand == brand), None)
    if same_original is not None and same_original.similarity > verdict.nearest_same:
        verdict.nearest_same = same_original.similarity
        verdict.nearest_same_path = Path(same_original.reference).name

    if verdict.nearest_same >= options.redundancia:
        verdict.problems.append(
            f"REDUNDANTE: {verdict.nearest_same:.3f} contra "
            f"{verdict.nearest_same_path} - nao acrescenta cobertura"
        )
    if verdict.nearest_other >= options.alerta:
        verdict.problems.append(
            f"PERIGOSA: {verdict.nearest_other:.3f} contra {verdict.nearest_other_brand!r}"
            " - falso positivo agendado"
        )
    # Teste de rotulo RELATIVO, e nao contra um limiar absoluto: a pergunta e
    # "o banco prefere outra marca a que voce declarou?". Comparar com um corte
    # fixo deixava passar o caso obvio - uma imagem de guarana declarada como
    # nike, com o banco lendo guarana a 0.905, escapava porque 0.905 < 0.92.
    prefers_other = verdict.top_brand != brand and verdict.top_similarity > (
        verdict.nearest_same + options.margem_rotulo
    )
    if prefers_other:
        verdict.problems.append(
            f"ROTULO SUSPEITO: o banco le esta imagem como {verdict.top_brand!r} "
            f"({verdict.top_similarity:.3f}), acima de qualquer {brand!r} "
            f"({verdict.nearest_same:.3f})"
        )
    if same is None and not prefers_other:
        verdict.notes.append(
            f"nenhuma referencia de {brand!r} entre os 60 vizinhos - variante nova, "
            "e o tipo mais valioso de referencia"
        )
    elif same is not None and verdict.nearest_same < 0.75:
        verdict.notes.append(
            f"distante das existentes ({verdict.nearest_same:.3f}) - cobre uma "
            "aplicacao que o banco nao tinha"
        )
    return verdict


def _prepare(
    path: Path,
    config: AppConfig,
    source: PillowImageSource,
    detector: Owlv2Detector,
) -> tuple[RgbImage, bool]:
    """Compoe alpha, reenquadra no logo e amplia o que for pequeno.

    Args:
        path: Imagem candidata.
        config: Configuracao completa.
        source: Acesso a imagem - ja compoe transparencia no carregamento.
        detector: Detector agnostico.

    Returns:
        A imagem preparada e se o detector chegou a reenquadrar.
    """
    image = source.load(path)
    width, height = image.size
    box = _best_box(detector.detect(image), width, height)
    if box is None:
        return image, False
    folded = box.with_margin(_MARGIN, width, height)
    return image.crop((folded.x1, folded.y1, folded.x2, folded.y2)), True


def _best_box(detections: tuple[Detection, ...], width: int, height: int) -> Box | None:
    """Escolhe a caixa que melhor representa o logo inteiro.

    A de maior area entre as de confianca decente: um wordmark rende varias
    caixas parciais, uma por palavra, e ficar com a maior mantem a marca inteira.

    Args:
        detections: Deteccoes na imagem.
        width: Largura da imagem.
        height: Altura da imagem.

    Returns:
        A caixa escolhida, ou None quando nada foi detectado ou a caixa cobre
        quase tudo - nesse caso reenquadrar nao mudaria nada.
    """
    if not detections:
        return None
    best = max(detections, key=lambda d: d.box.area)
    if best.box.area >= 0.92 * width * height:
        return None
    return best.box


def _write(
    verdict: Verdict,
    config: AppConfig,
    source: PillowImageSource,
    detector: Owlv2Detector,
    destination: Path,
    options: argparse.Namespace,
) -> None:
    """Grava a copia preparada.

    Args:
        verdict: Parecer da imagem.
        config: Configuracao completa.
        source: Acesso a imagem.
        detector: Detector agnostico.
        destination: Pasta `<destino>/<marca>/<variante>`.
        options: Opcoes de linha de comando.
    """
    prepared, _ = _prepare(verdict.path, config, source, detector)
    if min(prepared.size) < options.lado:
        fator = options.lado / min(prepared.size)
        novo = (round(prepared.size[0] * fator), round(prepared.size[1] * fator))
        prepared = prepared.resize(novo)
    saida = destination / f"{verdict.path.stem}.png"
    prepared.save(saida)


def _report(verdicts: list[Verdict], destination: Path, options: argparse.Namespace) -> None:
    """Imprime o relatorio de todas as candidatas.

    Args:
        verdicts: Pareceres.
        destination: Onde as aprovadas foram gravadas.
        options: Opcoes de linha de comando.
    """
    aprovadas = [v for v in verdicts if v.accepted]
    print(f"\n=== {len(verdicts)} candidatas | {len(aprovadas)} aprovadas ===\n")
    for v in verdicts:
        marca = "APROVADA" if v.accepted else "RECUSADA"
        print(f"[{marca}] {v.path.name}")
        print(
            f"    lado {v.side}px | {'reenquadrada' if v.reframed else 'sem reenquadrar'} | "
            f"o banco leria como {v.top_brand or '-'} ({v.top_similarity:.3f})"
        )
        if v.nearest_same:
            print(
                f"    mais proxima da MESMA marca:  {v.nearest_same:.3f}  "
                f"{v.nearest_same_path}"
            )
        if v.nearest_other:
            print(
                f"    mais proxima de OUTRA marca:  {v.nearest_other:.3f}  "
                f"{v.nearest_other_brand}"
            )
        for p in v.problems:
            print(f"    x  {p}")
        for n in v.notes:
            print(f"    -  {n}")
        print()

    if aprovadas and not options.so_relatorio:
        print(f"copias preparadas em {destination}")
        print("Confira no olho antes de reconstruir o banco:")
        print(f"  poetry run stratosphere --banco {options.banco} banco "
              f"--referencias {options.destino} --destino {options.banco}")
        print(f"  poetry run stratosphere --banco {options.banco} auditar")
    elif not aprovadas:
        print("nada aprovado - nenhuma copia gravada")


def _walk(root: Path) -> Iterator[Path]:
    """Percorre as imagens de uma pasta.

    Args:
        root: Pasta a percorrer.

    Yields:
        Caminhos de imagem, em ordem estavel.
    """
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _EXTENSIONS:
            yield path


def _parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos.

    Returns:
        O parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="prepara uma imagem de logo para o banco e diz se ela deve entrar"
    )
    parser.add_argument("--entrada", required=True, help="imagem ou pasta de imagens")
    parser.add_argument("--marca", required=True, help="marca declarada")
    parser.add_argument("--variante", default="promovido", help="subpasta da variante")
    parser.add_argument("--destino", default="referencias_v3", help="arvore de referencias")
    parser.add_argument("--banco", default="indice", help="indice para comparar")
    parser.add_argument(
        "--redundancia",
        type=float,
        default=AppConfig().search.redundancy_similarity,
        help="acima disto contra a MESMA marca, a candidata e copia",
    )
    parser.add_argument(
        "--alerta",
        type=float,
        default=AppConfig().search.alert_similarity,
        help="acima disto contra OUTRA marca, a candidata e perigosa",
    )
    parser.add_argument(
        "--margem-rotulo",
        type=float,
        default=0.02,
        help="quanto o banco precisa preferir outra marca para o rotulo ser suspeito",
    )
    parser.add_argument("--lado", type=int, default=224, help="menor lado da copia gravada")
    parser.add_argument(
        "--so-relatorio", action="store_true", help="avalia sem gravar copia nenhuma"
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
