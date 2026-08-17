"""Mede o custo de cada regra de admissao contra as referencias reais.

--------------------------------------------------------------------------
POR QUE UM TRADE-OFF DE ADMISSAO NAO SE PARECE COM O DO ROTEADOR
--------------------------------------------------------------------------
`sweep_routing.py` mede precisao e recall porque **existe rotulo**: a marca da
regiao esta certa ou errada, e da para contar.

Aqui nao existe. Nao ha um conjunto de imagens marcadas como "esta referencia
deveria entrar" e "esta nao deveria" - se houvesse, o problema estaria resolvido
por construcao. Reportar "precisao da regra" seria inventar um numero.

O que **e** medivel, e o que de fato pegou o erro real deste projeto, e o custo
de recusa: **quantas referencias legitimas cada limiar recusaria, e de quem**.
Foi assim que um corte de 48px no menor lado foi descartado - ele recusava 227
das 753 referencias reais, concentradas na variante `digital`, e nenhum teste de
medida inventada teria mostrado isso.

A leitura correta de cada numero:

- **recusa alta** nao significa "limiar errado" por si so. Significa "esta regra
  esta afirmando que 30% do banco atual e ruim" - e ai a pergunta e se voce
  acredita nisso.
- **concentracao por variante importa mais que o total.** Uma regra que recusa
  10% do banco espalhado e barata; uma que recusa 10% todo de uma variante cria
  carencia de cobertura, que e o defeito mais caro do sistema.

MEDIDO, e e o exemplo que justifica a coluna de concentracao: `digital` e 40% do
banco e tem menor lado mediano de **49px** - a variante mais numerosa e tambem a
menor. O corte de 48px partia a distribuicao dela ao meio (43% abaixo), enquanto
`oficial` (mediana 379px) e `promovido` (224px) nao perdiam nada. Ler so o total
- "recusa 30%" - esconde que a recusa cai quase toda sobre a variante mais
representada, e por ela ser intrinsecamente pequena, nao por ser ruim.

--------------------------------------------------------------------------
ESTA FERRAMENTA E AUTOCONTIDA DE PROPOSITO
--------------------------------------------------------------------------
Ela nao importa nada de `domain`, `application` ou `config`. Le a arvore de
referencias e o `.npz` do indice direto, e traz os limiares como constantes
locais - **sao propostas de diagnostico, nao configuracao do sistema**.

A razao e que a pipeline e publica e estavel: uma ferramenta de medicao nao deve
exigir mudanca de contrato para rodar. Se algum destes limiares virar politica
de producao um dia, ele muda de lugar - ate entao vive aqui, onde mexer nao
quebra ninguem.

Typical usage:
    poetry run python tools/sweep_admission.py --referencias referencias_v3
    poetry run python tools/sweep_admission.py --referencias referencias_v3 --indice indice
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere.sweep_admission")

_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

# Cinza medio, o mesmo que a pipeline usa no letterbox. Importa para o brilho:
# compor alpha sobre branco ou preto muda a medida.
_FILL_COLOR = (124, 116, 104)

# --------------------------------------------------------------------------
# LIMIARES PROPOSTOS - defaults de diagnostico, nao configuracao do sistema
# --------------------------------------------------------------------------
# Cada valor abaixo e uma PROPOSTA cujo custo esta na saida da ferramenta. O
# marcador `<-` na tabela indica qual valor estas constantes sugerem, para que a
# leitura seja "quanto custa o que eu proponho" e nao um numero solto.

# Menor lado abaixo do qual o recorte nao tem pixel para o codificador ver forma.
# MEDIDO: 227 das 753 referencias reais tem lado menor que 48px - simbolo em
# manga de camisa e logo em post SAO pequenos, e o detector da pipeline aceita
# 12px por esse motivo. Em 24px nenhuma referencia real e recusada.
PROPOSED_MIN_SIDE = 24

# Desvio padrao do brilho abaixo do qual a imagem e um retangulo de cor unica:
# exportacao vazia, recorte que caiu fora da imagem.
# MEDIDO: retangulo chapado da 0.0 exato, e a referencia real mais uniforme das
# 753 da 6.83. O corte fica no meio dessa faixa.
#
# ATENCAO: este limiar NAO detecta alpha achatado, e a tentativa foi abandonada.
# Um logo escuro achatado mede 7.55 - ACIMA da referencia real mais uniforme -
# e a fracao de cor dominante e identica a do mesmo arte composto corretamente
# (0.828 nos dois). Depois de achatado, o arquivo ruim e um logo legitimo sobre
# fundo preto sao o mesmo arquivo, pixel por pixel.
PROPOSED_MIN_BRIGHTNESS_STD = 3.0

# Densidade de borda abaixo da qual o recorte nao tem estrutura grafica.
# MEDIDO nas 753 reais: mediana 0.2857, p0.5 = 0.0070, e apenas 4 abaixo de
# 0.0050. Investigadas uma a uma, essas 4 sao fotos DESFOCADAS - tres delas ja
# marcadas `auto_rejeicao` quando foram garimpadas. Sao o que o limiar recusa.
PROPOSED_MIN_EDGE_DENSITY = 0.0035

# Acima disto contra a MESMA marca a candidata entra COM AVISO - nao e recusa.
# MEDIDO em 11.308 pares da mesma marca: max = 0.985, ou seja, o corte de
# redundancia esta no topo absoluto da distribuicao e nada dispara antes dele.
# Simulando cada referencia como candidata nova: 0.93 avisaria 63% do banco,
# 0.95 avisa 44%, 0.96 avisa 34%. Um terco disparar e o diagnostico, nao
# defeito do limiar - `amazon` tem 19 de 20 refs acima de 0.95 de alguma irma.
PROPOSED_VERY_SIMILAR = 0.96

# Acima disto contra a MESMA marca e quase-copia. Alto de proposito: quando era
# 0.95, a deduplicacao comeu referencia legitima e `amazon` caiu de 26 para 3.
PROPOSED_REDUNDANCY = 0.985

# Acima disto contra a MESMA marca e o mesmo arquivo, nao uma foto parecida.
PROPOSED_DUPLICATE = 0.999

# Os tres limiares de similaridade dependem da ESCALA DO CODIFICADOR e nao sao
# portateis. Medido: no DINOv2, pares da mesma marca em fotos diferentes tinham
# mediana 0.671; no SigLIP2, 0.840. Trocar o codificador invalida todos.

# Quantas referencias uma marca precisa ter para o centroide dela significar
# algo. Abaixo disso a media de dois ou tres vetores nao descreve uma classe.
MIN_REFERENCES_FOR_CENTROID = 5


@dataclass(frozen=True)
class Measured:
    """O que foi medido de uma referencia real.

    Attributes:
        brand: Marca, do primeiro nivel da pasta.
        variant: Variante, do segundo nivel.
        side: Menor lado em pixels.
        brightness: Desvio padrao do brilho, escala 0-255.
        edges: Densidade de borda, fracao entre 0 e 1.
        readable: Se o arquivo pode ser lido.
    """

    brand: str
    variant: str
    side: int
    brightness: float
    edges: float
    readable: bool = True


@dataclass(frozen=True)
class Cost:
    """Custo de um limiar sobre o conjunto real.

    Attributes:
        rejected: Quantas referencias reais o limiar recusaria.
        total: Quantas foram avaliadas.
        by_variant: Recusas por variante, para revelar concentracao.
    """

    rejected: int
    total: int
    by_variant: Counter[str]

    @property
    def share(self) -> float:
        """Fracao do banco que o limiar recusaria.

        Returns:
            Fracao entre 0.0 e 1.0.
        """
        return self.rejected / self.total if self.total else 0.0

    @property
    def concentration(self) -> str:
        """A variante mais atingida, quando a recusa se concentra.

        Concentracao e o sinal de perigo que o total esconde: recusar 10%
        espalhado e barato, recusar 10% todo de uma variante cria carencia de
        cobertura.

        Returns:
            Descricao da concentracao, ou vazio quando nao ha recusa.
        """
        if not self.by_variant:
            return ""
        variant, count = self.by_variant.most_common(1)[0]
        return f"{count}/{self.rejected} em {variant!r}"


def main(arguments: list[str] | None = None) -> int:
    """Mede o custo de cada regra de admissao.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 em sucesso, 1 quando nao ha referencias.
    """
    options = _parser().parse_args(arguments)
    configure_logging("INFO" if options.verboso else "WARNING")

    root = Path(options.referencias)
    references = _walk(root)
    if not references:
        log.error("nenhuma imagem em %s", root)
        return 1

    log.info("medindo %d referencias de %s", len(references), root)
    measured = [_measure(path, root) for path in references]
    usable = [item for item in measured if item.readable]
    unreadable = len(measured) - len(usable)

    print(f"\n=== {len(measured)} referencias | {len(usable)} legiveis", end="")
    print(f" | {unreadable} ilegiveis ===" if unreadable else " ===")

    _report_integrity(usable)
    _report_similarity(Path(options.indice))
    return 0


def _report_integrity(measured: list[Measured]) -> None:
    """Varre os limiares de integridade e imprime o custo de cada corte.

    Args:
        measured: Referencias reais ja medidas.
    """
    print("\n--------------------------------------------------------------")
    print("LIMIARES DE INTEGRIDADE - quanto do banco real cada corte recusa")
    print("--------------------------------------------------------------")
    print("O valor proposto esta marcado com <-. Recusa concentrada numa")
    print("variante e pior que recusa espalhada: e assim que se cria")
    print("carencia de cobertura.\n")

    _sweep(
        "menor lado (px)",
        measured,
        values=(12, 16, 24, 32, 40, 48, 64),
        proposed=PROPOSED_MIN_SIDE,
        predicate=lambda item, cut: item.side < cut,
        fmt="{:.0f}",
    )
    _sweep(
        "brilho std",
        measured,
        values=(0.5, 1.0, 3.0, 6.0, 10.0),
        proposed=PROPOSED_MIN_BRIGHTNESS_STD,
        predicate=lambda item, cut: item.brightness < cut,
        fmt="{:.1f}",
    )
    _sweep(
        "densidade de borda",
        measured,
        values=(0.0010, 0.0035, 0.0070, 0.0125, 0.0200),
        proposed=PROPOSED_MIN_EDGE_DENSITY,
        predicate=lambda item, cut: item.edges < cut,
        fmt="{:.4f}",
    )


def _sweep(
    label: str,
    measured: list[Measured],
    values: tuple[float, ...],
    proposed: float,
    predicate: Callable[[Measured, float], bool],
    fmt: str,
) -> None:
    """Imprime o custo de cada valor de um limiar.

    Args:
        label: Nome do limiar.
        measured: Referencias reais ja medidas.
        values: Valores a testar.
        proposed: Valor proposto, marcado na saida.
        predicate: Funcao `(Measured, valor) -> bool`, True quando recusaria.
        fmt: Formato do valor na tabela.
    """
    print(f"  {label}")
    print(f"    {'corte':>10} {'recusa':>8} {'do banco':>9}  concentracao")
    for value in values:
        rejected = [item for item in measured if predicate(item, value)]
        cost = Cost(
            rejected=len(rejected),
            total=len(measured),
            by_variant=Counter(item.variant for item in rejected),
        )
        marker = " <-" if abs(value - proposed) < 1e-9 else ""
        print(
            f"    {fmt.format(value):>10} {cost.rejected:>8} {cost.share:>8.1%}"
            f"  {cost.concentration}{marker}"
        )
    print()


def _report_similarity(index_folder: Path) -> None:
    """Varre os limiares de similaridade sobre o indice ja construido.

    Usa os vetores do `.npz` em vez de recodificar: sao os MESMOS vetores que a
    busca compara, e recodificar 753 imagens para obter numeros identicos
    custaria minutos de GPU sem mudar uma casa decimal.

    Args:
        index_folder: Pasta do indice vetorial.
    """
    vectors_file = index_folder / "referencias.npz"
    metadata_file = index_folder / "referencias.json"
    if not vectors_file.exists() or not metadata_file.exists():
        print(f"(indice ausente em {index_folder} - varredura de similaridade pulada)")
        return

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    vectors: NDArray[np.float32] = np.load(vectors_file)["vetores"]
    entries = metadata["entradas"]
    brands = np.array([entry["marca"] for entry in entries])
    variants = np.array([entry.get("variante", "") for entry in entries])

    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1.0)

    _report_distribution(similarities, brands, variants)
    _report_similarity_cuts(similarities, brands, variants)
    _report_cohesion(vectors, brands)


def _report_distribution(
    similarities: NDArray[np.float32],
    brands: NDArray[np.str_],
    variants: NDArray[np.str_],
) -> None:
    """Imprime a distribuicao de similaridade das duas populacoes.

    E o numero que revela por que o corte de redundancia nao dispara: ele esta
    no topo absoluto da distribuicao real.

    Args:
        similarities: Matriz de similaridade do banco contra si mesmo.
        brands: Marca de cada linha.
        variants: Variante de cada linha.
    """
    upper = np.triu_indices(len(brands), 1)
    same_brand = brands[upper[0]] == brands[upper[1]]
    same_variant = variants[upper[0]] == variants[upper[1]]
    pairs = similarities[upper]

    print("--------------------------------------------------------------")
    print("A DISTRIBUICAO REAL - onde os limiares caem")
    print("--------------------------------------------------------------")
    print(f"    {'populacao':<26} {'n':>8} {'p50':>7} {'p90':>7} {'p95':>7} {'max':>7}")
    for label, subset in (
        ("mesma marca", pairs[same_brand]),
        ("mesma marca + variante", pairs[same_brand & same_variant]),
        ("marcas DIFERENTES", pairs[~same_brand]),
    ):
        if not len(subset):
            continue
        print(
            f"    {label:<26} {len(subset):>8} {np.percentile(subset, 50):>7.3f} "
            f"{np.percentile(subset, 90):>7.3f} {np.percentile(subset, 95):>7.3f} "
            f"{subset.max():>7.3f}"
        )
    print()


def _report_similarity_cuts(
    similarities: NDArray[np.float32],
    brands: NDArray[np.str_],
    variants: NDArray[np.str_],
) -> None:
    """Simula cada referencia como candidata nova contra a propria marca.

    Args:
        similarities: Matriz de similaridade do banco contra si mesmo.
        brands: Marca de cada linha.
        variants: Variante de cada linha.
    """
    print("LIMIARES DE SIMILARIDADE - cada referencia como candidata nova")
    print("--------------------------------------------------------------")
    print("Contra o vizinho mais proximo da PROPRIA marca - a situacao exata")
    print("de uma candidata chegando.\n")
    print(f"    {'corte':>10} {'atinge':>8} {'do banco':>9} {'variante nova':>14}  concentracao")

    total = len(brands)
    for value in (0.93, 0.95, PROPOSED_VERY_SIMILAR, 0.97, PROPOSED_REDUNDANCY,
                  PROPOSED_DUPLICATE):
        hit: list[str] = []
        spared = 0
        for position in range(total):
            same = brands == brands[position]
            same[position] = False
            if not same.any() or similarities[position][same].max() < value:
                continue
            # Variante nova dispensa o aviso: cobrir aplicacao que faltava vale
            # mais que a semelhanca com o que a marca ja tem.
            same_variant = same & (variants == variants[position])
            if not same_variant.any():
                spared += 1
                continue
            hit.append(str(variants[position]))

        by_variant = Counter(hit)
        top = by_variant.most_common(1)
        concentration = f"{top[0][1]}/{len(hit)} em {top[0][0]!r}" if top else ""
        marker = ""
        if abs(value - PROPOSED_VERY_SIMILAR) < 1e-9:
            marker = " <- aviso"
        elif abs(value - PROPOSED_REDUNDANCY) < 1e-9:
            marker = " <- redundante"
        elif abs(value - PROPOSED_DUPLICATE) < 1e-9:
            marker = " <- duplicata"
        print(
            f"    {value:>10.3f} {len(hit):>8} {len(hit) / total:>8.1%}"
            f" {spared:>14}  {concentration}{marker}"
        )

    print("\n  'variante nova' = dispensadas por cobrirem aplicacao ausente.")
    print("  Acima de 0.985 nao ha par legitimo: o corte de redundancia esta no")
    print("  topo absoluto da distribuicao, e por isso nada dispara antes dele.\n")


def _report_cohesion(vectors: NDArray[np.float32], brands: NDArray[np.str_]) -> None:
    """Mede o quanto cada marca se parece consigo mesma.

    Cada referencia contra o centroide da propria marca, calculado SEM ela -
    senao ela se puxa e o numero perde sentido.

    Marca dispersa nao e defeito por si so: `cbf` tem escudo, wordmark e
    uniforme, e essa variedade E a cobertura que o sistema quer. O numero vira
    diagnostico quando cruzado com a composicao por variante.

    Args:
        vectors: Matriz de vetores do indice, L2-normalizados.
        brands: Marca de cada linha.
    """
    print("--------------------------------------------------------------")
    print("COESAO POR MARCA - cada ref contra o centroide da propria classe")
    print("--------------------------------------------------------------")
    print("Dispersa nao e ruim por si so - marca com escudo, wordmark e")
    print("uniforme E dispersa de proposito. Cruze com a composicao.\n")
    print(f"    {'marca':<24} {'n':>4} {'coesao':>8} {'pior ref':>9}")

    rows: list[tuple[float, str, int, float]] = []
    for brand in sorted(set(brands.tolist())):
        rows_of_brand = np.where(brands == brand)[0]
        if len(rows_of_brand) < MIN_REFERENCES_FOR_CENTROID:
            continue
        cohesions = []
        for position in rows_of_brand:
            others = rows_of_brand[rows_of_brand != position]
            centroid = vectors[others].mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm == 0.0:
                continue
            cohesions.append(float(vectors[position] @ (centroid / norm)))
        if cohesions:
            array = np.array(cohesions)
            rows.append((float(np.median(array)), brand, len(rows_of_brand), float(array.min())))

    rows.sort()
    for median, brand, count, worst in rows:
        flag = "  <- dispersa" if median < 0.90 else ""
        print(f"    {brand:<24} {count:>4} {median:>8.3f} {worst:>9.3f}{flag}")

    if rows:
        medians = np.array([row[0] for row in rows])
        print(f"\n  mediana global: {np.median(medians):.3f}")
    print()


def _measure(path: Path, root: Path) -> Measured:
    """Mede uma referencia real.

    Compoe a transparencia sobre o fundo neutro antes de medir, do mesmo jeito
    que a pipeline faz na leitura: `convert("RGB")` puro descarta o alfa e
    mantem o RGB de baixo, que em arte vetorial costuma ser preto - e ai o
    brilho medido seria de um retangulo que nao existe.

    Args:
        path: Arquivo a medir.
        root: Raiz da arvore, para extrair marca e variante.

    Returns:
        As medidas, ou `readable=False` quando o arquivo nao abre.
    """
    relative = path.relative_to(root)
    brand = relative.parts[0]
    variant = relative.parts[1] if len(relative.parts) > 2 else "raiz"
    try:
        opened = Image.open(path)
        oriented = ImageOps.exif_transpose(opened) or opened
        image = _flatten(oriented)
    except OSError:
        return Measured(brand, variant, 0, 0.0, 0.0, readable=False)

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    return Measured(
        brand=brand,
        variant=variant,
        side=min(image.size),
        brightness=float(gray.std()) if gray.size else 0.0,
        edges=_edge_density(gray),
    )


def _flatten(image: Image.Image) -> Image.Image:
    """Compoe transparencia sobre o fundo neutro antes de virar RGB.

    Args:
        image: Imagem recem aberta, com ou sem canal alfa.

    Returns:
        A imagem em RGB, sem transparencia.
    """
    transparent = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if not transparent:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGB", rgba.size, _FILL_COLOR)
    canvas.paste(rgba, mask=rgba.split()[-1])
    return canvas


def _edge_density(gray: NDArray[np.float32], max_side: int = 256) -> float:
    """Mede a fracao de pixels com gradiente forte.

    Mesma formula da pipeline - diferencas centrais numa versao reduzida, com
    corte de magnitude em 24 - para que os numeros sejam comparaveis com os
    limiares em uso.

    Args:
        gray: Imagem em escala de cinza.
        max_side: Maior lado da versao reduzida usada na medida.

    Returns:
        Fracao entre 0.0 e 1.0.
    """
    height, width = gray.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1.0:
        reduced = Image.fromarray(gray).resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
        gray = np.asarray(reduced, dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    gradient_x = gray[1:-1, 2:] - gray[1:-1, :-2]
    gradient_y = gray[2:, 1:-1] - gray[:-2, 1:-1]
    return float((np.hypot(gradient_x, gradient_y) > 24.0).mean())


def _walk(root: Path) -> tuple[Path, ...]:
    """Lista as imagens da arvore de referencias.

    Args:
        root: Raiz `<marca>/<variante>/arquivo`.

    Returns:
        Caminhos em ordem estavel. Vazio quando a raiz nao existe.
    """
    if not root.exists():
        return ()
    return tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _EXTENSIONS
    )


def _parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos.

    Returns:
        O parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="mede o custo de cada regra de admissao contra as referencias reais"
    )
    parser.add_argument("--referencias", default="referencias_v3", help="arvore de referencias")
    parser.add_argument(
        "--indice", default="indice", help="indice para a varredura de similaridade"
    )
    parser.add_argument("-v", "--verboso", action="store_true")
    return parser


if __name__ == "__main__":
    sys.exit(main())
