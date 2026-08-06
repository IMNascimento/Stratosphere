"""Garimpa candidatas a novas referencias em imagens nao rotuladas.

--------------------------------------------------------------------------
O DEFEITO QUE ISTO CONSERTA
--------------------------------------------------------------------------
O banco nao erra por ter referencia ruim: erra por **nao ter a variante certa**.
Medido numa foto de coletiva, com a marca correta em primeiro lugar na busca:

    marca         refs de backdrop    resultado
    volkswagen           11           aceita 3x
    ifood                11           aceita 2x
    amazon                9           aceita
    sadia                 3           revisao
    nike                  2           REJEITADA (similaridade 0.855)
    vivo                  1           REJEITADA

O swoosh gigante de um painel procura vizinho entre as 27 referencias de
uniforme da nike e nao acha nenhuma parecida. O consenso desaba — 2 de 25 — e a
regiao morre. **Nao e duvida sobre a marca; e aritmetica do banco.**

--------------------------------------------------------------------------
COMO ISTO NAO VIRA VIES
--------------------------------------------------------------------------
Promover recorte para o banco e o mecanismo de melhoria do proprio projeto — a
marca `cbf` ja tem 23 referencias `promovido`. Mas promover errado envenena o
banco, e promover a partir da imagem de avaliacao **fabrica metrica**: o sistema
passaria a reconhecer um recorte de si mesmo.

Duas travas, e a primeira e inegociavel:

- **`--excluir` tira arquivos da garimpagem.** A imagem usada para medir NUNCA
  pode entrar no banco que sera medido. Sem isso, todo numero produzido depois e
  ficcao.
- **Nada entra sozinho.** A ferramenta escreve candidatas numa pasta, com o
  recorte e o motivo no nome do arquivo, para alguem olhar. Aprovar e copiar
  para a pasta de referencias e reconstruir o banco.

Typical usage:
    poetry run python tools/mine_references.py --entrada fotos/ --marcas nike,vivo \\
        --destino runs/candidatas --excluir foto_de_teste.webp
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from application.dtos.analysis import AnalyzeImageCommand, RegionOutput
from config.settings import AppConfig
from domain.value_objects.box import Box
from infrastructure.container.container import build_container
from infrastructure.environment.env_settings import apply_env_overrides, load_env_file
from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere.mine_references")

_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def main(arguments: list[str] | None = None) -> int:
    """Garimpa e grava as candidatas.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 em sucesso, 1 quando a pasta de entrada nao existe.
    """
    options = _parser().parse_args(arguments)
    configure_logging("INFO")

    root = Path(options.entrada)
    if not root.is_dir():
        log.error("pasta de entrada nao existe: %s", root)
        return 1

    load_env_file()
    config = apply_env_overrides(AppConfig())
    config = replace(config, device=options.dispositivo, precision=options.precisao)
    container = build_container(config, database_path=Path(options.banco))
    if container.analyze_image is None:
        log.error("banco nao encontrado em %s", options.banco)
        return 1

    wanted = {m.strip().lower() for m in options.marcas.split(",") if m.strip()}
    excluded = {Path(p).name for p in options.excluir}
    destination = Path(options.destino)
    destination.mkdir(parents=True, exist_ok=True)

    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in _EXTENSIONS and p.name not in excluded
    ]
    log.info(
        "%d imagens (%d excluidas), procurando %s com similaridade >= %.2f",
        len(files),
        len(excluded),
        sorted(wanted) or "qualquer marca",
        options.similaridade,
    )
    if excluded:
        log.info("EXCLUIDAS da garimpagem: %s", sorted(excluded))

    found = 0
    for position, path in enumerate(files, start=1):
        try:
            output = container.analyze_image.execute(AnalyzeImageCommand(path=path))
        except OSError as error:
            log.warning("ilegivel: %s (%s)", path.name, error)
            continue

        image = container.image_source.load(path)
        for region in output.regions:
            brand = region.brand or _top_brand(region)
            if wanted and brand not in wanted:
                continue
            if region.similarity < options.similaridade:
                continue
            if region.queue.value == "auto_aceite" and not options.incluir_aceitas:
                # Ja aceita: o banco nao precisa de ajuda para este caso.
                continue

            found += 1
            crop = container.image_source.crop(
                image, _box(region), config.encoder.crop_margin, options.lado
            )
            name = (
                f"{brand}__sim{region.similarity:.3f}__{region.queue.value}"
                f"__{path.stem[:40]}__{region.identifier.rsplit('-', 1)[-1]}.png"
            )
            crop.save(destination / name)

        if position % 25 == 0:
            log.info("  %d/%d imagens, %d candidatas", position, len(files), found)

    log.info("%d candidatas em %s", found, destination)
    print(
        f"\n{found} candidatas gravadas em {destination}\n"
        "Olhe uma por uma. Para aprovar, copie as boas para\n"
        "  <referencias>/<marca>/<variante>/\n"
        "e reconstrua o banco com `stratosphere banco`."
    )
    return 0


def _box(region: RegionOutput) -> Box:
    """Reconstroi a caixa da regiao no formato que o recorte espera.

    Args:
        region: Regiao da saida.

    Returns:
        A caixa de dominio.
    """
    x1, y1, x2, y2 = region.box
    return Box(x1, y1, x2, y2)


def _top_brand(region: RegionOutput) -> str | None:
    """Marca proposta mesmo quando a regiao foi rejeitada.

    Regiao rejeitada tem `brand` nulo por invariante do dominio — nunca afirmar
    marca no que foi descartado. Mas para garimpar referencia e justamente o
    descarte que interessa, entao aqui a marca vem do motivo registrado.

    Args:
        region: Regiao da saida.

    Returns:
        A marca citada no primeiro motivo, ou None.
    """
    for reason in region.reasons:
        if "vs '" in reason:
            return str(reason.split("vs '", 1)[1].split("'", 1)[0])
    return None


def _parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos.

    Returns:
        O parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="garimpa candidatas a novas referencias, para aprovacao humana"
    )
    parser.add_argument("--entrada", required=True, help="pasta de imagens nao rotuladas")
    parser.add_argument("--banco", default="indice", help="pasta do indice vetorial")
    parser.add_argument("--destino", required=True, help="pasta onde gravar as candidatas")
    parser.add_argument("--marcas", default="", help="marcas de interesse, separadas por virgula")
    parser.add_argument(
        "--similaridade", type=float, default=0.80, help="similaridade minima da candidata"
    )
    parser.add_argument("--lado", type=int, default=448, help="lado do recorte gravado")
    parser.add_argument(
        "--incluir-aceitas", action="store_true", help="tambem grava o que ja foi aceito"
    )
    parser.add_argument(
        "--excluir",
        nargs="*",
        default=[],
        help="arquivos a NUNCA garimpar. Use para a imagem de avaliacao.",
    )
    parser.add_argument("--dispositivo", default="cuda:0")
    parser.add_argument("--precisao", default="float16")
    return parser


if __name__ == "__main__":
    sys.exit(main())
