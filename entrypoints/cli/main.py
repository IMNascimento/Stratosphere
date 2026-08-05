"""Interface de linha de comando do Stratosphere.

Quatro subcomandos, na ordem em que sao usados numa instalacao nova:

    ambiente   confere se da para rodar antes de baixar peso de modelo
    banco      pasta de referencias -> indice vetorial
    auditar    lista marcas do banco que se parecem demais entre si
    analisar   roda a pipeline numa imagem ou pasta

Os nomes dos subcomandos, das opcoes e das chaves do json de saida seguem em
portugues: eles sao a interface publica ja documentada no README, e traduzi-los
quebraria quem ja usa a ferramenta.

O entrypoint so fala com o container e com os casos de uso. Ele nao conhece
OWLv2, DINOv2 nem SIFT — trocar qualquer um deles nao muda nada aqui.

Typical usage:
    poetry run stratosphere banco --referencias marcas/ --destino indice/
    poetry run stratosphere analisar --entrada fotos/ --banco indice/
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from application.dtos.analysis import AnalyzeImageCommand, BuildDatabaseCommand, RegionOutput
from config.settings import AppConfig
from domain.enums.queue import Queue
from domain.exceptions.domain_exceptions import DomainError
from infrastructure.container.container import Container, build_container
from infrastructure.environment.env_settings import (
    ENV_FILE,
    HF_TOKEN_VARIABLE,
    apply_env_overrides,
    has_hf_token,
    load_env_file,
)
from infrastructure.image.pillow_annotator import annotated_path
from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere")

_USAGE_ERROR_CODE = 2
_DOMAIN_ERROR_CODE = 1

# Ordem de exibicao das filas: das que exigem acao humana para as que nao exigem.
_QUEUE_DISPLAY_ORDER = (
    Queue.AUTO_ACCEPT,
    Queue.CONFUSION,
    Queue.ORPHAN,
    Queue.REVIEW,
    Queue.NEGATIVE,
    Queue.AUTO_REJECT,
)


def main(arguments: list[str] | None = None) -> int:
    """Ponto de entrada da CLI.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        Codigo de saida: 0 em sucesso, 1 em erro de dominio, 2 em erro de uso.
    """
    parser = _build_parser()
    options = parser.parse_args(arguments)
    configure_logging("DEBUG" if options.verboso else "INFO")

    if load_env_file():
        log.debug("%s carregado", ENV_FILE)

    try:
        return int(options.function(options))
    except DomainError as error:
        log.error("%s", error)
        return _DOMAIN_ERROR_CODE
    except FileNotFoundError as error:
        log.error("%s", error)
        return _DOMAIN_ERROR_CODE
    except KeyboardInterrupt:
        log.warning("interrompido")
        return 130


# -- subcomandos -----------------------------------------------------------


def _command_environment(options: argparse.Namespace) -> int:
    """Confere se o ambiente aguenta rodar a pipeline.

    Roda antes de qualquer download de peso: transforma meia hora de espera
    seguida de erro num diagnostico de um segundo.

    Args:
        options: Opcoes ja analisadas.

    Returns:
        0 se tudo que a pipeline exige esta presente, 1 caso contrario.
    """
    all_present = True
    print("=== dependencias ===")
    for module, hint in (
        ("torch", "poetry add torch"),
        ("transformers", "poetry add transformers"),
        ("cv2", "poetry add opencv-contrib-python-headless"),
        ("PIL", "poetry add pillow"),
        ("numpy", "poetry add numpy"),
    ):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "ok")
            print(f"  {module:<14} {version}")
        except ImportError:
            print(f"  {module:<14} AUSENTE  ->  {hint}")
            all_present = False

    print("\n=== hugging face ===")
    config = _config(options)
    print(f"  {str(ENV_FILE):<14} {'presente' if ENV_FILE.is_file() else 'ausente'}")
    if has_hf_token():
        print(f"  {HF_TOKEN_VARIABLE:<14} definido")
    else:
        print(f"  {HF_TOKEN_VARIABLE:<14} ausente — so baixa modelo de acesso livre")
        print(f"  {'':<14} preencha em {ENV_FILE} (ver .env.example)")
    print(f"  {'detector':<14} {config.detector.identifier}")
    print(f"  {'codificador':<14} {config.encoder.identifier}")
    print(f"  {'execucao':<14} {config.device} | {config.precision}")

    print("\n=== aceleracao ===")
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            memory = properties.total_memory / 1024**3
            print(f"  GPU            {properties.name} | {memory:.1f} GB")
        else:
            print("  GPU            indisponivel — a pipeline roda em CPU, bem mais devagar")
    except ImportError:
        print("  GPU            nao verificavel sem torch")

    database = Path(options.banco)
    print("\n=== banco de referencia ===")
    if (database / "referencias.json").exists():
        metadata = json.loads((database / "referencias.json").read_text(encoding="utf-8"))
        manifest = metadata["manifesto"]
        print(f"  referencias    {manifest['total_de_referencias']}")
        print(f"  dimensao       {manifest['dimensao']}")
        print(f"  codificador    {manifest['assinatura_do_codificador']}")
        print(f"  por marca      {metadata['referencias_por_marca']}")
    else:
        print(f"  AUSENTE em {database}")
        print("  construa com:  poetry run stratosphere banco --referencias <pasta>")

    print("\n" + ("ambiente pronto." if all_present else "ambiente INCOMPLETO — ver acima."))
    return 0 if all_present else 1


def _command_database(options: argparse.Namespace) -> int:
    """Constroi o indice vetorial a partir da pasta de referencias.

    Args:
        options: Opcoes ja analisadas.

    Returns:
        0 em sucesso.

    Raises:
        ReferencesNotFoundError: Se a pasta nao tiver imagem utilizavel.
    """
    container = _container(options)
    output = container.build_database.execute(
        BuildDatabaseCommand(
            references_folder=Path(options.referencias), destination=Path(options.destino)
        )
    )

    print(f"\nbanco construido em {options.destino}")
    print(f"  {output.total_references} referencias de {len(output.references_by_brand)} marcas")
    if output.discarded_by_redundancy:
        print(
            f"  {output.discarded_by_redundancy} descartadas por redundancia "
            f"(quase identicas a outra da mesma marca)"
        )
    print()
    for brand, quantity in output.references_by_brand.items():
        alert = "   <- poucas referencias" if quantity < 15 else ""
        print(f"  {brand:<24} {quantity:>4}{alert}")

    print("\nProximo passo — SEMPRE audite antes de usar:")
    print("  poetry run stratosphere auditar")
    return 0


def _command_audit(options: argparse.Namespace) -> int:
    """Lista marcas do banco cujas referencias se parecem demais.

    Cada par reportado e um falso positivo esperando acontecer, ou um grupo de
    confusao ainda nao declarado na configuracao.

    Args:
        options: Opcoes ja analisadas.

    Returns:
        0 em sucesso, 1 se o banco nao existir.
    """
    container = _container(options)
    if container.audit_database is None:
        log.error("banco ausente em %s — construa com `stratosphere banco`", options.banco)
        return _DOMAIN_ERROR_CODE

    output = container.audit_database.execute()
    print(
        f"banco: {output.total_references} referencias, "
        f"{len(output.references_by_brand)} marcas\n"
    )

    thin = container.audit_database.brands_with_few_references(options.minimo)
    if thin:
        print("marcas com poucas referencias (esperar orfaos em vez de acertos):")
        for brand, quantity in thin.items():
            print(f"  {brand:<24} {quantity}")
        print()

    if not output.confusable_pairs:
        print("nenhum par de marcas diferentes acima do limiar. Banco limpo.")
        return 0

    print(f"{len(output.confusable_pairs)} pares de marcas DIFERENTES parecidas demais:")
    for brand_a, brand_b, similarity in output.confusable_pairs[: options.limite]:
        print(f"  {similarity:.3f}  {brand_a} x {brand_b}")
    print(
        "\nCada par e um falso positivo agendado. Remova a referencia ruim, ou "
        "declare o grupo em ConfusionConfig.groups."
    )
    return 0


def _command_analyze(options: argparse.Namespace) -> int:
    """Roda a pipeline numa imagem ou numa pasta de imagens.

    Args:
        options: Opcoes ja analisadas.

    Returns:
        0 em sucesso, 1 se o banco nao existir.
    """
    container = _container(options)
    if container.analyze_image is None:
        log.error("banco ausente em %s — construa com `stratosphere banco`", options.banco)
        return _DOMAIN_ERROR_CODE

    paths = _resolve_inputs(container, Path(options.entrada), options.limite)
    if not paths:
        log.error("nenhuma imagem em %s", options.entrada)
        return _DOMAIN_ERROR_CODE

    log.info("analisando %d imagens", len(paths))
    entry = Path(options.entrada)
    annotated_folder = Path(options.anotar) if options.anotar else None
    total_by_queue: dict[str, int] = {}
    rows: list[dict[str, object]] = []

    for position, path in enumerate(paths, start=1):
        output = container.analyze_image.execute(AnalyzeImageCommand(path=path))
        for queue_name, quantity in output.count_by_queue().items():
            total_by_queue[queue_name] = total_by_queue.get(queue_name, 0) + quantity

        visible = [
            region
            for region in output.regions
            if region.queue is not Queue.AUTO_REJECT or options.tudo
        ]
        brands = output.accepted_brands()
        rows.append(_row(path, output.discarded_by, brands, visible))

        if annotated_folder is not None:
            _annotate_by_queue(container, path, entry, annotated_folder, visible)

        if brands:
            print(f"  {path.name[:56]:<58} {', '.join(brands)}")
        if position % 25 == 0:
            log.info("  %d/%d imagens", position, len(paths))

    print(f"\n{len(paths)} imagens analisadas")
    for queue in _QUEUE_DISPLAY_ORDER:
        quantity = total_by_queue.get(queue.value, 0)
        if quantity:
            marker = "  <- exige humano" if queue.requires_human else ""
            print(f"  {queue.value:<16} {quantity:>6}{marker}")

    if options.saida:
        destination = Path(options.saida)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ndetalhe por regiao: {destination}")
    if annotated_folder is not None:
        print(f"imagens anotadas:   {annotated_folder}")
    return 0


# -- apoio -----------------------------------------------------------------


def _annotate_by_queue(
    container: Container,
    source: Path,
    root: Path,
    folder: Path,
    regions: Sequence[RegionOutput],
) -> None:
    """Grava uma copia anotada por fila presente na imagem.

    Cada copia leva **apenas as caixas daquela fila**. Quem abre
    `anotadas/orfao/` esta decidindo promocao para o banco, e caixa de outra
    fila no meio so atrapalha essa decisao. A mesma imagem aparece em mais de
    uma pasta quando tem regioes de filas diferentes — o que e a informacao
    certa: ela exige duas acoes distintas.

    Args:
        container: Container, de onde vem o anotador.
        source: Arquivo analisado.
        root: Raiz da entrada, para espelhar subpastas.
        folder: Pasta raiz das anotacoes.
        regions: Regioes que sobreviveram ao relatorio.
    """
    for queue in _QUEUE_DISPLAY_ORDER:
        selected = [region for region in regions if region.queue is queue]
        if not selected:
            continue
        container.annotator.annotate(
            source, selected, annotated_path(source, root, folder / queue.value)
        )


def _row(
    path: Path,
    discarded_by: str | None,
    brands: tuple[str, ...],
    regions: Sequence[RegionOutput],
) -> dict[str, object]:
    """Monta a linha do json de saida para uma imagem.

    As chaves seguem em portugues: sao o formato ja documentado no README.

    Args:
        path: Arquivo analisado.
        discarded_by: Motivo do descarte no pre-filtro, ou None.
        brands: Marcas entregues sem revisao humana.
        regions: Regioes que entram no relatorio.

    Returns:
        O dicionario pronto para serializar.
    """
    return {
        "imagem": str(path),
        "descartada_por": discarded_by,
        "marcas_aceitas": list(brands),
        "regioes": [
            {
                "identificador": region.identifier,
                "caixa": list(region.box),
                "fila": region.queue.value,
                "marca": region.brand,
                "pontuacao": round(region.score, 4),
                "similaridade": round(region.similarity, 4),
                "margem": round(region.margin, 4),
                "inliers": region.inliers,
                "motivos": list(region.reasons),
            }
            for region in regions
        ],
    }


def _container(options: argparse.Namespace) -> Container:
    """Monta o container com a configuracao vinda da linha de comando.

    Args:
        options: Opcoes ja analisadas.

    Returns:
        O container pronto.
    """
    return build_container(_config(options), Path(options.banco))


def _config(options: argparse.Namespace) -> AppConfig:
    """Monta a configuracao efetiva desta execucao.

    Precedencia, do mais fraco para o mais forte: default do dataclass, `.env`,
    variavel ja exportada no shell, flag da CLI. A flag vem por ultimo porque e
    a decisao mais explicita que alguem pode tomar.

    Args:
        options: Opcoes ja analisadas.

    Returns:
        A configuracao com ambiente e flags aplicados.
    """
    config = apply_env_overrides(AppConfig())
    if getattr(options, "cpu", False):
        config = replace(config, device="cpu", precision="float32")
    return config


def _resolve_inputs(container: Container, entry: Path, limit: int) -> list[Path]:
    """Resolve o argumento de entrada em uma lista de imagens.

    Args:
        container: Container, usado para listar pastas.
        entry: Arquivo ou pasta.
        limit: Maximo de imagens. Zero significa sem limite.

    Returns:
        Caminhos em ordem estavel.
    """
    if entry.is_file():
        return [entry]
    paths = list(container.image_source.list_images(entry))
    return paths[:limit] if limit else paths


def _build_parser() -> argparse.ArgumentParser:
    """Monta o analisador de argumentos com todos os subcomandos.

    Returns:
        O analisador configurado.
    """
    parser = argparse.ArgumentParser(
        prog="stratosphere",
        description="Deteccao de marcas: separa ONDE ha logo de QUAL logo e.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--banco", default="indice", help="pasta do indice vetorial")
    parser.add_argument("--cpu", action="store_true", help="forca execucao em CPU")
    parser.add_argument("-v", "--verboso", action="store_true")

    subcommands = parser.add_subparsers(dest="command", required=True)

    environment = subcommands.add_parser("ambiente", help="confere dependencias e banco")
    environment.set_defaults(function=_command_environment)

    database = subcommands.add_parser("banco", help="constroi o indice a partir das referencias")
    database.add_argument("--referencias", required=True, help="pasta <marca>/<variante>/arquivo")
    database.add_argument("--destino", default="indice", help="onde gravar o indice")
    database.set_defaults(function=_command_database)

    audit = subcommands.add_parser("auditar", help="marcas do banco parecidas demais")
    audit.add_argument("--limite", type=int, default=30, help="quantos pares exibir")
    audit.add_argument("--minimo", type=int, default=15, help="alerta abaixo desta contagem")
    audit.set_defaults(function=_command_audit)

    analyze = subcommands.add_parser("analisar", help="roda a pipeline numa imagem ou pasta")
    analyze.add_argument("--entrada", required=True, help="arquivo ou pasta de imagens")
    analyze.add_argument("--saida", default=None, help="json com o detalhe por regiao")
    analyze.add_argument("--limite", type=int, default=0, help="max de imagens (0 = todas)")
    analyze.add_argument(
        "--anotar", default=None, help="pasta onde gravar as imagens com as caixas desenhadas"
    )
    analyze.add_argument(
        "--tudo",
        action="store_true",
        help="inclui regioes rejeitadas no json de saida e nas imagens anotadas",
    )
    analyze.set_defaults(function=_command_analyze)

    return parser


if __name__ == "__main__":
    sys.exit(main())
