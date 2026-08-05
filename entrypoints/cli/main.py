"""Interface de linha de comando do Stratosphere.

Quatro subcomandos, na ordem em que sao usados numa instalacao nova:

    ambiente   confere se da para rodar antes de baixar peso de modelo
    banco      pasta de referencias -> indice vetorial
    auditar    lista marcas do banco que se parecem demais entre si
    analisar   roda a pipeline numa imagem ou pasta

O entrypoint so fala com o container e com os casos de uso. Ele nao conhece
OWLv2, DINOv2 nem SIFT — trocar qualquer um deles nao muda nada aqui.

Typical usage:
    poetry run stratosphere banco --referencias marcas/ --destino indice/
    poetry run stratosphere analisar --entrada fotos/ --banco indice/
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from application.dtos.analise import AnalisarImagemCommand, ConstruirBancoCommand
from config.settings import AppConfig
from domain.enums.fila import Fila
from domain.exceptions.domain_exceptions import DomainError
from infrastructure.container.container import Container, construir_container
from shared.logging.logger import configurar_logging, obter_logger

log = obter_logger("stratosphere")

_CODIGO_DE_ERRO_DE_USO = 2
_CODIGO_DE_ERRO_DE_DOMINIO = 1

# Ordem de exibicao das filas: das que exigem acao humana para as que nao exigem.
_ORDEM_DAS_FILAS = (
    Fila.AUTO_ACEITE,
    Fila.CONFUSAO,
    Fila.ORFAO,
    Fila.REVISAO,
    Fila.NEGATIVA,
    Fila.AUTO_REJEICAO,
)


def main(argumentos: list[str] | None = None) -> int:
    """Ponto de entrada da CLI.

    Args:
        argumentos: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        Codigo de saida: 0 em sucesso, 1 em erro de dominio, 2 em erro de uso.
    """
    analisador = _montar_analisador()
    opcoes = analisador.parse_args(argumentos)
    configurar_logging("DEBUG" if opcoes.verboso else "INFO")

    try:
        return int(opcoes.funcao(opcoes))
    except DomainError as erro:
        log.error("%s", erro)
        return _CODIGO_DE_ERRO_DE_DOMINIO
    except FileNotFoundError as erro:
        log.error("%s", erro)
        return _CODIGO_DE_ERRO_DE_DOMINIO
    except KeyboardInterrupt:
        log.warning("interrompido")
        return 130


# -- subcomandos -----------------------------------------------------------


def _comando_ambiente(opcoes: argparse.Namespace) -> int:
    """Confere se o ambiente aguenta rodar a pipeline.

    Roda antes de qualquer download de peso: transforma meia hora de espera
    seguida de erro num diagnostico de um segundo.

    Args:
        opcoes: Opcoes ja analisadas.

    Returns:
        0 se tudo que a pipeline exige esta presente, 1 caso contrario.
    """
    tudo_certo = True
    print("=== dependencias ===")
    for modulo, dica in (
        ("torch", "poetry add torch"),
        ("transformers", "poetry add transformers"),
        ("cv2", "poetry add opencv-contrib-python-headless"),
        ("PIL", "poetry add pillow"),
        ("numpy", "poetry add numpy"),
    ):
        try:
            importado = __import__(modulo)
            versao = getattr(importado, "__version__", "ok")
            print(f"  {modulo:<14} {versao}")
        except ImportError:
            print(f"  {modulo:<14} AUSENTE  ->  {dica}")
            tudo_certo = False

    print("\n=== aceleracao ===")
    try:
        import torch

        if torch.cuda.is_available():
            propriedades = torch.cuda.get_device_properties(0)
            memoria = propriedades.total_memory / 1024**3
            print(f"  GPU            {propriedades.name} | {memoria:.1f} GB")
        else:
            print("  GPU            indisponivel — a pipeline roda em CPU, bem mais devagar")
    except ImportError:
        print("  GPU            nao verificavel sem torch")

    banco = Path(opcoes.banco)
    print("\n=== banco de referencia ===")
    if (banco / "referencias.json").exists():
        metadados = json.loads((banco / "referencias.json").read_text(encoding="utf-8"))
        manifesto = metadados["manifesto"]
        print(f"  referencias    {manifesto['total_de_referencias']}")
        print(f"  dimensao       {manifesto['dimensao']}")
        print(f"  codificador    {manifesto['assinatura_do_codificador']}")
        print(f"  por marca      {metadados['referencias_por_marca']}")
    else:
        print(f"  AUSENTE em {banco}")
        print("  construa com:  poetry run stratosphere banco --referencias <pasta>")

    print("\n" + ("ambiente pronto." if tudo_certo else "ambiente INCOMPLETO — ver acima."))
    return 0 if tudo_certo else 1


def _comando_banco(opcoes: argparse.Namespace) -> int:
    """Constroi o indice vetorial a partir da pasta de referencias.

    Args:
        opcoes: Opcoes ja analisadas.

    Returns:
        0 em sucesso.

    Raises:
        ReferenciasNaoEncontradasError: Se a pasta nao tiver imagem utilizavel.
    """
    container = _container(opcoes)
    saida = container.construir_banco.execute(
        ConstruirBancoCommand(
            pasta_de_referencias=Path(opcoes.referencias), destino=Path(opcoes.destino)
        )
    )

    print(f"\nbanco construido em {opcoes.destino}")
    print(
        f"  {saida.total_de_referencias} referencias de {len(saida.referencias_por_marca)} marcas"
    )
    if saida.descartadas_por_redundancia:
        print(
            f"  {saida.descartadas_por_redundancia} descartadas por redundancia "
            f"(quase identicas a outra da mesma marca)"
        )
    print()
    for marca, quantidade in saida.referencias_por_marca.items():
        alerta = "   <- poucas referencias" if quantidade < 15 else ""
        print(f"  {marca:<24} {quantidade:>4}{alerta}")

    print("\nProximo passo — SEMPRE audite antes de usar:")
    print("  poetry run stratosphere auditar")
    return 0


def _comando_auditar(opcoes: argparse.Namespace) -> int:
    """Lista marcas do banco cujas referencias se parecem demais.

    Cada par reportado e um falso positivo esperando acontecer, ou um grupo de
    confusao ainda nao declarado na configuracao.

    Args:
        opcoes: Opcoes ja analisadas.

    Returns:
        0 em sucesso, 1 se o banco nao existir.
    """
    container = _container(opcoes)
    if container.auditar_banco is None:
        log.error("banco ausente em %s — construa com `stratosphere banco`", opcoes.banco)
        return _CODIGO_DE_ERRO_DE_DOMINIO

    saida = container.auditar_banco.execute()
    print(
        f"banco: {saida.total_de_referencias} referencias, "
        f"{len(saida.referencias_por_marca)} marcas\n"
    )

    magras = container.auditar_banco.marcas_com_poucas_referencias(opcoes.minimo)
    if magras:
        print("marcas com poucas referencias (esperar orfaos em vez de acertos):")
        for marca, quantidade in magras.items():
            print(f"  {marca:<24} {quantidade}")
        print()

    if not saida.pares_confundiveis:
        print("nenhum par de marcas diferentes acima do limiar. Banco limpo.")
        return 0

    print(f"{len(saida.pares_confundiveis)} pares de marcas DIFERENTES parecidas demais:")
    for marca_a, marca_b, similaridade in saida.pares_confundiveis[: opcoes.limite]:
        print(f"  {similaridade:.3f}  {marca_a} x {marca_b}")
    print(
        "\nCada par e um falso positivo agendado. Remova a referencia ruim, ou "
        "declare o grupo em ConfusaoConfig.grupos."
    )
    return 0


def _comando_analisar(opcoes: argparse.Namespace) -> int:
    """Roda a pipeline numa imagem ou numa pasta de imagens.

    Args:
        opcoes: Opcoes ja analisadas.

    Returns:
        0 em sucesso, 1 se o banco nao existir.
    """
    container = _container(opcoes)
    if container.analisar_imagem is None:
        log.error("banco ausente em %s — construa com `stratosphere banco`", opcoes.banco)
        return _CODIGO_DE_ERRO_DE_DOMINIO

    caminhos = _resolver_entradas(container, Path(opcoes.entrada), opcoes.limite)
    if not caminhos:
        log.error("nenhuma imagem em %s", opcoes.entrada)
        return _CODIGO_DE_ERRO_DE_DOMINIO

    log.info("analisando %d imagens", len(caminhos))
    total_por_fila: dict[str, int] = {}
    linhas: list[dict[str, object]] = []

    for posicao, caminho in enumerate(caminhos, start=1):
        saida = container.analisar_imagem.execute(AnalisarImagemCommand(caminho=caminho))
        for nome_da_fila, quantidade in saida.contagem_por_fila().items():
            total_por_fila[nome_da_fila] = total_por_fila.get(nome_da_fila, 0) + quantidade

        marcas = saida.marcas_aceitas()
        linhas.append(
            {
                "imagem": str(caminho),
                "descartada_por": saida.descartada_por,
                "marcas_aceitas": list(marcas),
                "regioes": [
                    {
                        "identificador": regiao.identificador,
                        "caixa": list(regiao.caixa),
                        "fila": regiao.fila.value,
                        "marca": regiao.marca,
                        "pontuacao": round(regiao.pontuacao, 4),
                        "similaridade": round(regiao.similaridade, 4),
                        "margem": round(regiao.margem, 4),
                        "inliers": regiao.inliers,
                        "motivos": list(regiao.motivos),
                    }
                    for regiao in saida.regioes
                    if regiao.fila is not Fila.AUTO_REJEICAO or opcoes.tudo
                ],
            }
        )
        if marcas:
            print(f"  {caminho.name[:56]:<58} {', '.join(marcas)}")
        if posicao % 25 == 0:
            log.info("  %d/%d imagens", posicao, len(caminhos))

    print(f"\n{len(caminhos)} imagens analisadas")
    for fila in _ORDEM_DAS_FILAS:
        quantidade = total_por_fila.get(fila.value, 0)
        if quantidade:
            marcador = "  <- exige humano" if fila.exige_humano else ""
            print(f"  {fila.value:<16} {quantidade:>6}{marcador}")

    if opcoes.saida:
        destino = Path(opcoes.saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(linhas, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\ndetalhe por regiao: {destino}")
    return 0


# -- apoio -----------------------------------------------------------------


def _container(opcoes: argparse.Namespace) -> Container:
    """Monta o container com a configuracao vinda da linha de comando.

    Args:
        opcoes: Opcoes ja analisadas.

    Returns:
        O container pronto.
    """
    config = AppConfig()
    if getattr(opcoes, "cpu", False):
        config = replace(config, dispositivo="cpu", precisao="float32")
    return construir_container(config, Path(opcoes.banco))


def _resolver_entradas(container: Container, entrada: Path, limite: int) -> list[Path]:
    """Resolve o argumento de entrada em uma lista de imagens.

    Args:
        container: Container, usado para listar pastas.
        entrada: Arquivo ou pasta.
        limite: Maximo de imagens. Zero significa sem limite.

    Returns:
        Caminhos em ordem estavel.
    """
    if entrada.is_file():
        return [entrada]
    caminhos = list(container.fonte_de_imagens.listar(entrada))
    return caminhos[:limite] if limite else caminhos


def _montar_analisador() -> argparse.ArgumentParser:
    """Monta o analisador de argumentos com todos os subcomandos.

    Returns:
        O analisador configurado.
    """
    analisador = argparse.ArgumentParser(
        prog="stratosphere",
        description="Deteccao de marcas: separa ONDE ha logo de QUAL logo e.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    analisador.add_argument("--banco", default="indice", help="pasta do indice vetorial")
    analisador.add_argument("--cpu", action="store_true", help="forca execucao em CPU")
    analisador.add_argument("-v", "--verboso", action="store_true")

    subcomandos = analisador.add_subparsers(dest="comando", required=True)

    ambiente = subcomandos.add_parser("ambiente", help="confere dependencias e banco")
    ambiente.set_defaults(funcao=_comando_ambiente)

    banco = subcomandos.add_parser("banco", help="constroi o indice a partir das referencias")
    banco.add_argument("--referencias", required=True, help="pasta <marca>/<variante>/arquivo")
    banco.add_argument("--destino", default="indice", help="onde gravar o indice")
    banco.set_defaults(funcao=_comando_banco)

    auditar = subcomandos.add_parser("auditar", help="marcas do banco parecidas demais")
    auditar.add_argument("--limite", type=int, default=30, help="quantos pares exibir")
    auditar.add_argument("--minimo", type=int, default=15, help="alerta abaixo desta contagem")
    auditar.set_defaults(funcao=_comando_auditar)

    analisar = subcomandos.add_parser("analisar", help="roda a pipeline numa imagem ou pasta")
    analisar.add_argument("--entrada", required=True, help="arquivo ou pasta de imagens")
    analisar.add_argument("--saida", default=None, help="json com o detalhe por regiao")
    analisar.add_argument("--limite", type=int, default=0, help="max de imagens (0 = todas)")
    analisar.add_argument(
        "--tudo", action="store_true", help="inclui regioes rejeitadas no json de saida"
    )
    analisar.set_defaults(funcao=_comando_analisar)

    return analisador


if __name__ == "__main__":
    sys.exit(main())
