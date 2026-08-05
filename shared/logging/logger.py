"""Logger do projeto.

Camada fina sobre `logging` da biblioteca padrao. Existe para que nenhum modulo
precise decidir formato de saida por conta propria, e para que a configuracao
aconteca uma vez so, no entrypoint.

Typical usage:
    configurar_logging("INFO")
    log = obter_logger(__name__)
    log.info("banco carregado: %d referencias", banco.tamanho())
"""

import logging
import sys

_FORMATO = "%(levelname)-7s %(name)-28s %(message)s"
_NIVEIS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


def configurar_logging(nivel: str = "INFO") -> None:
    """Configura a saida de log do processo. Idempotente.

    Args:
        nivel: Nome do nivel, entre DEBUG, INFO, WARNING e ERROR.

    Raises:
        ValueError: Se o nivel nao existir.
    """
    if nivel.upper() not in _NIVEIS:
        raise ValueError(f"nivel {nivel!r} desconhecido. Validos: {sorted(_NIVEIS)}")

    raiz = logging.getLogger()
    if raiz.handlers:
        raiz.setLevel(nivel.upper())
        return

    manipulador = logging.StreamHandler(sys.stderr)
    manipulador.setFormatter(logging.Formatter(_FORMATO))
    raiz.addHandler(manipulador)
    raiz.setLevel(nivel.upper())


def obter_logger(nome: str) -> logging.Logger:
    """Retorna o logger daquele modulo.

    Args:
        nome: Normalmente `__name__` do modulo chamador.

    Returns:
        O logger correspondente.
    """
    return logging.getLogger(nome)
