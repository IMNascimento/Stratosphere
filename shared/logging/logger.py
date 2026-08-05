"""Logger do projeto.

Camada fina sobre `logging` da biblioteca padrao. Existe para que nenhum modulo
precise decidir formato de saida por conta propria, e para que a configuracao
aconteca uma vez so, no entrypoint.

Typical usage:
    configure_logging("INFO")
    log = get_logger(__name__)
    log.info("banco carregado: %d referencias", database.size())
"""

import logging
import sys

_FORMAT = "%(levelname)-7s %(name)-28s %(message)s"
_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


def configure_logging(level: str = "INFO") -> None:
    """Configura a saida de log do processo. Idempotente.

    Args:
        level: Nome do nivel, entre DEBUG, INFO, WARNING e ERROR.

    Raises:
        ValueError: Se o nivel nao existir.
    """
    if level.upper() not in _LEVELS:
        raise ValueError(f"nivel {level!r} desconhecido. Validos: {sorted(_LEVELS)}")

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Retorna o logger daquele modulo.

    Args:
        name: Normalmente `__name__` do modulo chamador.

    Returns:
        O logger correspondente.
    """
    return logging.getLogger(name)
