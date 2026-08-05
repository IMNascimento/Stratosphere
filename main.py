"""Atalho para a CLI, para quem prefere `python main.py` ao script instalado.

O ponto de entrada de verdade e `entrypoints.cli.main`. Este arquivo existe
apenas porque `main.py` na raiz e a convencao da estrutura do projeto.

Typical usage:
    poetry run python main.py analisar --entrada fotos/
"""

import sys

from entrypoints.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
