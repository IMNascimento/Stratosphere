"""Contrato de acesso a imagem — leitura, recorte e medida de estrutura.

Existe como porta para que o dominio e os casos de uso nao conhecam biblioteca
de imagem. `ImagemRgb` e deliberadamente opaco: quem consome so repassa o objeto
entre as camadas, nunca inspeciona.

Typical usage:
    imagem = fonte.carregar(caminho)
    recorte = fonte.recortar(imagem, caixa, margem=0.12, lado=224)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from domain.value_objects.caixa import Caixa

# Imagem em memoria, no formato da biblioteca concreta. Opaco de proposito:
# tratar como tipo nominal aqui obrigaria dominio e aplicacao a importar a
# biblioteca de imagem, que e exatamente o acoplamento que a porta evita.
ImagemRgb = Any


class IFonteDeImagens(ABC):
    """Acesso a imagens em disco e operacoes de recorte."""

    @abstractmethod
    def carregar(self, caminho: Path) -> ImagemRgb:
        """Carrega uma imagem em RGB, com a rotacao de metadados ja aplicada.

        Args:
            caminho: Caminho do arquivo.

        Returns:
            A imagem pronta para uso, sempre em RGB.

        Raises:
            OSError: Se o arquivo nao existir ou nao puder ser decodificado.
        """
        ...

    @abstractmethod
    def dimensoes(self, imagem: ImagemRgb) -> tuple[int, int]:
        """Retorna `(largura, altura)` da imagem, em pixels.

        Args:
            imagem: Imagem carregada.

        Returns:
            Tupla com largura e altura.
        """
        ...

    @abstractmethod
    def recortar(self, imagem: ImagemRgb, caixa: Caixa, margem: float, lado: int) -> ImagemRgb:
        """Recorta a regiao com margem e devolve um quadrado de lado fixo.

        O quadrado e produzido com letterbox, preservando a proporcao original.
        Esticar um wordmark largo para caber num quadrado destroi exatamente a
        caracteristica que o distingue de outro.

        Args:
            imagem: Imagem de origem.
            caixa: Regiao a recortar, em coordenadas da imagem original.
            margem: Contexto proporcional a acrescentar em cada lado.
            lado: Lado do quadrado de saida, em pixels.

        Returns:
            Recorte quadrado, pronto para o codificador.
        """
        ...

    @abstractmethod
    def densidade_de_bordas(self, imagem: ImagemRgb) -> float:
        """Mede a fracao de pixels com gradiente forte.

        E um proxy barato de "esta imagem tem alguma estrutura?", usado para
        descartar imagem vazia antes de gastar uma passada de detector.

        Args:
            imagem: Imagem a medir.

        Returns:
            Fracao entre 0.0 e 1.0.
        """
        ...

    @abstractmethod
    def listar(self, pasta: Path) -> tuple[Path, ...]:
        """Lista imagens de uma pasta, recursivamente.

        Args:
            pasta: Raiz da busca.

        Returns:
            Caminhos em ordem estavel — a ordem e contrato, porque
            reprodutibilidade depende dela.

        Raises:
            FileNotFoundError: Se a pasta nao existir.
        """
        ...
