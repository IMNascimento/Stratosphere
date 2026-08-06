"""Contrato de acesso a imagem - leitura, recorte e medida de estrutura.

Existe como porta para que o dominio e os casos de uso nao conhecam biblioteca
de imagem. `RgbImage` e deliberadamente opaco: quem consome so repassa o objeto
entre as camadas, nunca inspeciona.

Typical usage:
    image = source.load(path)
    crop = source.crop(image, box, margin=0.12, side=224)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from domain.value_objects.box import Box

# Imagem em memoria, no formato da biblioteca concreta. Opaco de proposito:
# tratar como tipo nominal aqui obrigaria dominio e aplicacao a importar a
# biblioteca de imagem, que e exatamente o acoplamento que a porta evita.
RgbImage = Any


class IImageSource(ABC):
    """Acesso a imagens em disco e operacoes de recorte."""

    @abstractmethod
    def load(self, path: Path) -> RgbImage:
        """Carrega uma imagem em RGB, com a rotacao de metadados ja aplicada.

        Args:
            path: Caminho do arquivo.

        Returns:
            A imagem pronta para uso, sempre em RGB.

        Raises:
            OSError: Se o arquivo nao existir ou nao puder ser decodificado.
        """
        ...

    @abstractmethod
    def dimensions(self, image: RgbImage) -> tuple[int, int]:
        """Retorna `(largura, altura)` da imagem, em pixels.

        Args:
            image: Imagem carregada.

        Returns:
            Tupla com largura e altura.
        """
        ...

    @abstractmethod
    def crop(self, image: RgbImage, box: Box, margin: float, side: int) -> RgbImage:
        """Recorta a regiao com margem e devolve um quadrado de lado fixo.

        O quadrado e produzido com letterbox, preservando a proporcao original.
        Esticar um wordmark largo para caber num quadrado destroi exatamente a
        caracteristica que o distingue de outro.

        Args:
            image: Imagem de origem.
            box: Regiao a recortar, em coordenadas da imagem original.
            margin: Contexto proporcional a acrescentar em cada lado.
            side: Lado do quadrado de saida, em pixels.

        Returns:
            Recorte quadrado, pronto para o codificador.
        """
        ...

    @abstractmethod
    def edge_density(self, image: RgbImage) -> float:
        """Mede a fracao de pixels com gradiente forte.

        E um proxy barato de "esta imagem tem alguma estrutura?", usado para
        descartar imagem vazia antes de gastar uma passada de detector.

        Args:
            image: Imagem a medir.

        Returns:
            Fracao entre 0.0 e 1.0.
        """
        ...

    @abstractmethod
    def list_images(self, folder: Path) -> tuple[Path, ...]:
        """Lista imagens de uma pasta, recursivamente.

        Args:
            folder: Raiz da busca.

        Returns:
            Caminhos em ordem estavel - a ordem e contrato, porque
            reprodutibilidade depende dela.

        Raises:
            FileNotFoundError: Se a pasta nao existir.
        """
        ...
