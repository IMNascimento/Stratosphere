"""Desenha as caixas decididas sobre a imagem, para conferencia no olho.

Nao participa da pipeline: e saida de diagnostico. A analise ja terminou quando
esta camada entra, e nada do que acontece aqui muda decisao nenhuma.

**Por que recarrega a imagem pela porta em vez de abrir com Pillow direto.** As
caixas estao em coordenadas da imagem ORIGINAL, e o `IImageSource.load` aplica a
rotacao EXIF antes de medir. Abrir o arquivo por fora dessa porta pularia a
rotacao, e toda foto tirada de celular na vertical sairia com as caixas
deslocadas — o tipo de defeito que so aparece em algumas fotos e faz duvidar do
detector, nao do desenho.

A cor vem da fila, nao da marca: o que se confere no olho e "esta regiao foi
para o lugar certo?", e a fila e a resposta a essa pergunta.

Typical usage:
    annotator = PillowAnnotator(source)
    annotator.annotate(Path("foto.jpg"), regions, Path("runs/anotadas/foto.jpg"))
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import ImageDraw, ImageFont

from application.dtos.analysis import RegionOutput
from application.ports.i_image_source import IImageSource, RgbImage
from domain.enums.queue import Queue

# Cor por fila. Verde entra sozinho, ambar e azul pedem humano, roxo e a
# referencia que falta, cinza e marca de fora do portfolio, vermelho e descarte.
_QUEUE_COLORS: dict[Queue, tuple[int, int, int]] = {
    Queue.AUTO_ACCEPT: (46, 160, 67),
    Queue.REVIEW: (219, 154, 4),
    Queue.CONFUSION: (47, 129, 247),
    Queue.ORPHAN: (163, 113, 247),
    Queue.NEGATIVE: (139, 148, 158),
    Queue.AUTO_REJECT: (248, 81, 73),
}
_FALLBACK_COLOR: tuple[int, int, int] = (200, 200, 200)
_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)

# Espessura e fonte em proporcao do menor lado: caixa de 3px some numa foto de
# 4000px e engole o logo numa miniatura.
_LINE_RATIO = 0.004
_FONT_RATIO = 0.018
_MIN_LINE_WIDTH = 2
_MIN_FONT_SIZE = 12
_LABEL_PADDING = 3

# JPEG nao aceita alpha e e o formato da maioria das entradas.
_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})
_JPEG_QUALITY = 92


class PillowAnnotator:
    """Grava uma copia da imagem com as caixas e os rotulos desenhados."""

    def __init__(self, source: IImageSource) -> None:
        """Recebe a porta de imagem usada pela analise.

        Args:
            source: Mesma fonte que a pipeline usa. Garante que a imagem
                desenhada passe pela mesma rotacao EXIF que produziu as caixas.
        """
        self._source = source

    def annotate(self, source: Path, regions: Sequence[RegionOutput], destination: Path) -> Path:
        """Desenha as regioes sobre a imagem e grava no destino.

        Args:
            source: Arquivo analisado.
            regions: Regioes a desenhar. Vazio grava a imagem sem marcacao —
                util para ver que a pipeline nao achou nada ali.
            destination: Arquivo de saida. Pastas intermediarias sao criadas.

        Returns:
            O caminho gravado.

        Raises:
            OSError: Se a imagem nao puder ser lida ou gravada.
        """
        image: RgbImage = self._source.load(source)
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)

        width, height = canvas.size
        smaller_side = min(width, height)
        line_width = max(_MIN_LINE_WIDTH, round(smaller_side * _LINE_RATIO))
        font = self._font(max(_MIN_FONT_SIZE, round(smaller_side * _FONT_RATIO)))

        for region in regions:
            color = _QUEUE_COLORS.get(region.queue, _FALLBACK_COLOR)
            draw.rectangle(region.box, outline=color, width=line_width)
            self._label(draw, region, color, font, width, height)

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in _JPEG_SUFFIXES:
            canvas.save(destination, quality=_JPEG_QUALITY)
        else:
            canvas.save(destination)
        return destination

    # -- etapas internas ---------------------------------------------------

    def _label(
        self,
        draw: Any,
        region: RegionOutput,
        color: tuple[int, int, int],
        font: Any,
        width: int,
        height: int,
    ) -> None:
        """Escreve marca, fila e pontuacao junto da caixa.

        O rotulo e preso aos quatro lados da imagem: sobe para cima da caixa
        quando ha espaco, desce para dentro quando ela encosta no topo, e recua
        para a esquerda quando passaria da borda direita. Logo em canto e o caso
        comum — patrocinio costuma ficar na quina —, e rotulo cortado na borda
        nao se le.

        Args:
            draw: Contexto de desenho.
            region: Regiao sendo desenhada.
            color: Cor da fila, usada no fundo do rotulo.
            font: Fonte ja dimensionada.
            width: Largura da imagem.
            height: Altura da imagem.
        """
        x1, y1, _, y2 = region.box
        text = f"{region.brand or '?'} · {region.queue.value} · {region.score:.2f}"

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        label_width = right - left + 2 * _LABEL_PADDING
        label_height = bottom - top + 2 * _LABEL_PADDING

        label_left = max(0, min(x1, width - label_width))
        label_top = y1 - label_height
        if label_top < 0:
            label_top = min(y1, height - label_height)
        label_top = max(0, min(label_top, max(0, y2)))

        draw.rectangle(
            (label_left, label_top, label_left + label_width, label_top + label_height),
            fill=color,
        )
        draw.text(
            (label_left + _LABEL_PADDING - left, label_top + _LABEL_PADDING - top),
            text,
            fill=_TEXT_COLOR,
            font=font,
        )

    @staticmethod
    def _font(size: int) -> Any:
        """Devolve a fonte embutida do Pillow no tamanho pedido.

        Args:
            size: Altura desejada, em pixels.

        Returns:
            A fonte. Versoes antigas do Pillow ignoram o tamanho e devolvem a
            fonte fixa — o rotulo fica pequeno, mas continua legivel, e isso e
            melhor que quebrar a anotacao inteira.
        """
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def annotated_path(source: Path, root: Path, destination_folder: Path) -> Path:
    """Calcula onde gravar a copia anotada de uma imagem.

    Espelha a estrutura da entrada dentro da pasta de saida. Sem isso, duas
    imagens com o mesmo nome em subpastas diferentes — `nike/01.jpg` e
    `itau/01.jpg` — gravariam uma por cima da outra, e a segunda apagaria a
    primeira em silencio.

    Args:
        source: Arquivo analisado.
        root: Raiz da entrada. Quando a entrada e um arquivo, e o proprio.
        destination_folder: Pasta onde gravar.

    Returns:
        Caminho completo do arquivo de saida.
    """
    if root.is_dir():
        try:
            return destination_folder / source.relative_to(root)
        except ValueError:
            return destination_folder / source.name
    return destination_folder / source.name
