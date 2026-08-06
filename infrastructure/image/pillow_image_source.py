"""Acesso a imagem com Pillow.

Duas decisoes de preparo influenciam o resultado mais do que parecem:

**Letterbox em vez de esticar.** O recorte final e quadrado porque o codificador
espera quadrado, mas a proporcao original e preservada e o resto e preenchido.
Esticar um wordmark largo para caber num quadrado destroi exatamente a
caracteristica que o distingue de outro wordmark.

**Ampliar recorte minusculo.** Um recorte de 15x10 pixels esticado para 224 vira
borrao - mas um borrao *consistente*, e o banco tem referencias degradadas
justamente para casar com ele. Deixar em 15x10 e pior: o modelo recebe ruido de
interpolacao.

Typical usage:
    source = PillowImageSource()
    image = source.load(Path("foto.jpg"))
    crop = source.crop(image, box, margin=0.12, side=224)
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from application.ports.i_image_source import IImageSource, RgbImage
from domain.value_objects.box import Box

ACCEPTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)

# Cinza medio para o preenchimento do letterbox: neutro o bastante para nao criar
# borda de alto contraste, que o codificador leria como forma.
_FILL_COLOR: tuple[int, int, int] = (124, 116, 104)

# Redes sociais entregam JPEG truncado com frequencia. Melhor decodificar o que
# der do que perder a imagem inteira.
Image.MAX_IMAGE_PIXELS = 200_000_000


class PillowImageSource(IImageSource):
    """Implementacao de acesso a imagem sobre Pillow."""

    def __init__(self, min_side_to_upscale: int = 64) -> None:
        """Configura o limiar de ampliacao de recortes minusculos.

        Args:
            min_side_to_upscale: Recorte cujo maior lado for menor que isto e
                ampliado antes de seguir para o codificador.
        """
        self._min_side_to_upscale = min_side_to_upscale

    def load(self, path: Path) -> RgbImage:
        """Carrega uma imagem em RGB, com a rotacao de metadados aplicada.

        Args:
            path: Caminho do arquivo.

        Returns:
            Imagem em RGB.

        Raises:
            OSError: Se o arquivo nao existir ou nao puder ser decodificado.
        """
        file = Image.open(path)
        # `exif_transpose` devolve None quando nao ha metadado de orientacao.
        oriented = ImageOps.exif_transpose(file) or file
        return self._flatten(oriented)

    @staticmethod
    def _flatten(image: RgbImage) -> RgbImage:
        """Achata transparencia sobre fundo neutro antes de virar RGB.

        `convert("RGB")` puro **descarta** o canal alfa e mantem o RGB que
        estiver embaixo - que em arte vetorial exportada costuma ser preto. Uma
        logo escura sobre fundo transparente vira, entao, um retangulo preto
        solido. Medido no banco real: quatro referencias `oficial/*.png` tinham
        brilho medio 0.0, eram vetores praticamente identicos entre si, e a
        melhor correspondencia da logo oficial da nike era uma referencia de
        cimed a 0.951 - com zero nike no top-25.

        O fundo e o mesmo cinza do letterbox de proposito. Medido na mesma
        referencia: composta sobre este cinza ela puxa 19 de 25 vizinhos nike;
        sobre branco, 7 de 25. A uniformidade com o preenchimento que o resto da
        pipeline ja usa vale mais que o branco convencional de arte.

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

    def dimensions(self, image: RgbImage) -> tuple[int, int]:
        """Retorna `(largura, altura)` da imagem.

        Args:
            image: Imagem carregada.

        Returns:
            Largura e altura em pixels.
        """
        width, height = image.size
        return int(width), int(height)

    def crop(self, image: RgbImage, box: Box, margin: float, side: int) -> RgbImage:
        """Recorta com margem, amplia se minusculo, e devolve quadrado com letterbox.

        Args:
            image: Imagem de origem.
            box: Regiao a recortar.
            margin: Contexto proporcional em cada lado.
            side: Lado do quadrado de saida.

        Returns:
            Recorte quadrado pronto para o codificador.
        """
        width, height = self.dimensions(image)
        expanded = box.with_margin(margin, width, height) if margin > 0 else box
        cropped = image.crop((expanded.x1, expanded.y1, expanded.x2, expanded.y2))

        longest_side = max(cropped.size)
        if 0 < longest_side < self._min_side_to_upscale:
            factor = self._min_side_to_upscale / longest_side
            cropped = cropped.resize(
                (max(1, int(cropped.width * factor)), max(1, int(cropped.height * factor))),
                Image.Resampling.LANCZOS,
            )
        return self._letterbox(cropped, side)

    def edge_density(self, image: RgbImage, max_side: int = 256) -> float:
        """Mede a fracao de pixels com gradiente forte.

        Usa diferencas centrais numa versao reduzida da imagem - roda em
        microssegundos e nao exige biblioteca de visao.

        Args:
            image: Imagem a medir.
            max_side: Maior lado da versao reduzida usada na medida.

        Returns:
            Fracao entre 0.0 e 1.0 de pixels com gradiente acima do limiar.
        """
        width, height = self.dimensions(image)
        scale = max_side / max(width, height)
        if scale < 1.0:
            image = image.resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.BILINEAR,
            )

        gray = np.asarray(image.convert("L"), dtype=np.float32)
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return 0.0

        gradient_x = gray[1:-1, 2:] - gray[1:-1, :-2]
        gradient_y = gray[2:, 1:-1] - gray[:-2, 1:-1]
        magnitude = np.hypot(gradient_x, gradient_y)
        return float((magnitude > 24.0).mean())

    def list_images(self, folder: Path) -> tuple[Path, ...]:
        """Lista imagens da pasta, recursivamente e em ordem estavel.

        Args:
            folder: Raiz da busca.

        Returns:
            Caminhos ordenados. A ordem e contrato: reprodutibilidade depende dela.

        Raises:
            FileNotFoundError: Se a pasta nao existir.
        """
        if not folder.exists():
            raise FileNotFoundError(f"pasta nao existe: {folder}")
        return tuple(
            sorted(
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in ACCEPTED_EXTENSIONS
            )
        )

    def _letterbox(self, image: RgbImage, side: int) -> RgbImage:
        """Redimensiona preservando proporcao e centraliza num quadrado.

        Args:
            image: Imagem a ajustar.
            side: Lado do quadrado de saida.

        Returns:
            Quadrado de `side x side` com a imagem centralizada.
        """
        width, height = image.size
        if width == 0 or height == 0:
            return Image.new("RGB", (side, side), _FILL_COLOR)

        scale = side / max(width, height)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = image.resize(new_size, Image.Resampling.BICUBIC)

        canvas = Image.new("RGB", (side, side), _FILL_COLOR)
        canvas.paste(resized, ((side - new_size[0]) // 2, (side - new_size[1]) // 2))
        return canvas
