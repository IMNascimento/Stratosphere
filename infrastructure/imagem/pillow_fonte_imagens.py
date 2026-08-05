"""Acesso a imagem com Pillow.

Duas decisoes de preparo influenciam o resultado mais do que parecem:

**Letterbox em vez de esticar.** O recorte final e quadrado porque o codificador
espera quadrado, mas a proporcao original e preservada e o resto e preenchido.
Esticar um wordmark largo para caber num quadrado destroi exatamente a
caracteristica que o distingue de outro wordmark.

**Ampliar recorte minusculo.** Um recorte de 15x10 pixels esticado para 224 vira
borrao — mas um borrao *consistente*, e o banco tem referencias degradadas
justamente para casar com ele. Deixar em 15x10 e pior: o modelo recebe ruido de
interpolacao.

Typical usage:
    fonte = PillowFonteImagens()
    imagem = fonte.carregar(Path("foto.jpg"))
    recorte = fonte.recortar(imagem, caixa, margem=0.12, lado=224)
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from application.ports.i_fonte_de_imagens import IFonteDeImagens, ImagemRgb
from domain.value_objects.caixa import Caixa

EXTENSOES_ACEITAS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
)

# Cinza medio para o preenchimento do letterbox: neutro o bastante para nao criar
# borda de alto contraste, que o codificador leria como forma.
_COR_DE_PREENCHIMENTO: tuple[int, int, int] = (124, 116, 104)

# Redes sociais entregam JPEG truncado com frequencia. Melhor decodificar o que
# der do que perder a imagem inteira.
Image.MAX_IMAGE_PIXELS = 200_000_000


class PillowFonteImagens(IFonteDeImagens):
    """Implementacao de acesso a imagem sobre Pillow."""

    def __init__(self, lado_minimo_para_ampliar: int = 64) -> None:
        """Configura o limiar de ampliacao de recortes minusculos.

        Args:
            lado_minimo_para_ampliar: Recorte cujo maior lado for menor que isto
                e ampliado antes de seguir para o codificador.
        """
        self._lado_minimo_para_ampliar = lado_minimo_para_ampliar

    def carregar(self, caminho: Path) -> ImagemRgb:
        """Carrega uma imagem em RGB, com a rotacao de metadados aplicada.

        Args:
            caminho: Caminho do arquivo.

        Returns:
            Imagem em RGB.

        Raises:
            OSError: Se o arquivo nao existir ou nao puder ser decodificado.
        """
        arquivo = Image.open(caminho)
        # `exif_transpose` devolve None quando nao ha metadado de orientacao.
        orientada = ImageOps.exif_transpose(arquivo) or arquivo
        return orientada.convert("RGB")

    def dimensoes(self, imagem: ImagemRgb) -> tuple[int, int]:
        """Retorna `(largura, altura)` da imagem.

        Args:
            imagem: Imagem carregada.

        Returns:
            Largura e altura em pixels.
        """
        largura, altura = imagem.size
        return int(largura), int(altura)

    def recortar(self, imagem: ImagemRgb, caixa: Caixa, margem: float, lado: int) -> ImagemRgb:
        """Recorta com margem, amplia se minusculo, e devolve quadrado com letterbox.

        Args:
            imagem: Imagem de origem.
            caixa: Regiao a recortar.
            margem: Contexto proporcional em cada lado.
            lado: Lado do quadrado de saida.

        Returns:
            Recorte quadrado pronto para o codificador.
        """
        largura, altura = self.dimensoes(imagem)
        expandida = caixa.com_margem(margem, largura, altura) if margem > 0 else caixa
        recorte = imagem.crop((expandida.x1, expandida.y1, expandida.x2, expandida.y2))

        maior_lado = max(recorte.size)
        if 0 < maior_lado < self._lado_minimo_para_ampliar:
            fator = self._lado_minimo_para_ampliar / maior_lado
            recorte = recorte.resize(
                (max(1, int(recorte.width * fator)), max(1, int(recorte.height * fator))),
                Image.Resampling.LANCZOS,
            )
        return self._letterbox(recorte, lado)

    def densidade_de_bordas(self, imagem: ImagemRgb, lado_maximo: int = 256) -> float:
        """Mede a fracao de pixels com gradiente forte.

        Usa diferencas centrais numa versao reduzida da imagem — roda em
        microssegundos e nao exige biblioteca de visao.

        Args:
            imagem: Imagem a medir.
            lado_maximo: Maior lado da versao reduzida usada na medida.

        Returns:
            Fracao entre 0.0 e 1.0 de pixels com gradiente acima do limiar.
        """
        largura, altura = self.dimensoes(imagem)
        escala = lado_maximo / max(largura, altura)
        if escala < 1.0:
            imagem = imagem.resize(
                (max(1, int(largura * escala)), max(1, int(altura * escala))),
                Image.Resampling.BILINEAR,
            )

        cinza = np.asarray(imagem.convert("L"), dtype=np.float32)
        if cinza.shape[0] < 3 or cinza.shape[1] < 3:
            return 0.0

        gradiente_x = cinza[1:-1, 2:] - cinza[1:-1, :-2]
        gradiente_y = cinza[2:, 1:-1] - cinza[:-2, 1:-1]
        magnitude = np.hypot(gradiente_x, gradiente_y)
        return float((magnitude > 24.0).mean())

    def listar(self, pasta: Path) -> tuple[Path, ...]:
        """Lista imagens da pasta, recursivamente e em ordem estavel.

        Args:
            pasta: Raiz da busca.

        Returns:
            Caminhos ordenados. A ordem e contrato: reprodutibilidade depende dela.

        Raises:
            FileNotFoundError: Se a pasta nao existir.
        """
        if not pasta.exists():
            raise FileNotFoundError(f"pasta nao existe: {pasta}")
        return tuple(
            sorted(
                caminho
                for caminho in pasta.rglob("*")
                if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_ACEITAS
            )
        )

    def _letterbox(self, imagem: ImagemRgb, lado: int) -> ImagemRgb:
        """Redimensiona preservando proporcao e centraliza num quadrado.

        Args:
            imagem: Imagem a ajustar.
            lado: Lado do quadrado de saida.

        Returns:
            Quadrado de `lado x lado` com a imagem centralizada.
        """
        largura, altura = imagem.size
        if largura == 0 or altura == 0:
            return Image.new("RGB", (lado, lado), _COR_DE_PREENCHIMENTO)

        escala = lado / max(largura, altura)
        novo = (max(1, int(round(largura * escala))), max(1, int(round(altura * escala))))
        redimensionada = imagem.resize(novo, Image.Resampling.BICUBIC)

        tela = Image.new("RGB", (lado, lado), _COR_DE_PREENCHIMENTO)
        tela.paste(redimensionada, ((lado - novo[0]) // 2, (lado - novo[1]) // 2))
        return tela
