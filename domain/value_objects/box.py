"""Value object de regiao retangular numa imagem.

Todas as coordenadas sao absolutas e em pixels da imagem ORIGINAL. O detector
trabalha numa versao redimensionada da imagem; converter de volta e
responsabilidade do adaptador de infraestrutura. Manter uma unica convencao no
dominio evita o defeito classico de recorte deslocado quando se muda a resolucao
de inferencia.

Typical usage:
    box = Box(x1=10, y1=20, x2=110, y2=80)
    enlarged = box.with_margin(0.12, max_width=1920, max_height=1080)
    overlap = box.intersection_over_union(other)
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import InvalidBoxError


@dataclass(frozen=True)
class Box:
    """Regiao retangular em coordenadas absolutas da imagem original.

    Attributes:
        x1: Borda esquerda, em pixels. Nao negativa.
        y1: Borda superior, em pixels. Nao negativa.
        x2: Borda direita, em pixels. Estritamente maior que x1.
        y2: Borda inferior, em pixels. Estritamente maior que y1.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        """Valida a invariante de retangulo com area positiva.

        Raises:
            InvalidBoxError: Se alguma coordenada for negativa, ou se a caixa
                tiver largura ou altura nao positiva.
        """
        if self.x1 < 0 or self.y1 < 0:
            raise InvalidBoxError(f"coordenadas nao podem ser negativas: ({self.x1}, {self.y1})")
        if self.x2 <= self.x1:
            raise InvalidBoxError(f"x2 deve ser maior que x1: {self.x2} <= {self.x1}")
        if self.y2 <= self.y1:
            raise InvalidBoxError(f"y2 deve ser maior que y1: {self.y2} <= {self.y1}")

    @property
    def width(self) -> int:
        """Largura da caixa em pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Altura da caixa em pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area da caixa em pixels quadrados."""
        return self.width * self.height

    @property
    def shorter_side(self) -> int:
        """Menor entre largura e altura, em pixels.

        E a dimensao que decide se a regiao ainda carrega forma reconhecivel:
        um recorte de 200x4 pixels tem area razoavel e nenhuma informacao.
        """
        return min(self.width, self.height)

    def intersection_over_union(self, other: "Box") -> float:
        """Calcula o IoU entre esta caixa e outra.

        Args:
            other: Caixa a comparar, nas mesmas coordenadas de imagem.

        Returns:
            Valor entre 0.0 e 1.0. Zero quando nao ha sobreposicao.
        """
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        common_width = max(0, x2 - x1)
        common_height = max(0, y2 - y1)
        intersection = common_width * common_height
        if intersection == 0:
            return 0.0

        union = self.area + other.area - intersection
        return intersection / union

    def containment_with(self, other: "Box") -> float:
        """Quanto da MENOR das duas caixas esta coberto pela outra.

        Existe porque o IoU e cego para aninhamento, e aninhamento e o caso
        comum quando um detector de vocabulario aberto olha o mesmo logo: um
        recorte justo no simbolo e outro folgado em volta descrevem a mesma
        coisa, mas dividem a area da uniao e ficam com IoU baixo. Medido em
        imagem real, um recorte inteiramente dentro do outro deu IoU 0.32 —
        longe de qualquer limiar de supressao razoavel — e contencao 1.0.

        Args:
            other: Caixa a comparar, nas mesmas coordenadas de imagem.

        Returns:
            Valor entre 0.0 e 1.0. Um significa que uma das caixas esta
            inteiramente dentro da outra, em qualquer das duas direcoes.
        """
        common_width = max(0, min(self.x2, other.x2) - max(self.x1, other.x1))
        common_height = max(0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = common_width * common_height
        if intersection == 0:
            return 0.0
        return intersection / min(self.area, other.area)

    def with_margin(self, fraction: float, max_width: int, max_height: int) -> "Box":
        """Devolve a caixa expandida proporcionalmente, presa aos limites da imagem.

        A margem existe porque o detector tende a colar a caixa no logo, e um
        simbolo recortado exatamente na borda perde a silhueta que o codificador
        usa para reconhece-lo. Margem demais tem o efeito oposto: o vetor passa a
        descrever a camiseta em vez do logo.

        Args:
            fraction: Proporcao de cada lado a acrescentar. `0.12` acrescenta 12%
                da largura de cada lado horizontal e 12% da altura de cada lado
                vertical. Deve ser nao negativa.
            max_width: Largura da imagem, usada como teto para x2.
            max_height: Altura da imagem, usada como teto para y2.

        Returns:
            Nova caixa expandida e presa a `[0, max_width] x [0, max_height]`.

        Raises:
            InvalidBoxError: Se `fraction` for negativa, ou se os limites da
                imagem forem menores que a origem da caixa.
        """
        if fraction < 0:
            raise InvalidBoxError(f"fracao de margem nao pode ser negativa: {fraction}")

        margin_x = int(round(self.width * fraction))
        margin_y = int(round(self.height * fraction))

        return Box(
            x1=max(0, self.x1 - margin_x),
            y1=max(0, self.y1 - margin_y),
            x2=min(max_width, self.x2 + margin_x),
            y2=min(max_height, self.y2 + margin_y),
        )
