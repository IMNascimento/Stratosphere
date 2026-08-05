"""Value object de regiao retangular numa imagem.

Todas as coordenadas sao absolutas e em pixels da imagem ORIGINAL. O detector
trabalha numa versao redimensionada da imagem; converter de volta e
responsabilidade do adaptador de infraestrutura. Manter uma unica convencao no
dominio evita o defeito classico de recorte deslocado quando se muda a resolucao
de inferencia.

Typical usage:
    caixa = Caixa(x1=10, y1=20, x2=110, y2=80)
    ampliada = caixa.com_margem(0.12, largura_max=1920, altura_max=1080)
    sobreposicao = caixa.intersecao_sobre_uniao(outra)
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import CaixaInvalidaError


@dataclass(frozen=True)
class Caixa:
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
            CaixaInvalidaError: Se alguma coordenada for negativa, ou se a caixa
                tiver largura ou altura nao positiva.
        """
        if self.x1 < 0 or self.y1 < 0:
            raise CaixaInvalidaError(f"coordenadas nao podem ser negativas: ({self.x1}, {self.y1})")
        if self.x2 <= self.x1:
            raise CaixaInvalidaError(f"x2 deve ser maior que x1: {self.x2} <= {self.x1}")
        if self.y2 <= self.y1:
            raise CaixaInvalidaError(f"y2 deve ser maior que y1: {self.y2} <= {self.y1}")

    @property
    def largura(self) -> int:
        """Largura da caixa em pixels."""
        return self.x2 - self.x1

    @property
    def altura(self) -> int:
        """Altura da caixa em pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area da caixa em pixels quadrados."""
        return self.largura * self.altura

    @property
    def menor_lado(self) -> int:
        """Menor entre largura e altura, em pixels.

        E a dimensao que decide se a regiao ainda carrega forma reconhecivel:
        um recorte de 200x4 pixels tem area razoavel e nenhuma informacao.
        """
        return min(self.largura, self.altura)

    def intersecao_sobre_uniao(self, outra: "Caixa") -> float:
        """Calcula o IoU entre esta caixa e outra.

        Args:
            outra: Caixa a comparar, nas mesmas coordenadas de imagem.

        Returns:
            Valor entre 0.0 e 1.0. Zero quando nao ha sobreposicao.
        """
        x1 = max(self.x1, outra.x1)
        y1 = max(self.y1, outra.y1)
        x2 = min(self.x2, outra.x2)
        y2 = min(self.y2, outra.y2)

        largura_comum = max(0, x2 - x1)
        altura_comum = max(0, y2 - y1)
        intersecao = largura_comum * altura_comum
        if intersecao == 0:
            return 0.0

        uniao = self.area + outra.area - intersecao
        return intersecao / uniao

    def com_margem(self, fracao: float, largura_max: int, altura_max: int) -> "Caixa":
        """Devolve a caixa expandida proporcionalmente, presa aos limites da imagem.

        A margem existe porque o detector tende a colar a caixa no logo, e um
        simbolo recortado exatamente na borda perde a silhueta que o codificador
        usa para reconhece-lo. Margem demais tem o efeito oposto: o vetor passa a
        descrever a camiseta em vez do logo.

        Args:
            fracao: Proporcao de cada lado a acrescentar. `0.12` acrescenta 12%
                da largura de cada lado horizontal e 12% da altura de cada lado
                vertical. Deve ser nao negativa.
            largura_max: Largura da imagem, usada como teto para x2.
            altura_max: Altura da imagem, usada como teto para y2.

        Returns:
            Nova caixa expandida e presa a `[0, largura_max] x [0, altura_max]`.

        Raises:
            CaixaInvalidaError: Se `fracao` for negativa, ou se os limites da
                imagem forem menores que a origem da caixa.
        """
        if fracao < 0:
            raise CaixaInvalidaError(f"fracao de margem nao pode ser negativa: {fracao}")

        margem_x = int(round(self.largura * fracao))
        margem_y = int(round(self.altura * fracao))

        return Caixa(
            x1=max(0, self.x1 - margem_x),
            y1=max(0, self.y1 - margem_y),
            x2=min(largura_max, self.x2 + margem_x),
            y2=min(altura_max, self.y2 + margem_y),
        )
