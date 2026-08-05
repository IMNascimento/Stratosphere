"""Politica de logo repetido: o mesmo simbolo entra uma vez so no relatorio.

Um detector de vocabulario aberto devolve varios recortes do MESMO logo — um
justo no simbolo, um folgado com o fundo em volta, um intermediario. A supressao
do detector nao remove esses porque ela usa IoU, e IoU e cego para aninhamento:
recorte dentro de recorte divide a area da uniao e sai com IoU baixo.

O estrago nao e caixa duplicada no desenho, e fila humana inflada. Medido numa
imagem real, o mesmo escudo produziu tres regioes com tres destinos diferentes —
`auto_aceite` no recorte justo, `revisao` no folgado e `orfao` no intermediario.
**Os orfaos eram falsos**: recortes internos de um logo ja aceito com
similaridade 0.98. A fila mais valiosa do sistema e justamente a que mais sofre,
porque similaridade media com geometria confirmada e exatamente o que um recorte
mal enquadrado produz.

--------------------------------------------------------------------------
POR QUE VENCE A MELHOR PONTUACAO, E NAO A MAIOR CAIXA
--------------------------------------------------------------------------
A intuicao diz "fica com a caixa que engloba as outras". Os dados dizem o
contrario: no caso medido, a caixa maior tinha pontuacao 0.51 — ela engolia o
texto embaixo do escudo e parte do fundo — e a menor, justa no simbolo, tinha
0.87 com similaridade 0.96. Manter a maior entregaria uma regiao que descreve
mais cena que marca.

Fica a regiao inteira de melhor pontuacao: a caixa dela e a decisao dela.

--------------------------------------------------------------------------
DUAS RESTRICOES QUE MANTEM A REGRA CONSERVADORA
--------------------------------------------------------------------------
- **So agrupa marca igual.** Caixa pequena de outra marca dentro de uma maior e
  backdrop com varios patrocinadores, nao duplicata. As duas sao legitimas.
- **So agrupa quem afirma marca.** Regiao sem marca nao tem o que deduplicar.

Typical usage:
    resolver = NestedRegionResolver(containment=0.80)
    sobreviventes = resolver.resolve(decided)
"""

from collections.abc import Sequence

from domain.entities.analyzed_region import AnalyzedRegion
from domain.entities.decision import Decision

DecidedRegion = tuple[AnalyzedRegion, Decision]


class NestedRegionResolver:
    """Remove recortes aninhados do mesmo logo, mantendo o de melhor evidencia.

    Servico de dominio puro: sem I/O, sem estado entre chamadas. Duas chamadas
    com a mesma entrada devolvem a mesma saida.

    Attributes:
        containment: Fracao da menor caixa coberta pela maior a partir da qual
            as duas descrevem o mesmo logo.
    """

    def __init__(self, containment: float) -> None:
        """Inicializa a politica com o limiar de contencao.

        Args:
            containment: Entre 0 e 1. Exigir 1.0 seria rigido demais — nem todo
                aninhamento e perfeito, e o recorte folgado costuma ultrapassar
                a borda do justo em alguns pixels.

        Raises:
            ValueError: Se o limiar estiver fora de (0, 1].
        """
        if not 0.0 < containment <= 1.0:
            raise ValueError(f"containment deve estar em (0, 1]: {containment}")
        self.containment = containment

    def resolve(self, decided: Sequence[DecidedRegion]) -> tuple[DecidedRegion, ...]:
        """Devolve apenas uma regiao por logo detectado.

        Args:
            decided: Pares `(regiao, decisao)` na ordem em que a pipeline
                produziu.

        Returns:
            Os sobreviventes, **na ordem original** — a ordem do detector e
            estavel e reproduzivel, e trocar por ordem de pontuacao tornaria o
            relatorio mais dificil de comparar entre execucoes.
        """
        by_score = sorted(
            range(len(decided)), key=lambda index: decided[index][1].score, reverse=True
        )

        kept: list[int] = []
        for index in by_score:
            if not any(self._same_logo(decided[index], decided[other]) for other in kept):
                kept.append(index)
        return tuple(decided[index] for index in sorted(kept))

    def _same_logo(self, candidate: DecidedRegion, keeper: DecidedRegion) -> bool:
        """Decide se dois pares descrevem o mesmo logo.

        Args:
            candidate: Par sob avaliacao.
            keeper: Par ja mantido, de pontuacao maior ou igual.

        Returns:
            True quando as duas afirmam a mesma marca e uma caixa esta contida
            na outra acima do limiar.
        """
        candidate_region, candidate_decision = candidate
        keeper_region, keeper_decision = keeper

        if candidate_decision.brand is None or keeper_decision.brand is None:
            return False
        if candidate_decision.brand != keeper_decision.brand:
            return False
        return (
            candidate_region.detection.box.containment_with(keeper_region.detection.box)
            >= self.containment
        )
