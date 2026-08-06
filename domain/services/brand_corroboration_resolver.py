"""Promove regiao em revisao quando a MESMA imagem ja confirmou aquela marca.

--------------------------------------------------------------------------
O SINAL QUE A PIPELINE IGNORAVA
--------------------------------------------------------------------------
Todo o resto do sistema decide **regiao por regiao**, como se cada caixa
chegasse sozinha. Mas ninguem fotografa um logo isolado: fotografa um backdrop
com o patrocinador repetido oito vezes, um uniforme com a marca no peito e na
manga, uma arena com o painel inteiro da mesma marca.

Numa imagem real medida aqui, `volkswagen` aparecia tres vezes - uma aceita com
110 inliers e duas em revisao - e `ifood` duas vezes, uma aceita com 127 inliers
e uma em revisao. As tres regioes em revisao eram a mesma marca que a imagem ja
tinha confirmado com folga, e ainda assim iam para a fila humana como se ninguem
soubesse de nada.

--------------------------------------------------------------------------
POR QUE ISTO NAO E BAIXAR O LIMIAR
--------------------------------------------------------------------------
Baixar o limiar global compra as mesmas promocoes e paga com **falso aceite em
toda regiao do sistema**, inclusive nas imagens onde nao ha confirmacao nenhuma.
Aqui a promocao exige uma evidencia que o limiar nao tem como exprimir: outra
regiao, na mesma foto, com a mesma marca, aceita por conta propria.

Duas travas mantem isso honesto:

- **A corroboracao vem de um aceite, nunca de outra revisao.** Duas regioes
  duvidosas nao viram uma certeza por se apoiarem; isso so propagaria o erro.
- **A regiao promovida precisa de similaridade propria alta.** Medido, regiao
  errada tem similaridade mediana 0.539 e p90 de 0.755. Exigir 0.85 mantem a
  promocao longe dessa distribuicao - a corroboracao decide entre "e esta marca"
  e "nao sei", e nao entre "e logo" e "e parede".

--------------------------------------------------------------------------
ORDEM
--------------------------------------------------------------------------
Roda **depois** do `NestedRegionResolver`. Corroborar antes usaria como testemunha
uma caixa que vai ser colapsada, e uma caixa que nao chega ao relatorio nao pode
confirmar nada.

Typical usage:
    resolver = BrandCorroborationResolver(min_similarity=0.85)
    promoted = resolver.resolve(decided)
"""

from collections.abc import Sequence

from domain.entities.analyzed_region import AnalyzedRegion
from domain.entities.decision import Decision
from domain.enums.queue import Queue


class BrandCorroborationResolver:
    """Aceita regiao em revisao quando a imagem ja confirmou a marca em outra caixa."""

    def __init__(self, min_similarity: float, enabled: bool = True) -> None:
        """Guarda a politica.

        Args:
            min_similarity: Similaridade propria minima da regiao promovida.
            enabled: Se a regra roda.
        """
        self._min_similarity = min_similarity
        self._enabled = enabled

    def resolve(
        self, decided: Sequence[tuple[AnalyzedRegion, Decision]]
    ) -> tuple[tuple[AnalyzedRegion, Decision], ...]:
        """Promove as regioes em revisao corroboradas por um aceite da mesma imagem.

        Args:
            decided: Pares `(regiao, decisao)` de **uma unica imagem**, ja
                passados pelo roteador e pelo resolvedor de aninhamento.

        Returns:
            Os mesmos pares, na mesma ordem, com as decisoes promovidas
            substituidas.
        """
        if not self._enabled:
            return tuple(decided)

        confirmed = {
            decision.brand
            for _, decision in decided
            if decision.queue is Queue.AUTO_ACCEPT and decision.brand is not None
        }
        if not confirmed:
            return tuple(decided)

        return tuple(
            (region, self._promote(region, decision, confirmed)) for region, decision in decided
        )

    def _promote(
        self, region: AnalyzedRegion, decision: Decision, confirmed: set[str]
    ) -> Decision:
        """Devolve a decisao promovida, ou a original quando a regra nao se aplica.

        Args:
            region: Regiao analisada.
            decision: Decisao atual.
            confirmed: Marcas ja aceitas nesta imagem.

        Returns:
            A decisao final desta regiao.
        """
        if decision.queue is not Queue.REVIEW:
            return decision
        brand = region.top_brand
        if brand is None or brand not in confirmed:
            return decision
        if region.top_similarity < self._min_similarity:
            return decision

        return Decision(
            queue=Queue.AUTO_ACCEPT,
            brand=brand,
            score=decision.score,
            reasons=(
                *decision.reasons,
                f"esta imagem ja confirmou {brand!r} em outra regiao, por conta propria. "
                f"Com similaridade {region.top_similarity:.3f} aqui, mandar para conferencia "
                "humana seria pedir que alguem confirmasse o que a propria foto ja mostrou.",
            ),
            contributions=decision.contributions,
        )
