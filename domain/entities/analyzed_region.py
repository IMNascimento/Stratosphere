"""Entidade que acumula tudo que se descobriu sobre uma regiao detectada.

E o objeto que atravessa a pipeline: sai do detector com apenas a deteccao,
ganha candidatos na busca vetorial, ganha vereditos na verificacao geometrica, e
chega ao roteador completo. As propriedades derivadas existem para que o roteador
receba os sinais prontos e nao precise reimplementar as mesmas contas.

Typical usage:
    region = AnalyzedRegion(identifier="img-00", detection=detection)
    region = region.with_candidates(candidates)
    region = region.with_verdicts(verdicts)
    decision = router.route(region)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from domain.entities.detection import Detection
from domain.value_objects.candidate import Candidate
from domain.value_objects.geometric_verdict import GeometricVerdict


@dataclass(frozen=True)
class AnalyzedRegion:
    """Uma deteccao acrescida do que as camadas seguintes descobriram.

    Attributes:
        identifier: Chave estavel da regiao, unica dentro de uma execucao.
        detection: O que o detector agnostico produziu.
        candidates: Respostas do banco, **ordenadas por similaridade
            decrescente**. A ordem e contrato — as propriedades derivadas
            dependem dela.
        verdicts: Resultados da verificacao geometrica. Pode ser vazio: nem
            toda regiao tem pontos suficientes, e ausencia de veredito e
            diferente de veredito negativo.
    """

    identifier: str
    detection: Detection
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    verdicts: tuple[GeometricVerdict, ...] = field(default_factory=tuple)

    def with_candidates(self, candidates: Sequence[Candidate]) -> "AnalyzedRegion":
        """Devolve uma copia com os candidatos do banco preenchidos.

        Args:
            candidates: Respostas do banco, ja ordenadas por similaridade
                decrescente.

        Returns:
            Nova instancia — a entidade e imutavel.
        """
        return replace(self, candidates=tuple(candidates))

    def with_verdicts(self, verdicts: Sequence[GeometricVerdict]) -> "AnalyzedRegion":
        """Devolve uma copia com os vereditos geometricos preenchidos.

        Args:
            verdicts: Resultados da verificacao geometrica. Pode ser vazio.

        Returns:
            Nova instancia — a entidade e imutavel.
        """
        return replace(self, verdicts=tuple(verdicts))

    @property
    def top_similarity(self) -> float:
        """Similaridade do melhor candidato, ou 0.0 se o banco nao respondeu."""
        return self.candidates[0].similarity if self.candidates else 0.0

    @property
    def top_brand(self) -> str | None:
        """Marca do melhor candidato, ou None se o banco nao respondeu."""
        return self.candidates[0].brand if self.candidates else None

    @property
    def rival_similarity(self) -> float:
        """Similaridade do melhor candidato de OUTRA marca.

        **Nao e o segundo vizinho.** Com dezenas de referencias por marca, o
        segundo vizinho e quase sempre outra foto da mesma marca; usar a
        similaridade dele faria a margem colapsar para perto de zero em todos os
        casos, o termo de margem morreria, e o peso dele viraria teto morto no
        score — o maximo alcancavel cairia abaixo do limiar de aceite. **O
        defeito piora conforme o banco cresce**, ou seja, exatamente na direcao
        em que o sistema deve evoluir.

        Returns:
            Similaridade da rival mais proxima, ou 0.0 quando o top-k inteiro e
            de uma marca so. Nesse caso nao ha rival a vista e a margem e
            maxima — mas isso depende de o top-k ser grande o bastante para uma
            segunda marca aparecer.
        """
        top = self.top_brand
        for candidate in self.candidates:
            if candidate.brand != top:
                return candidate.similarity
        return 0.0

    @property
    def rival_brand(self) -> str | None:
        """Marca do melhor candidato que nao e a do topo, ou None se nao houver."""
        top = self.top_brand
        for candidate in self.candidates:
            if candidate.brand != top:
                return candidate.brand
        return None

    @property
    def margin(self) -> float:
        """Quanto a marca escolhida supera a rival mais proxima.

        Returns:
            Diferenca entre a similaridade do topo e a da rival. Mede confianca
            na escolha da MARCA, nao na presenca de logo.
        """
        return self.top_similarity - self.rival_similarity

    @property
    def best_verdict(self) -> GeometricVerdict | None:
        """Veredito geometrico com mais inliers, ou None se nao houve verificacao.

        Returns:
            O veredito mais forte disponivel. None significa que a camada nao
            opinou — o que e diferente de ter opinado contra.
        """
        if not self.verdicts:
            return None
        return max(self.verdicts, key=lambda verdict: verdict.inliers)

    @property
    def confirmed_verdict(self) -> GeometricVerdict | None:
        """Melhor veredito entre os que confirmam, ou None se nenhum confirma."""
        confirmed = [verdict for verdict in self.verdicts if verdict.confirms]
        if not confirmed:
            return None
        return max(confirmed, key=lambda verdict: verdict.inliers)
