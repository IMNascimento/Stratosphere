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
        brand_references: Quantas referencias a marca do topo tem no banco.
            Serve de teto para o consenso — sem isso, marca com 5 referencias
            jamais alcancaria o mesmo consenso de uma com 40, e a cobertura do
            banco viraria criterio de decisao sem ninguem ter escolhido isso.
    """

    identifier: str
    detection: Detection
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)
    verdicts: tuple[GeometricVerdict, ...] = field(default_factory=tuple)
    brand_references: int = 0

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

    def with_brand_references(self, quantity: int) -> "AnalyzedRegion":
        """Devolve uma copia sabendo quantas referencias a marca do topo tem.

        Args:
            quantity: Contagem vinda do banco. Zero significa desconhecido, e
                o consenso cai para a fracao simples sobre o top-k.

        Returns:
            Nova instancia — a entidade e imutavel.
        """
        return replace(self, brand_references=max(0, quantity))

    @property
    def agreeing_candidates(self) -> int:
        """Quantos candidatos do top-k afirmam a marca escolhida.

        E o numero ABSOLUTO, e nao a fracao. Existe porque o consenso e
        normalizado pelo teto da marca: uma marca com 2 referencias no banco
        alcanca consenso 1.0 com dois candidatos, e isso nao vale o mesmo que
        25 de 25. Quem decide aceite sozinho precisa olhar os dois numeros.

        Returns:
            Contagem de candidatos com a marca do topo. Zero se o banco nao
            respondeu.
        """
        if not self.candidates:
            return 0
        top = self.top_brand
        return sum(1 for candidate in self.candidates if candidate.brand == top)

    @property
    def brand_consensus(self) -> float:
        """Quanto do top-k concorda com a marca escolhida.

        **E o sinal que separa casamento real de vizinho mais proximo por
        acaso**, e faz isso melhor que a similaridade absoluta. Medido em imagem
        real: um swoosh de verdade teve similaridade 0.712 com 21 de 25
        vizinhos da mesma marca; um texto sem marca nenhuma teve similaridade
        **maior**, 0.837, com apenas 3 de 25 — e os demais espalhados entre
        quatro marcas sem relacao. Pela similaridade sozinha, o ruido ganha do
        logo; pelo consenso, nao.

        O teto e `min(top-k, referencias da marca)`: uma marca com 5 referencias
        no banco nunca poderia ocupar 25 posicoes, e comparar contra 25 puniria
        marca pouco coberta por um limite que e do banco, nao da evidencia.

        Returns:
            Entre 0.0 e 1.0. Zero quando o banco nao respondeu.
        """
        if not self.candidates:
            return 0.0
        top = self.top_brand
        agreeing = sum(1 for candidate in self.candidates if candidate.brand == top)
        ceiling = len(self.candidates)
        if self.brand_references:
            ceiling = min(ceiling, self.brand_references)
        if ceiling <= 0:
            return 0.0
        return min(1.0, agreeing / ceiling)

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
