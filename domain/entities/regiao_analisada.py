"""Entidade que acumula tudo que se descobriu sobre uma regiao detectada.

E o objeto que atravessa a pipeline: sai do detector com apenas a deteccao,
ganha candidatos na busca vetorial, ganha vereditos na verificacao geometrica, e
chega ao roteador completo. As propriedades derivadas existem para que o roteador
receba os sinais prontos e nao precise reimplementar as mesmas contas.

Typical usage:
    regiao = RegiaoAnalisada(identificador="img-00", deteccao=deteccao)
    regiao = regiao.com_candidatos(candidatos)
    regiao = regiao.com_vereditos(vereditos)
    decisao = roteador.rotear(regiao)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from domain.entities.deteccao import Deteccao
from domain.value_objects.candidato import Candidato
from domain.value_objects.veredito_geometrico import VereditoGeometrico


@dataclass(frozen=True)
class RegiaoAnalisada:
    """Uma deteccao acrescida do que as camadas seguintes descobriram.

    Attributes:
        identificador: Chave estavel da regiao, unica dentro de uma execucao.
        deteccao: O que o detector agnostico produziu.
        candidatos: Respostas do banco, **ordenadas por similaridade
            decrescente**. A ordem e contrato — as propriedades derivadas
            dependem dela.
        vereditos: Resultados da verificacao geometrica. Pode ser vazio: nem
            toda regiao tem pontos suficientes, e ausencia de veredito e
            diferente de veredito negativo.
    """

    identificador: str
    deteccao: Deteccao
    candidatos: tuple[Candidato, ...] = field(default_factory=tuple)
    vereditos: tuple[VereditoGeometrico, ...] = field(default_factory=tuple)

    def com_candidatos(self, candidatos: Sequence[Candidato]) -> "RegiaoAnalisada":
        """Devolve uma copia com os candidatos do banco preenchidos.

        Args:
            candidatos: Respostas do banco, ja ordenadas por similaridade
                decrescente.

        Returns:
            Nova instancia — a entidade e imutavel.
        """
        return replace(self, candidatos=tuple(candidatos))

    def com_vereditos(self, vereditos: Sequence[VereditoGeometrico]) -> "RegiaoAnalisada":
        """Devolve uma copia com os vereditos geometricos preenchidos.

        Args:
            vereditos: Resultados da verificacao geometrica. Pode ser vazio.

        Returns:
            Nova instancia — a entidade e imutavel.
        """
        return replace(self, vereditos=tuple(vereditos))

    @property
    def similaridade_topo(self) -> float:
        """Similaridade do melhor candidato, ou 0.0 se o banco nao respondeu."""
        return self.candidatos[0].similaridade if self.candidatos else 0.0

    @property
    def marca_topo(self) -> str | None:
        """Marca do melhor candidato, ou None se o banco nao respondeu."""
        return self.candidatos[0].marca if self.candidatos else None

    @property
    def similaridade_rival(self) -> float:
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
        topo = self.marca_topo
        for candidato in self.candidatos:
            if candidato.marca != topo:
                return candidato.similaridade
        return 0.0

    @property
    def marca_rival(self) -> str | None:
        """Marca do melhor candidato que nao e a do topo, ou None se nao houver."""
        topo = self.marca_topo
        for candidato in self.candidatos:
            if candidato.marca != topo:
                return candidato.marca
        return None

    @property
    def margem(self) -> float:
        """Quanto a marca escolhida supera a rival mais proxima.

        Returns:
            Diferenca entre a similaridade do topo e a da rival. Mede confianca
            na escolha da MARCA, nao na presenca de logo.
        """
        return self.similaridade_topo - self.similaridade_rival

    @property
    def melhor_veredito(self) -> VereditoGeometrico | None:
        """Veredito geometrico com mais inliers, ou None se nao houve verificacao.

        Returns:
            O veredito mais forte disponivel. None significa que a camada nao
            opinou — o que e diferente de ter opinado contra.
        """
        if not self.vereditos:
            return None
        return max(self.vereditos, key=lambda veredito: veredito.inliers)

    @property
    def veredito_confirmado(self) -> VereditoGeometrico | None:
        """Melhor veredito entre os que confirmam, ou None se nenhum confirma."""
        confirmados = [veredito for veredito in self.vereditos if veredito.confirma]
        if not confirmados:
            return None
        return max(confirmados, key=lambda veredito: veredito.inliers)
