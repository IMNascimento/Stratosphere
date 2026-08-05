"""Politica de marcas confundiveis e de marcas fora do portfolio.

Duas politicas de negocio moram aqui, e nenhuma das duas e estatistica:

**Grupos de confusao.** Conjuntos de marcas que compartilham linguagem visual e
contexto — mesmo setor, mesma paleta, mesmo tipo de aplicacao. Quando as duas
melhores respostas do banco caem no mesmo grupo com margem pequena, o custo de
errar e maior que o custo de perguntar, e a decisao vai para desempate
independentemente da pontuacao.

**Marcas negativas.** Concorrentes que entram no banco **sem serem clientes**.
Sem elas o sistema e obrigado a escolher entre as marcas do portfolio e sempre
escolhe alguma. Com elas, "isto e um logo de quem nao interessa" passa a ser uma
resposta possivel — e um acerto, nao uma rejeicao.

Typical usage:
    groups = ConfusionGroups(
        groups=(("gatorade", "powerade"),),
        negatives=("powerade",),
        tie_margin=0.06,
    )
    if groups.same_group("gatorade", "powerade"):
        ...
"""

from collections.abc import Iterable, Sequence


class ConfusionGroups:
    """Consulta a politica de marcas confundiveis e fora do portfolio.

    A instancia e imutavel apos a construcao: todos os dados vem por parametro e
    nada e lido de ambiente ou de arquivo.

    Attributes:
        tie_margin: Diferenca de similaridade abaixo da qual duas marcas do
            mesmo grupo sao consideradas empatadas.
    """

    def __init__(
        self,
        groups: Iterable[Sequence[str]],
        negatives: Iterable[str],
        tie_margin: float,
    ) -> None:
        """Inicializa a politica com os grupos e as marcas negativas.

        Args:
            groups: Conjuntos de marcas confundiveis entre si. Marcas sao
                comparadas em minusculo.
            negatives: Marcas presentes no banco que nao pertencem ao portfolio.
            tie_margin: Diferenca de similaridade abaixo da qual duas marcas do
                mesmo grupo empatam. Deve ser nao negativa.

        Raises:
            ValueError: Se `tie_margin` for negativa.
        """
        if tie_margin < 0:
            raise ValueError(f"tie_margin nao pode ser negativa: {tie_margin}")
        self._groups: tuple[frozenset[str], ...] = tuple(
            frozenset(brand.lower() for brand in group) for group in groups
        )
        self._negatives: frozenset[str] = frozenset(brand.lower() for brand in negatives)
        self.tie_margin = tie_margin

    def same_group(self, brand_a: str | None, brand_b: str | None) -> bool:
        """Verifica se duas marcas pertencem ao mesmo grupo de confusao.

        Args:
            brand_a: Primeira marca. Aceita None por conveniencia do chamador.
            brand_b: Segunda marca. Aceita None por conveniencia do chamador.

        Returns:
            True apenas se as duas forem preenchidas, diferentes entre si, e
            compartilharem ao menos um grupo declarado.
        """
        if not brand_a or not brand_b:
            return False
        first = brand_a.lower()
        second = brand_b.lower()
        if first == second:
            return False
        return any(first in group and second in group for group in self._groups)

    def group_of(self, brand: str | None) -> tuple[str, ...]:
        """Retorna o grupo de confusao ao qual a marca pertence.

        Args:
            brand: Marca a consultar.

        Returns:
            Marcas do grupo em ordem alfabetica, ou tupla vazia se a marca nao
            pertence a grupo nenhum.
        """
        if not brand:
            return ()
        target = brand.lower()
        for group in self._groups:
            if target in group:
                return tuple(sorted(group))
        return ()

    def is_negative(self, brand: str | None) -> bool:
        """Verifica se a marca esta no banco mas fora do portfolio.

        Args:
            brand: Marca a consultar.

        Returns:
            True se a marca e concorrente cadastrada e nao cliente.
        """
        if not brand:
            return False
        return brand.lower() in self._negatives
