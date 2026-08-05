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
    grupos = GruposDeConfusao(
        grupos=(("gatorade", "powerade"),),
        negativas=("powerade",),
    )
    if grupos.mesmo_grupo("gatorade", "powerade"):
        ...
"""

from collections.abc import Iterable, Sequence


class GruposDeConfusao:
    """Consulta a politica de marcas confundiveis e fora do portfolio.

    A instancia e imutavel apos a construcao: todos os dados vem por parametro e
    nada e lido de ambiente ou de arquivo.

    Attributes:
        margem_de_empate: Diferenca de similaridade abaixo da qual duas marcas do
            mesmo grupo sao consideradas empatadas.
    """

    def __init__(
        self,
        grupos: Iterable[Sequence[str]],
        negativas: Iterable[str],
        margem_de_empate: float,
    ) -> None:
        """Inicializa a politica com os grupos e as marcas negativas.

        Args:
            grupos: Conjuntos de marcas confundiveis entre si. Marcas sao
                comparadas em minusculo.
            negativas: Marcas presentes no banco que nao pertencem ao portfolio.
            margem_de_empate: Diferenca de similaridade abaixo da qual duas
                marcas do mesmo grupo empatam. Deve ser nao negativa.

        Raises:
            ValueError: Se `margem_de_empate` for negativa.
        """
        if margem_de_empate < 0:
            raise ValueError(f"margem_de_empate nao pode ser negativa: {margem_de_empate}")
        self._grupos: tuple[frozenset[str], ...] = tuple(
            frozenset(marca.lower() for marca in grupo) for grupo in grupos
        )
        self._negativas: frozenset[str] = frozenset(marca.lower() for marca in negativas)
        self.margem_de_empate = margem_de_empate

    def mesmo_grupo(self, marca_a: str | None, marca_b: str | None) -> bool:
        """Verifica se duas marcas pertencem ao mesmo grupo de confusao.

        Args:
            marca_a: Primeira marca. Aceita None por conveniencia do chamador.
            marca_b: Segunda marca. Aceita None por conveniencia do chamador.

        Returns:
            True apenas se as duas forem preenchidas, diferentes entre si, e
            compartilharem ao menos um grupo declarado.
        """
        if not marca_a or not marca_b:
            return False
        primeira = marca_a.lower()
        segunda = marca_b.lower()
        if primeira == segunda:
            return False
        return any(primeira in grupo and segunda in grupo for grupo in self._grupos)

    def grupo_de(self, marca: str | None) -> tuple[str, ...]:
        """Retorna o grupo de confusao ao qual a marca pertence.

        Args:
            marca: Marca a consultar.

        Returns:
            Marcas do grupo em ordem alfabetica, ou tupla vazia se a marca nao
            pertence a grupo nenhum.
        """
        if not marca:
            return ()
        alvo = marca.lower()
        for grupo in self._grupos:
            if alvo in grupo:
                return tuple(sorted(grupo))
        return ()

    def e_negativa(self, marca: str | None) -> bool:
        """Verifica se a marca esta no banco mas fora do portfolio.

        Args:
            marca: Marca a consultar.

        Returns:
            True se a marca e concorrente cadastrada e nao cliente.
        """
        if not marca:
            return False
        return marca.lower() in self._negativas
