"""Contrato do banco de referencia — a peca que responde QUAL marca e.

O banco e um repositorio de dominio, nao um port de aplicacao: "quais marcas se
parecem com este vetor" e uma pergunta do dominio. A tecnologia que responde
(matriz em memoria, banco vetorial, servico externo) e detalhe de infraestrutura.

E aqui que mora a propriedade central da arquitetura: **marca nova e um append
neste banco, nao um ciclo de retreino**.

Typical usage:
    candidatos = banco.buscar(vetor, quantidade=25)
    if candidatos and candidatos[0].similaridade > limiar:
        ...
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from domain.value_objects.candidato import Candidato


class IBancoReferencia(ABC):
    """Banco vetorial de referencias de marca, agnostico de tecnologia."""

    @abstractmethod
    def buscar(self, vetor: NDArray[np.float32], quantidade: int) -> tuple[Candidato, ...]:
        """Retorna as marcas mais parecidas com o vetor consultado.

        Args:
            vetor: Vetor da regiao, **ja L2-normalizado**, com a mesma dimensao
                dos vetores do banco. A normalizacao previa e o que torna o
                produto interno equivalente ao cosseno.
            quantidade: Quantos vizinhos retornar. Precisa ser grande o bastante
                para que uma **segunda marca** apareca: com dezenas de
                referencias por marca, um top-5 pode ser inteiramente da mesma
                marca, e a margem calculada a partir dele perde o sentido.

        Returns:
            Candidatos ordenados por similaridade decrescente. Tupla vazia se o
            banco estiver vazio.

        Raises:
            DimensaoIncompativelError: Se a dimensao do vetor nao bater com a do
                banco. Nao degrada silenciosamente — uma consulta com dimensao
                errada produz numeros plausiveis e sem significado.
        """
        ...

    @abstractmethod
    def marcas(self) -> tuple[str, ...]:
        """Retorna as marcas presentes no banco, em ordem alfabetica."""
        ...

    @abstractmethod
    def tamanho(self) -> int:
        """Retorna quantas referencias o banco contem."""
        ...

    @abstractmethod
    def dimensao(self) -> int:
        """Retorna a dimensao dos vetores armazenados."""
        ...

    @abstractmethod
    def referencias_por_marca(self) -> dict[str, int]:
        """Retorna quantas referencias cada marca tem.

        Returns:
            Mapa marca -> contagem. Marca com poucas referencias tende a gerar
            orfaos em vez de acertos, e saber disso de antemao muda a leitura de
            qualquer metrica por marca.
        """
        ...

    @abstractmethod
    def pares_confundiveis(self, limiar: float) -> tuple[tuple[str, str, float], ...]:
        """Lista pares de marcas DIFERENTES cujas referencias se parecem demais.

        Roda o banco contra ele mesmo. Cada par retornado e um falso positivo
        esperando acontecer, ou um grupo de confusao ainda nao declarado.

        Args:
            limiar: Similaridade acima da qual o par e reportado.

        Returns:
            Tuplas `(marca_a, marca_b, similaridade)` em ordem decrescente de
            similaridade. Pares da mesma marca nunca aparecem — referencias
            parecidas dentro de uma marca sao redundancia, nao risco.
        """
        ...
