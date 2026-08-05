"""Contrato do codificador — a peca que NAO treina e mesmo assim identifica.

O codificador nao sabe o que e uma marca. Ele sabe apenas que duas formas
coincidem. E por isso que funciona sem treino por marca, e e por isso que marca
nova custa minutos em vez de um ciclo de retreino.

Typical usage:
    vetores = codificador.codificar([recorte_a, recorte_b])
    assinatura = codificador.identificacao()
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from application.ports.i_fonte_de_imagens import ImagemRgb


class ICodificador(ABC):
    """Transforma regioes recortadas em vetores comparaveis por cosseno."""

    @abstractmethod
    def codificar(self, recortes: Sequence[ImagemRgb]) -> NDArray[np.float32]:
        """Converte recortes em vetores.

        Args:
            recortes: Regioes ja recortadas e no tamanho esperado pelo modelo.

        Returns:
            Matriz `(len(recortes), dimensao)` em float32, com cada linha
            **L2-normalizada**. A normalizacao e contrato: e o que torna o
            produto interno equivalente ao cosseno e permite que a busca seja
            uma unica multiplicacao de matriz.
        """
        ...

    @abstractmethod
    def identificacao(self) -> str:
        """Retorna a assinatura estavel do codificador.

        A assinatura inclui o modelo **e o modo de agregacao**: o mesmo modelo
        com agregacoes diferentes produz espacos vetoriais distintos, e as
        dimensoes podem coincidir por acaso.

        Returns:
            Texto estavel entre execucoes, gravado no manifesto do banco. E o
            que permite recusar a consulta de um banco construido com outro
            codificador — sem essa checagem o sistema devolve similaridades
            plausiveis e completamente sem significado.
        """
        ...

    @abstractmethod
    def dimensao(self) -> int:
        """Retorna a dimensao dos vetores produzidos.

        Returns:
            Numero de componentes. Valido apenas apos `preparar()`.
        """
        ...

    @abstractmethod
    def preparar(self) -> None:
        """Carrega os pesos do modelo. Idempotente."""
        ...
