"""Banco de referencia em `npz` mais `json`, com busca por produto interno.

**Por que nao um banco vetorial.** Dezenas de marcas com algumas dezenas de
referencias cada dao milhares de vetores. Uma consulta e uma multiplicacao de
matriz `(1, D) x (D, N)` — milissegundos. Banco vetorial passa a valer a partir
de centenas de milhares de vetores; antes disso ele acrescenta um servico para
operar, um processo para subir e um estado a mais para dessincronizar. O `json`
de metadados, em compensacao, e auditavel com um editor de texto — o que importa
muito quando a pergunta e "por que esta referencia esta puxando falso positivo?".

**A trava que evita uma semana perdida.** O manifesto guarda a assinatura do
codificador que produziu os vetores. Um banco construido com um codificador e
consultado com outro devolve similaridades plausiveis e completamente sem
significado — os dois espacos vetoriais nao tem relacao nenhuma. Aqui isso vira
excecao em vez de virar relatorio errado.

Typical usage:
    banco = BancoReferenciaNpz.carregar(Path("indice"), assinatura_atual)
    candidatos = banco.buscar(vetor, quantidade=25)
"""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from domain.exceptions.domain_exceptions import (
    BancoVazioError,
    CodificadorIncompativelError,
    DimensaoIncompativelError,
)
from domain.repositories.i_banco_referencia import IBancoReferencia
from domain.value_objects.candidato import Candidato

ARQUIVO_DE_VETORES = "referencias.npz"
ARQUIVO_DE_METADADOS = "referencias.json"


class BancoReferenciaNpz(IBancoReferencia):
    """Banco vetorial em memoria, persistido em disco como npz mais json."""

    def __init__(
        self,
        vetores: NDArray[np.float32],
        marcas: tuple[str, ...],
        variantes: tuple[str, ...],
        caminhos: tuple[str, ...],
        assinatura_do_codificador: str,
    ) -> None:
        """Monta o banco a partir dos vetores e seus metadados.

        Args:
            vetores: Matriz `(N, D)` L2-normalizada.
            marcas: Marca de cada linha.
            variantes: Variante de cada linha.
            caminhos: Arquivo de origem de cada linha.
            assinatura_do_codificador: Quem produziu os vetores.

        Raises:
            BancoVazioError: Se nao houver vetores, ou se os metadados nao
                tiverem o mesmo comprimento da matriz.
        """
        if vetores.size == 0:
            raise BancoVazioError("o banco de referencia esta vazio")
        if not (len(marcas) == len(variantes) == len(caminhos) == vetores.shape[0]):
            raise BancoVazioError(
                f"metadados inconsistentes: {vetores.shape[0]} vetores, "
                f"{len(marcas)} marcas, {len(variantes)} variantes, {len(caminhos)} caminhos"
            )
        self._vetores = vetores.astype(np.float32, copy=False)
        self._marcas = marcas
        self._variantes = variantes
        self._caminhos = caminhos
        self._assinatura = assinatura_do_codificador

    # -- consulta ----------------------------------------------------------

    def buscar(self, vetor: NDArray[np.float32], quantidade: int) -> tuple[Candidato, ...]:
        """Retorna as referencias mais parecidas com o vetor consultado.

        Args:
            vetor: Vetor da regiao, ja L2-normalizado.
            quantidade: Quantos vizinhos retornar.

        Returns:
            Candidatos em ordem decrescente de similaridade.

        Raises:
            DimensaoIncompativelError: Se a dimensao nao bater com a do banco.
        """
        achatado = np.asarray(vetor, dtype=np.float32).ravel()
        if achatado.shape[0] != self.dimensao():
            raise DimensaoIncompativelError(achatado.shape[0], self.dimensao())

        similaridades = self._vetores @ achatado
        quantidade = max(1, min(quantidade, self.tamanho()))

        # argpartition evita ordenar N valores quando so interessam os primeiros.
        parciais = np.argpartition(-similaridades, kth=quantidade - 1)[:quantidade]
        ordenados = parciais[np.argsort(-similaridades[parciais])]

        return tuple(
            Candidato(
                marca=self._marcas[indice],
                similaridade=float(np.clip(similaridades[indice], -1.0, 1.0)),
                referencia=self._caminhos[indice],
            )
            for indice in ordenados
        )

    def marcas(self) -> tuple[str, ...]:
        """Retorna as marcas presentes no banco, em ordem alfabetica."""
        return tuple(sorted(set(self._marcas)))

    def tamanho(self) -> int:
        """Retorna quantas referencias o banco contem."""
        return int(self._vetores.shape[0])

    def dimensao(self) -> int:
        """Retorna a dimensao dos vetores armazenados."""
        return int(self._vetores.shape[1])

    def assinatura_do_codificador(self) -> str:
        """Retorna a assinatura do codificador que produziu os vetores."""
        return self._assinatura

    def referencias_por_marca(self) -> dict[str, int]:
        """Retorna quantas referencias cada marca tem, em ordem alfabetica."""
        contagem: dict[str, int] = {}
        for marca in self._marcas:
            contagem[marca] = contagem.get(marca, 0) + 1
        return dict(sorted(contagem.items()))

    def pares_confundiveis(self, limiar: float) -> tuple[tuple[str, str, float], ...]:
        """Roda o banco contra ele mesmo e lista marcas parecidas demais.

        Args:
            limiar: Similaridade acima da qual o par e reportado.

        Returns:
            Tuplas `(marca_a, marca_b, similaridade)` de marcas DIFERENTES, em
            ordem decrescente. Pares da mesma marca sao ignorados: referencias
            parecidas dentro de uma marca sao redundancia, nao risco.
        """
        if self.tamanho() < 2:
            return ()

        similaridades = self._vetores @ self._vetores.T
        np.fill_diagonal(similaridades, -1.0)
        marcas = np.array(self._marcas)

        linhas, colunas = np.nonzero(np.triu(similaridades > limiar, k=1))
        pares = [
            (str(marcas[linha]), str(marcas[coluna]), float(similaridades[linha, coluna]))
            for linha, coluna in zip(linhas, colunas, strict=True)
            if marcas[linha] != marcas[coluna]
        ]
        pares.sort(key=lambda par: par[2], reverse=True)
        return tuple(pares)

    # -- persistencia ------------------------------------------------------

    @classmethod
    def carregar(cls, origem: Path, assinatura_esperada: str | None = None) -> "BancoReferenciaNpz":
        """Carrega um banco gravado e valida a compatibilidade do codificador.

        Args:
            origem: Pasta com os arquivos do banco.
            assinatura_esperada: Assinatura do codificador em uso. Quando
                informada, um banco produzido por outro codificador e recusado.

        Returns:
            O banco pronto para consulta.

        Raises:
            FileNotFoundError: Se os arquivos do banco nao existirem.
            CodificadorIncompativelError: Se a assinatura nao bater.
            BancoVazioError: Se os metadados forem inconsistentes.
        """
        arquivo_de_vetores = origem / ARQUIVO_DE_VETORES
        arquivo_de_metadados = origem / ARQUIVO_DE_METADADOS
        if not arquivo_de_vetores.exists() or not arquivo_de_metadados.exists():
            raise FileNotFoundError(
                f"banco ausente em {origem}. Construa primeiro com "
                f"`stratosphere banco --referencias <pasta>`"
            )

        with np.load(arquivo_de_vetores, allow_pickle=False) as blob:
            vetores = blob["vetores"].astype(np.float32)
        metadados = json.loads(arquivo_de_metadados.read_text(encoding="utf-8"))

        assinatura = str(metadados["manifesto"]["assinatura_do_codificador"])
        if assinatura_esperada is not None and assinatura != assinatura_esperada:
            raise CodificadorIncompativelError(assinatura, assinatura_esperada)

        entradas = metadados["entradas"]
        return cls(
            vetores=vetores,
            marcas=tuple(entrada["marca"] for entrada in entradas),
            variantes=tuple(entrada["variante"] for entrada in entradas),
            caminhos=tuple(entrada["caminho"] for entrada in entradas),
            assinatura_do_codificador=assinatura,
        )


class GravadorBancoNpz:
    """Escreve um banco em disco.

    Separado de `BancoReferenciaNpz` de proposito. O banco tem a invariante de
    nao estar vazio — faz sentido para quem consulta e nao faz nenhum para quem
    grava. Juntar os dois obrigaria a construir um banco falso so para poder
    escrever o verdadeiro, na primeira execucao de uma instalacao limpa.
    """

    def gravar(
        self,
        destino: Path,
        vetores: NDArray[np.float32],
        marcas: tuple[str, ...],
        variantes: tuple[str, ...],
        caminhos: tuple[str, ...],
        assinatura_do_codificador: str,
    ) -> None:
        """Persiste um banco construido em `destino`.

        Args:
            destino: Pasta onde gravar. Criada se nao existir.
            vetores: Matriz L2-normalizada.
            marcas: Marca de cada linha.
            variantes: Variante de cada linha.
            caminhos: Arquivo de origem de cada linha.
            assinatura_do_codificador: Quem produziu os vetores. Sem isso, um
                banco consultado por outro codificador devolveria similaridades
                plausiveis e sem significado.
        """
        destino.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destino / ARQUIVO_DE_VETORES, vetores=vetores)

        contagem: dict[str, int] = {}
        for marca in marcas:
            contagem[marca] = contagem.get(marca, 0) + 1

        metadados = {
            "manifesto": {
                "assinatura_do_codificador": assinatura_do_codificador,
                "dimensao": int(vetores.shape[1]),
                "total_de_referencias": int(vetores.shape[0]),
            },
            "referencias_por_marca": dict(sorted(contagem.items())),
            "entradas": [
                {"marca": marca, "variante": variante, "caminho": caminho}
                for marca, variante, caminho in zip(marcas, variantes, caminhos, strict=True)
            ],
        }
        (destino / ARQUIVO_DE_METADADOS).write_text(
            json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8"
        )
