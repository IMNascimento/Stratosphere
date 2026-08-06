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

Os nomes dos arquivos e as chaves do json seguem em portugues de proposito: eles
sao o formato ja gravado em disco, e traduzi-los invalidaria todo indice
existente sem ganho nenhum de legibilidade de codigo.

Typical usage:
    database = NpzReferenceDatabase.load(Path("indice"), current_signature)
    candidates = database.search(vector, count=25)
"""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from domain.exceptions.domain_exceptions import (
    EmptyDatabaseError,
    IncompatibleDimensionError,
    IncompatibleEncoderError,
)
from domain.repositories.i_reference_database import IReferenceDatabase
from domain.value_objects.candidate import Candidate

VECTORS_FILE = "referencias.npz"
METADATA_FILE = "referencias.json"


class NpzReferenceDatabase(IReferenceDatabase):
    """Banco vetorial em memoria, persistido em disco como npz mais json."""

    def __init__(
        self,
        vectors: NDArray[np.float32],
        brands: tuple[str, ...],
        variants: tuple[str, ...],
        paths: tuple[str, ...],
        encoder_signature: str,
    ) -> None:
        """Monta o banco a partir dos vetores e seus metadados.

        Args:
            vectors: Matriz `(N, D)` L2-normalizada.
            brands: Marca de cada linha.
            variants: Variante de cada linha.
            paths: Arquivo de origem de cada linha.
            encoder_signature: Quem produziu os vetores.

        Raises:
            EmptyDatabaseError: Se nao houver vetores, ou se os metadados nao
                tiverem o mesmo comprimento da matriz.
        """
        if vectors.size == 0:
            raise EmptyDatabaseError("o banco de referencia esta vazio")
        if not (len(brands) == len(variants) == len(paths) == vectors.shape[0]):
            raise EmptyDatabaseError(
                f"metadados inconsistentes: {vectors.shape[0]} vetores, "
                f"{len(brands)} marcas, {len(variants)} variantes, {len(paths)} caminhos"
            )
        self._vectors = vectors.astype(np.float32, copy=False)
        self._brands = brands
        self._variants = variants
        self._paths = paths
        self._signature = encoder_signature

    # -- consulta ----------------------------------------------------------

    def search(self, vector: NDArray[np.float32], count: int) -> tuple[Candidate, ...]:
        """Retorna as referencias mais parecidas com o vetor consultado.

        Args:
            vector: Vetor da regiao, ja L2-normalizado.
            count: Quantos vizinhos retornar.

        Returns:
            Candidatos em ordem decrescente de similaridade.

        Raises:
            IncompatibleDimensionError: Se a dimensao nao bater com a do banco.
        """
        flat = np.asarray(vector, dtype=np.float32).ravel()
        if flat.shape[0] != self.dimension():
            raise IncompatibleDimensionError(flat.shape[0], self.dimension())

        similarities = self._vectors @ flat
        count = max(1, min(count, self.size()))

        # argpartition evita ordenar N valores quando so interessam os primeiros.
        partial = np.argpartition(-similarities, kth=count - 1)[:count]
        ordered = partial[np.argsort(-similarities[partial])]

        return tuple(
            Candidate(
                brand=self._brands[index],
                similarity=float(np.clip(similarities[index], -1.0, 1.0)),
                reference=self._paths[index],
            )
            for index in ordered
        )

    def brands(self) -> tuple[str, ...]:
        """Retorna as marcas presentes no banco, em ordem alfabetica."""
        return tuple(sorted(set(self._brands)))

    def size(self) -> int:
        """Retorna quantas referencias o banco contem."""
        return int(self._vectors.shape[0])

    def dimension(self) -> int:
        """Retorna a dimensao dos vetores armazenados."""
        return int(self._vectors.shape[1])

    def encoder_signature(self) -> str:
        """Retorna a assinatura do codificador que produziu os vetores."""
        return self._signature

    def references_by_brand(self) -> dict[str, int]:
        """Retorna quantas referencias cada marca tem, em ordem alfabetica."""
        counts: dict[str, int] = {}
        for brand in self._brands:
            counts[brand] = counts.get(brand, 0) + 1
        return dict(sorted(counts.items()))

    def confusable_pairs(self, threshold: float) -> tuple[tuple[str, str, float], ...]:
        """Roda o banco contra ele mesmo e lista marcas parecidas demais.

        Args:
            threshold: Similaridade acima da qual o par e reportado.

        Returns:
            Tuplas `(brand_a, brand_b, similaridade)` de marcas DIFERENTES, em
            ordem decrescente. Pares da mesma marca sao ignorados: referencias
            parecidas dentro de uma marca sao redundancia, nao risco.
        """
        if self.size() < 2:
            return ()

        similarities = self._vectors @ self._vectors.T
        np.fill_diagonal(similarities, -1.0)
        brands = np.array(self._brands)

        rows, columns = np.nonzero(np.triu(similarities > threshold, k=1))
        pairs = [
            (str(brands[row]), str(brands[column]), float(similarities[row, column]))
            for row, column in zip(rows, columns, strict=True)
            if brands[row] != brands[column]
        ]
        pairs.sort(key=lambda pair: pair[2], reverse=True)
        return tuple(pairs)

    # -- persistencia ------------------------------------------------------

    @classmethod
    def load(cls, source: Path, expected_signature: str | None = None) -> "NpzReferenceDatabase":
        """Carrega um banco gravado e valida a compatibilidade do codificador.

        Args:
            source: Pasta com os arquivos do banco.
            expected_signature: Assinatura do codificador em uso. Quando
                informada, um banco produzido por outro codificador e recusado.

        Returns:
            O banco pronto para consulta.

        Raises:
            FileNotFoundError: Se os arquivos do banco nao existirem.
            IncompatibleEncoderError: Se a assinatura nao bater.
            EmptyDatabaseError: Se os metadados forem inconsistentes.
        """
        vectors_file = source / VECTORS_FILE
        metadata_file = source / METADATA_FILE
        if not vectors_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"banco ausente em {source}. Construa primeiro com "
                f"`stratosphere banco --referencias <pasta>`"
            )

        with np.load(vectors_file, allow_pickle=False) as blob:
            vectors = blob["vetores"].astype(np.float32)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        signature = str(metadata["manifesto"]["assinatura_do_codificador"])
        if expected_signature is not None and signature != expected_signature:
            raise IncompatibleEncoderError(signature, expected_signature)

        entries = metadata["entradas"]
        return cls(
            vectors=vectors,
            brands=tuple(entry["marca"] for entry in entries),
            variants=tuple(entry["variante"] for entry in entries),
            paths=tuple(entry["caminho"] for entry in entries),
            encoder_signature=signature,
        )


class NpzDatabaseWriter:
    """Escreve um banco em disco.

    Separado de `NpzReferenceDatabase` de proposito. O banco tem a invariante de
    nao estar vazio — faz sentido para quem consulta e nao faz nenhum para quem
    grava. Juntar os dois obrigaria a construir um banco falso so para poder
    escrever o verdadeiro, na primeira execucao de uma instalacao limpa.
    """

    def write(
        self,
        destination: Path,
        vectors: NDArray[np.float32],
        brands: tuple[str, ...],
        variants: tuple[str, ...],
        paths: tuple[str, ...],
        encoder_signature: str,
    ) -> None:
        """Persiste um banco construido em `destination`.

        Args:
            destination: Pasta onde gravar. Criada se nao existir.
            vectors: Matriz L2-normalizada.
            brands: Marca de cada linha.
            variants: Variante de cada linha.
            paths: Arquivo de origem de cada linha.
            encoder_signature: Quem produziu os vetores. Sem isso, um banco
                consultado por outro codificador devolveria similaridades
                plausiveis e sem significado.
        """
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / VECTORS_FILE, vetores=vectors)

        counts: dict[str, int] = {}
        for brand in brands:
            counts[brand] = counts.get(brand, 0) + 1

        metadata = {
            "manifesto": {
                "assinatura_do_codificador": encoder_signature,
                "dimensao": int(vectors.shape[1]),
                "total_de_referencias": int(vectors.shape[0]),
            },
            "referencias_por_marca": dict(sorted(counts.items())),
            "entradas": [
                {"marca": brand, "variante": variant, "caminho": path}
                for brand, variant, path in zip(brands, variants, paths, strict=True)
            ],
        }
        (destination / METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
