"""Caso de uso que transforma uma pasta de referencias no banco vetorial.

E este caso de uso que sustenta a afirmacao "marca nova em minutos": nao ha
treino, apenas codificacao de imagens novas e escrita de um arquivo.

O layout esperado e `<raiz>/<marca>/<variante>/arquivo`. A marca e o primeiro
nivel — e o rotulo que o sistema devolve. A variante e o segundo, e serve para
tornar visivel no `ls` qual tipo de aplicacao da marca esta sub-representado.

Typical usage:
    output = use_case.execute(BuildDatabaseCommand(folder, destination))
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from application.dtos.analysis import BuildDatabaseCommand, DatabaseOutput
from application.ports.i_encoder import IEncoder
from application.ports.i_image_source import IImageSource
from config.settings import AppConfig
from domain.exceptions.domain_exceptions import ReferencesNotFoundError
from domain.value_objects.box import Box

_IGNORED_FOLDER_PREFIXES = (".", "_")


class BuildDatabaseUseCase:
    """Codifica as referencias de uma pasta e grava o banco vetorial."""

    def __init__(
        self,
        encoder: IEncoder,
        source: IImageSource,
        writer: "DatabaseWriter",
        config: AppConfig,
    ) -> None:
        """Recebe as portas e a funcao de gravacao ja construidas.

        Args:
            encoder: Camada que transforma referencia em vetor.
            source: Acesso a imagem e recorte.
            writer: Objeto capaz de persistir o banco construido.
            config: Parametros de codificacao e de busca.
        """
        self._encoder = encoder
        self._source = source
        self._writer = writer
        self._config = config

    def execute(self, command: BuildDatabaseCommand) -> DatabaseOutput:
        """Constroi o banco a partir da pasta de referencias.

        Args:
            command: Pasta de origem e destino do banco.

        Returns:
            Resumo do que entrou, o que foi descartado por redundancia, e quais
            pares de marcas diferentes ficaram parecidos demais.

        Raises:
            ReferencesNotFoundError: Se a pasta nao contiver imagem utilizavel.
            FileNotFoundError: Se a pasta nao existir.
        """
        entries = tuple(self._walk(command.references_folder))
        if not entries:
            raise ReferencesNotFoundError(
                f"nenhuma imagem em {command.references_folder}. "
                f"Estrutura esperada: <marca>/<variante>/arquivo.jpg"
            )

        vectors = self._encode(entries)
        kept, discarded = self._remove_redundant(entries, vectors)

        indexes = [index for index, _ in kept]
        self._writer.write(
            destination=command.destination,
            vectors=vectors[indexes],
            brands=tuple(brand for _, (brand, _, _) in kept),
            variants=tuple(variant for _, (_, variant, _) in kept),
            paths=tuple(str(path) for _, (_, _, path) in kept),
            encoder_signature=self._encoder.signature(),
        )

        counts: dict[str, int] = {}
        for _, (brand, _, _) in kept:
            counts[brand] = counts.get(brand, 0) + 1

        return DatabaseOutput(
            total_references=len(kept),
            references_by_brand=dict(sorted(counts.items())),
            discarded_by_redundancy=discarded,
        )

    def _walk(self, root: Path) -> Iterable[tuple[str, str, Path]]:
        """Percorre a pasta de referencias produzindo `(marca, variante, caminho)`.

        Args:
            root: Pasta com o layout `<marca>/<variante>/arquivo`.

        Yields:
            Uma tupla por imagem encontrada, em ordem estavel.

        Raises:
            FileNotFoundError: Se a raiz nao existir.
        """
        if not root.exists():
            raise FileNotFoundError(f"pasta de referencias nao existe: {root}")

        for brand_folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if brand_folder.name.startswith(_IGNORED_FOLDER_PREFIXES):
                continue
            brand = brand_folder.name.strip().lower()
            for path in self._source.list_images(brand_folder):
                relative = path.relative_to(brand_folder)
                if any(part.startswith(_IGNORED_FOLDER_PREFIXES) for part in relative.parts[:-1]):
                    continue
                variant = relative.parts[0] if len(relative.parts) > 1 else "raiz"
                yield brand, variant, path

    def _encode(self, entries: tuple[tuple[str, str, Path], ...]) -> NDArray[np.float32]:
        """Codifica todas as referencias em lote.

        As referencias sao preparadas do mesmo jeito que uma regiao de consulta
        sera preparada. Assimetria aqui — referencia esticada e consulta com
        letterbox — e uma fonte silenciosa de similaridade baixa em par que
        deveria casar.

        Args:
            entries: Tuplas `(marca, variante, caminho)`.

        Returns:
            Matriz `(len(entries), dimensao)` L2-normalizada.
        """
        side = self._config.encoder.crop_side
        crops = []
        for _, _, path in entries:
            image = self._source.load(path)
            width, height = self._source.dimensions(image)
            crops.append(self._source.crop(image, Box(0, 0, width, height), 0.0, side))
        return self._encoder.encode(crops)

    def _remove_redundant(
        self,
        entries: tuple[tuple[str, str, Path], ...],
        vectors: NDArray[np.float32],
    ) -> tuple[list[tuple[int, tuple[str, str, Path]]], int]:
        """Descarta referencias quase identicas a outra da MESMA marca.

        O corte e por marca e **nunca entre marcas**: duas marcas parecidas
        acima do limiar sao exatamente o que a auditoria precisa enxergar, nao
        algo a colapsar. Dentro de uma marca, porem, trinta fotos do mesmo
        angulo sao um vetor — elas ocupam o topo da busca com copias e escondem
        os angulos que faltam.

        Args:
            entries: Tuplas `(marca, variante, caminho)`.
            vectors: Vetores correspondentes, L2-normalizados.

        Returns:
            Tupla `(kept, discarded)`, onde `kept` preserva o indice original de
            cada entrada.
        """
        threshold = self._config.search.redundancy_similarity
        by_brand: dict[str, list[int]] = {}
        for index, (brand, _, _) in enumerate(entries):
            by_brand.setdefault(brand, []).append(index)

        kept: list[tuple[int, tuple[str, str, Path]]] = []
        discarded = 0
        for indexes in by_brand.values():
            accepted: list[int] = []
            for index in indexes:
                similarities = [float(vectors[index] @ vectors[j]) for j in accepted]
                if similarities and max(similarities) > threshold:
                    discarded += 1
                    continue
                accepted.append(index)
                kept.append((index, entries[index]))
        kept.sort(key=lambda pair: pair[0])
        return kept, discarded


class DatabaseWriter(Protocol):
    """Contrato de persistencia do banco, satisfeito estruturalmente.

    E um `Protocol` e nao uma ABC de proposito: a infraestrutura nao precisa
    herdar nada para servir aqui, basta ter o metodo com a assinatura certa.
    Herdar obrigaria `infrastructure` a importar `application` so para declarar
    conformidade, o que inverte a direcao natural desta dependencia.
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
        """Persiste o banco construido.

        Args:
            destination: Pasta onde gravar.
            vectors: Matriz L2-normalizada.
            brands: Marca de cada linha.
            variants: Variante de cada linha.
            paths: Arquivo de origem de cada linha.
            encoder_signature: Assinatura de quem produziu os vetores.
        """
        ...
