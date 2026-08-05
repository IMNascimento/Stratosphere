"""Verificacao geometrica — confirma que e literalmente o mesmo desenho.

Pergunta diferente da que a busca vetorial responde. A busca diz "parece com";
esta camada diz "e o mesmo desenho, sob uma transformacao coerente".

**Por que as duas convivem: elas falham de formas diferentes.** A busca vetorial
e fraca justamente em mudanca de ponto de vista; o casamento de pontos sob
homografia foi projetado para isso. Um simbolo a 60 graus numa manga curva pode
ter similaridade baixa e ainda assim casar dezenas de pontos sob uma unica
transformacao.

E dai vem o portao permissivo: candidatos entram na verificacao a partir de uma
similaridade **baixa**. Um portao alto so deixa passar o que ja estava decidido,
e a verificacao vira enfeite.

**O limite desta camada.** Ela nao opina em logo chapado, pequeno ou vetorial
demais — nao ha canto para extrair ponto. Isso e resultado valido e diferente de
veredito negativo; quem consome precisa tratar os dois casos separadamente, e o
roteador faz isso renormalizando os pesos.

Nada aqui tem peso treinado. E algoritmo puro — motivo pelo qual esta camada
nunca precisa ser retreinada quando entra marca nova.

Typical usage:
    verifier = SiftVerifier(config)
    verdicts = verifier.verify(crop, candidates)
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from application.ports.i_geometric_verifier import IGeometricVerifier
from application.ports.i_image_source import RgbImage
from config.settings import GeometryConfig
from domain.value_objects.candidate import Candidate
from domain.value_objects.geometric_verdict import GeometricVerdict

# Abaixo disto nem vale tentar casar: o ajuste robusto encontra concordancia em
# qualquer conjunto pequeno de pontos.
_MIN_POINTS = 6

# Minimo matematico para estimar uma homografia.
_POINTS_FOR_HOMOGRAPHY = 4


class SiftVerifier(IGeometricVerifier):
    """Confirma correspondencia de desenho por casamento de pontos e ajuste robusto."""

    def __init__(self, config: GeometryConfig) -> None:
        """Guarda a configuracao sem construir o extrator.

        Args:
            config: Parametros da verificacao geometrica.
        """
        self._config = config
        self._extractor: Any = None
        self._matcher: Any = None
        # Cache por instancia, e nao `lru_cache` no metodo: o decorador guarda
        # `self` na chave e mantem o verificador e todos os descritores vivos
        # pelo tempo do processo. Com o dicionario aqui, o cache morre junto com
        # a instancia.
        self._reference_descriptors: dict[str, tuple[Any, Any]] = {}

    def verify(
        self, crop: RgbImage, candidates: Sequence[Candidate]
    ) -> tuple[GeometricVerdict, ...]:
        """Compara a regiao com as referencias dos candidatos mais promissores.

        Args:
            crop: Regiao recortada.
            candidates: Candidatos do banco, ordenados por similaridade.

        Returns:
            Um veredito por referencia comparada. Tupla vazia quando nenhum
            candidato passa do portao ou quando a regiao nao tem pontos.
        """
        if not self._config.enabled or not candidates:
            return ()

        selected = self._select(candidates)
        if not selected:
            return ()

        self._prepare()
        points, descriptors = self._extract(self._to_gray(crop))
        if descriptors is None or len(points) < _MIN_POINTS:
            quantity = len(points) if points is not None else 0
            return tuple(
                GeometricVerdict(
                    brand=candidate.brand,
                    inliers=0,
                    matches=0,
                    confirms=False,
                    reason=f"regiao com {quantity} pontos, minimo {_MIN_POINTS}",
                )
                for candidate in selected
            )

        return tuple(self._verify_pair(points, descriptors, candidate) for candidate in selected)

    # -- etapas internas ---------------------------------------------------

    def _select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Seleciona no maximo um candidato por marca, acima do portao.

        Uma referencia por marca porque o objetivo e decidir **entre** marcas —
        gastar comparacao em quatro fotos da mesma marca nao acrescenta
        informacao para essa decisao.

        Args:
            candidates: Candidatos ordenados por similaridade.

        Returns:
            Os selecionados, na ordem original.
        """
        selected: list[Candidate] = []
        seen_brands: set[str] = set()
        for candidate in candidates:
            if candidate.similarity < self._config.entry_similarity:
                continue
            if candidate.brand in seen_brands:
                continue
            seen_brands.add(candidate.brand)
            selected.append(candidate)
            if len(selected) >= self._config.max_references:
                break
        return selected

    def _prepare(self) -> None:
        """Constroi o extrator de pontos e o casador. Idempotente.

        Raises:
            RuntimeError: Se a biblioteca de visao nao estiver disponivel.
        """
        if self._extractor is not None:
            return
        cv2 = self._cv2()
        self._extractor = cv2.SIFT_create(nfeatures=self._config.max_points)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def _cv2() -> Any:
        """Importa a biblioteca de visao computacional.

        Returns:
            O modulo cv2.

        Raises:
            RuntimeError: Se a biblioteca nao estiver instalada.
        """
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "a verificacao geometrica exige opencv: "
                "poetry add opencv-contrib-python-headless"
            ) from error
        return cv2

    def _extract(self, gray: NDArray[np.uint8]) -> tuple[Any, Any]:
        """Extrai pontos e descritores de uma imagem em tons de cinza.

        Args:
            gray: Imagem como matriz de inteiros sem sinal.

        Returns:
            Tupla `(pontos, descritores)`. Descritores pode ser None.
        """
        points, descriptors = self._extractor.detectAndCompute(gray, None)
        return points, descriptors

    def _extract_reference(self, path: str) -> tuple[Any, Any]:
        """Extrai pontos de uma referencia, reaproveitando entre chamadas.

        O cache importa: cada referencia e comparada com muitas regioes ao longo
        de uma execucao, e reextrair a mesma imagem seria a maior parte do custo
        desta camada.

        Args:
            path: Caminho da imagem de referencia.

        Returns:
            Tupla `(pontos, descritores)`.
        """
        cached = self._reference_descriptors.get(path)
        if cached is not None:
            return cached

        self._prepare()
        with Image.open(path) as image:
            gray = self._to_gray(image.convert("RGB"))
        extracted = self._extract(gray)
        self._reference_descriptors[path] = extracted
        return extracted

    @staticmethod
    def _to_gray(image: RgbImage) -> NDArray[np.uint8]:
        """Converte a imagem para matriz de tons de cinza.

        Args:
            image: Imagem em RGB.

        Returns:
            Matriz de inteiros sem sinal.
        """
        return np.asarray(image.convert("L"), dtype=np.uint8)

    def _verify_pair(self, points: Any, descriptors: Any, candidate: Candidate) -> GeometricVerdict:
        """Verifica a regiao contra uma referencia especifica.

        Args:
            points: Pontos extraidos da regiao.
            descriptors: Descritores da regiao.
            candidate: Candidato cuja referencia sera comparada.

        Returns:
            O veredito, com o motivo preenchido quando nao confirma.
        """
        cv2 = self._cv2()
        try:
            reference_points, reference_descriptors = self._extract_reference(candidate.reference)
        except Exception as error:  # noqa: BLE001 - referencia ilegivel nao derruba a analise
            return GeometricVerdict(candidate.brand, 0, 0, False, f"referencia ilegivel: {error}")

        if reference_descriptors is None or len(reference_points) < _MIN_POINTS:
            return GeometricVerdict(candidate.brand, 0, 0, False, "referencia com poucos pontos")

        good = self._filter_by_ratio(descriptors, reference_descriptors)
        if len(good) < _POINTS_FOR_HOMOGRAPHY:
            return GeometricVerdict(
                candidate.brand,
                0,
                len(good),
                False,
                f"{len(good)} correspondencias, minimo {_POINTS_FOR_HOMOGRAPHY}",
            )

        source = np.array([points[pair.queryIdx].pt for pair in good], dtype=np.float32).reshape(
            -1, 1, 2
        )
        target = np.array(
            [reference_points[pair.trainIdx].pt for pair in good], dtype=np.float32
        ).reshape(-1, 1, 2)
        transform, mask = cv2.findHomography(
            source, target, cv2.RANSAC, self._config.reprojection_error
        )

        if transform is None or mask is None:
            return GeometricVerdict(
                candidate.brand, 0, len(good), False, "sem transformacao coerente"
            )

        inliers = int(mask.sum())
        if self._is_degenerate(transform):
            return GeometricVerdict(
                candidate.brand,
                inliers,
                len(good),
                False,
                "transformacao degenerada (colapso ou reflexao)",
            )
        if inliers < self._config.min_inliers:
            return GeometricVerdict(
                candidate.brand,
                inliers,
                len(good),
                False,
                f"{inliers} inliers, minimo {self._config.min_inliers}",
            )
        return GeometricVerdict(candidate.brand, inliers, len(good), True)

    def _filter_by_ratio(self, descriptors: Any, reference_descriptors: Any) -> list[Any]:
        """Mantem apenas correspondencias claramente melhores que a segunda opcao.

        Sem este filtro o ajuste robusto recebe ruido demais e encontra
        transformacao coerente em praticamente qualquer par.

        Args:
            descriptors: Descritores da regiao.
            reference_descriptors: Descritores da referencia.

        Returns:
            Correspondencias que passaram no teste de razao.
        """
        cv2 = self._cv2()
        try:
            neighbors = self._matcher.knnMatch(descriptors, reference_descriptors, k=2)
        except cv2.error:
            return []

        ratio = self._config.lowe_ratio
        return [
            best
            for pair in neighbors
            if len(pair) == 2
            for best, second in [pair]
            if best.distance < ratio * second.distance
        ]

    @staticmethod
    def _is_degenerate(transform: NDArray[np.float64]) -> bool:
        """Rejeita transformacao sem sentido fisico.

        Determinante proximo de zero significa que a transformacao colapsa o
        plano numa linha; determinante negativo significa reflexao. Nenhum dos
        dois corresponde a "mesmo desenho visto de outro angulo", mas o ajuste
        robusto produz ambos com prazer quando os pontos sao ruido.

        Args:
            transform: Matriz de homografia.

        Returns:
            True se a transformacao deve ser descartada.
        """
        try:
            determinant = float(np.linalg.det(transform[:2, :2]))
        except (ValueError, np.linalg.LinAlgError):
            return True
        return not np.isfinite(determinant) or abs(determinant) < 1e-4 or determinant < 0
