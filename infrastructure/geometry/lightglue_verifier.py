"""Verificacao geometrica com pontos aprendidos - DISK para detectar, LightGlue para casar.

Mesma pergunta do `SiftVerifier` e mesmo contrato: "e literalmente o mesmo desenho,
sob uma transformacao coerente?". Muda quem responde, e isso muda tudo.

--------------------------------------------------------------------------
POR QUE O SIFT NAO SERVIA
--------------------------------------------------------------------------
SIFT procura **canto**. Logo chapado - swoosh, wordmark, simbolo vetorial - nao
tem canto, e a camada inteira emudece exatamente onde o resto da pipeline mais
precisa dela. Medido em 60 pares de referencia da mesma marca contra 60 pares de
marcas diferentes, a 448px:

    verificador       AUC    mudo em par certo   recall com precisao 100%
    SIFT              0.711        45%                     0%
    DISK+LightGlue    0.796         3%                    63%
    LoFTR             0.793         5%                    60%

A coluna que decide e a ultima: **`recall com precisao 100%` e quanto do par certo
o verificador confirma sem deixar passar um unico par errado.** No SIFT isso e
zero - nao existe limiar em que ele possa confirmar alguma coisa com seguranca,
porque o melhor par errado empata com quase todo par certo. Ele nunca teve como
tirar regiao da fila humana; so tinha como concordar com quem ja estava decidido.

DISK detecta ponto onde nao ha canto, e LightGlue casa com atencao em vez de
distancia de descritor - ele ve os dois conjuntos de pontos ao mesmo tempo e
decide junto, entao nao precisa do teste de razao de Lowe nem do portao que ele
impunha.

LoFTR empata em qualidade e perde em velocidade e em faixa dinamica (mediana 18
contra 86 inliers em par certo), entao a escolha foi DISK.

--------------------------------------------------------------------------
A ESCALA MUDOU - LIMIAR ANTIGO NAO VALE
--------------------------------------------------------------------------
SIFT devolve mediana de 2 inliers em par certo; DISK+LightGlue devolve 86. Todo
limiar calibrado para SIFT (`GeometryConfig.min_inliers`,
`thresholds.DEFAULT_CONFIDENT_INLIERS`) precisa ser refeito com
`tools/calibrate_thresholds.py`, senao a camada passa a confirmar tudo.

Os pesos sao abertos: DISK e Apache-2.0, LightGlue e Apache-2.0. SuperPoint
ficou de fora de proposito - os pesos do Magic Leap sao **non-commercial**.

Typical usage:
    verifier = LightGlueVerifier(config, device="cuda:0")
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
from shared.logging.logger import get_logger

log = get_logger(__name__)

# Minimo matematico para estimar uma homografia.
_POINTS_FOR_HOMOGRAPHY = 4

# Abaixo disto nem vale tentar casar.
_MIN_POINTS = 6


class LightGlueVerifier(IGeometricVerifier):
    """Confirma correspondencia de desenho com pontos aprendidos e casamento por atencao."""

    def __init__(self, config: GeometryConfig, device: str) -> None:
        """Guarda a configuracao sem carregar pesos.

        Args:
            config: Parametros da verificacao geometrica.
            device: Onde os modelos rodam.
        """
        self._config = config
        self._device = device
        self._detector: Any = None
        self._matcher: Any = None
        self._torch: Any = None
        # Cache por instancia, e nao `lru_cache` no metodo: o decorador guarda
        # `self` na chave e mantem o verificador vivo pelo tempo do processo.
        self._reference_features: dict[str, tuple[Any, Any] | None] = {}

    def verify(
        self, crop: RgbImage, candidates: Sequence[Candidate]
    ) -> tuple[GeometricVerdict, ...]:
        """Compara a regiao com as referencias dos candidatos mais promissores.

        Args:
            crop: Regiao recortada.
            candidates: Candidatos do banco, ordenados por similaridade.

        Returns:
            Um veredito por referencia comparada. Tupla vazia quando nenhum
            candidato passa do portao, quando a regiao nao rende pontos, ou
            quando os pesos nao carregam.
        """
        if not self._config.enabled or not candidates:
            return ()

        selected = self._select(candidates)
        if not selected:
            return ()

        try:
            self._prepare()
            features = self._extract(np.asarray(crop.convert("RGB"), dtype=np.uint8))
        except Exception as error:  # noqa: BLE001 - geometria fora nao derruba a analise
            log.warning("verificacao geometrica indisponivel: %s", error)
            return ()

        if features is None:
            # Silencio, e nao veredito zerado. Um veredito com 0 inliers diz
            # "comparei e nao bate"; aqui nao houve comparacao. Devolver 0
            # faria o roteador multiplicar o peso da geometria por zero e
            # derrubar o teto da pontuacao de uma regiao que nunca teve chance.
            return ()

        verdicts = tuple(self._verify_pair(features, candidate) for candidate in selected)
        return tuple(verdict for verdict in verdicts if verdict is not None)

    # -- etapas internas ---------------------------------------------------

    def _select(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Seleciona no maximo um candidato por marca, acima do portao.

        Uma referencia por marca porque o objetivo e decidir **entre** marcas -
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
        """Carrega o detector e o casador. Idempotente.

        Raises:
            RuntimeError: Se as bibliotecas nao estiverem disponiveis.
        """
        if self._detector is not None:
            return
        try:
            import torch
            from kornia.feature import DISK, LightGlueMatcher, laf_from_center_scale_ori
        except ImportError as error:
            raise RuntimeError(
                "a verificacao geometrica aprendida exige kornia: poetry add kornia"
            ) from error

        self._torch = torch
        self._detector = DISK.from_pretrained("depth").to(self._device).eval()
        self._matcher = LightGlueMatcher("disk").to(self._device).eval()
        self._lafs = laf_from_center_scale_ori

    def _extract(self, image: NDArray[np.uint8]) -> tuple[Any, Any] | None:
        """Extrai pontos e descritores aprendidos de uma imagem.

        A imagem e reamostrada para `GeometryConfig.match_side`. Resolucao
        importa aqui: ponto aprendido em recorte de 224px e escasso, e a mesma
        comparacao a 448px rende varias vezes mais correspondencia.

        Args:
            image: Imagem RGB como matriz de inteiros sem sinal.

        Returns:
            Tupla `(pontos, descritores)`, ou None quando a imagem nao rende
            pontos suficientes para uma comparacao ter sentido.
        """
        import cv2

        side = self._config.match_side
        resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
        tensor = (
            self._torch.from_numpy(resized).permute(2, 0, 1)[None].float().to(self._device) / 255.0
        )
        with self._torch.inference_mode():
            features = self._detector(
                tensor, n=self._config.max_points, pad_if_not_divisible=True
            )[0]
        if features.keypoints.shape[0] < _MIN_POINTS:
            return None
        return features.keypoints, features.descriptors

    def _extract_reference(self, path: str) -> tuple[Any, Any] | None:
        """Extrai pontos de uma referencia, reaproveitando entre chamadas.

        O cache importa: cada referencia e comparada com muitas regioes ao longo
        de uma execucao, e reextrair a mesma imagem seria a maior parte do custo
        desta camada. Referencia que nao rende pontos e cacheada como None, para
        nao ser reprocessada a cada regiao.

        Args:
            path: Caminho da imagem de referencia.

        Returns:
            Tupla `(pontos, descritores)`, ou None quando a referencia nao serve.
        """
        if path in self._reference_features:
            return self._reference_features[path]

        try:
            with Image.open(path) as image:
                extracted = self._extract(np.asarray(image.convert("RGB"), dtype=np.uint8))
        except Exception as error:  # noqa: BLE001 - referencia ilegivel nao derruba a analise
            log.warning("referencia ilegivel, geometria nao opinou: %s", error)
            extracted = None

        self._reference_features[path] = extracted
        return extracted

    def _verify_pair(
        self, features: tuple[Any, Any], candidate: Candidate
    ) -> GeometricVerdict | None:
        """Verifica a regiao contra uma referencia especifica.

        Args:
            features: Pontos e descritores da regiao.
            candidate: Candidato cuja referencia sera comparada.

        Returns:
            O veredito, com o motivo preenchido quando nao confirma. **None
            quando nao houve comparacao** - referencia sem pontos, ou menos de
            quatro pares casados. Ausencia de veredito e diferente de veredito
            negativo, e so o None preserva essa distincao ate o roteador.
        """
        import cv2

        reference = self._extract_reference(candidate.reference)
        if reference is None:
            return None

        points, descriptors = features
        reference_points, reference_descriptors = reference

        with self._torch.inference_mode():
            _, indexes = self._matcher(
                descriptors,
                reference_descriptors,
                self._laf(points),
                self._laf(reference_points),
            )

        matched = int(indexes.shape[0])
        if matched < _POINTS_FOR_HOMOGRAPHY:
            # Tambem e silencio. Abaixo de quatro pares nao existe transformacao
            # a estimar, entao a pergunta desta camada nao chegou a ser feita.
            return None

        source = points[indexes[:, 0]].cpu().numpy().astype(np.float32).reshape(-1, 1, 2)
        target = (
            reference_points[indexes[:, 1]].cpu().numpy().astype(np.float32).reshape(-1, 1, 2)
        )
        transform, mask = cv2.findHomography(
            source, target, cv2.USAC_MAGSAC, self._config.reprojection_error
        )

        if transform is None or mask is None:
            return GeometricVerdict(
                candidate.brand, 0, matched, False, "sem transformacao coerente"
            )

        inliers = int(mask.sum())
        if self._is_degenerate(transform):
            return GeometricVerdict(
                candidate.brand,
                inliers,
                matched,
                False,
                "transformacao degenerada (colapso ou reflexao)",
            )
        if inliers < self._config.min_inliers:
            return GeometricVerdict(
                candidate.brand,
                inliers,
                matched,
                False,
                f"{inliers} inliers, minimo {self._config.min_inliers}",
            )
        return GeometricVerdict(candidate.brand, inliers, matched, True)

    def _laf(self, keypoints: Any) -> Any:
        """Converte pontos em quadros locais, que e o formato que o casador espera.

        Escala e orientacao vao neutras: DISK devolve ponto sem elas, e o
        LightGlue nao as usa para casar - quem resolve rotacao e escala e a
        atencao dele, e nao a geometria do ponto.

        Args:
            keypoints: Pontos detectados.

        Returns:
            Quadros locais no formato do kornia.
        """
        return self._lafs(
            keypoints[None],
            self._torch.ones(1, keypoints.shape[0], 1, 1, device=self._device),
        )

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
