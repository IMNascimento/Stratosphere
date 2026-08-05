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
    verificador = SiftVerificador(config)
    vereditos = verificador.verificar(recorte, candidatos)
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from application.ports.i_fonte_de_imagens import ImagemRgb
from application.ports.i_verificador_geometrico import IVerificadorGeometrico
from config.settings import GeometriaConfig
from domain.value_objects.candidato import Candidato
from domain.value_objects.veredito_geometrico import VereditoGeometrico

# Abaixo disto nem vale tentar casar: o ajuste robusto encontra concordancia em
# qualquer conjunto pequeno de pontos.
_PONTOS_MINIMOS = 6

# Minimo matematico para estimar uma homografia.
_PONTOS_PARA_HOMOGRAFIA = 4


class SiftVerificador(IVerificadorGeometrico):
    """Confirma correspondencia de desenho por casamento de pontos e ajuste robusto."""

    def __init__(self, config: GeometriaConfig) -> None:
        """Guarda a configuracao sem construir o extrator.

        Args:
            config: Parametros da verificacao geometrica.
        """
        self._config = config
        self._extrator: Any = None
        self._casador: Any = None
        # Cache por instancia, e nao `lru_cache` no metodo: o decorador guarda
        # `self` na chave e mantem o verificador e todos os descritores vivos
        # pelo tempo do processo. Com o dicionario aqui, o cache morre junto com
        # a instancia.
        self._descritores_de_referencia: dict[str, tuple[Any, Any]] = {}

    def verificar(
        self, recorte: ImagemRgb, candidatos: Sequence[Candidato]
    ) -> tuple[VereditoGeometrico, ...]:
        """Compara a regiao com as referencias dos candidatos mais promissores.

        Args:
            recorte: Regiao recortada.
            candidatos: Candidatos do banco, ordenados por similaridade.

        Returns:
            Um veredito por referencia comparada. Tupla vazia quando nenhum
            candidato passa do portao ou quando a regiao nao tem pontos.
        """
        if not self._config.ativa or not candidatos:
            return ()

        escolhidos = self._escolher(candidatos)
        if not escolhidos:
            return ()

        self._preparar()
        pontos, descritores = self._extrair(self._para_cinza(recorte))
        if descritores is None or len(pontos) < _PONTOS_MINIMOS:
            quantidade = len(pontos) if pontos is not None else 0
            return tuple(
                VereditoGeometrico(
                    marca=candidato.marca,
                    inliers=0,
                    correspondencias=0,
                    confirma=False,
                    motivo=f"regiao com {quantidade} pontos, minimo {_PONTOS_MINIMOS}",
                )
                for candidato in escolhidos
            )

        return tuple(
            self._verificar_par(pontos, descritores, candidato) for candidato in escolhidos
        )

    # -- etapas internas ---------------------------------------------------

    def _escolher(self, candidatos: Sequence[Candidato]) -> list[Candidato]:
        """Seleciona no maximo um candidato por marca, acima do portao.

        Uma referencia por marca porque o objetivo e decidir **entre** marcas —
        gastar comparacao em quatro fotos da mesma marca nao acrescenta
        informacao para essa decisao.

        Args:
            candidatos: Candidatos ordenados por similaridade.

        Returns:
            Os selecionados, na ordem original.
        """
        escolhidos: list[Candidato] = []
        marcas_vistas: set[str] = set()
        for candidato in candidatos:
            if candidato.similaridade < self._config.similaridade_de_entrada:
                continue
            if candidato.marca in marcas_vistas:
                continue
            marcas_vistas.add(candidato.marca)
            escolhidos.append(candidato)
            if len(escolhidos) >= self._config.maximo_de_referencias:
                break
        return escolhidos

    def _preparar(self) -> None:
        """Constroi o extrator de pontos e o casador. Idempotente.

        Raises:
            RuntimeError: Se a biblioteca de visao nao estiver disponivel.
        """
        if self._extrator is not None:
            return
        cv2 = self._cv2()
        self._extrator = cv2.SIFT_create(nfeatures=self._config.maximo_de_pontos)
        self._casador = cv2.BFMatcher(cv2.NORM_L2)

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
        except ImportError as erro:
            raise RuntimeError(
                "a verificacao geometrica exige opencv: "
                "poetry add opencv-contrib-python-headless"
            ) from erro
        return cv2

    def _extrair(self, cinza: NDArray[np.uint8]) -> tuple[Any, Any]:
        """Extrai pontos e descritores de uma imagem em tons de cinza.

        Args:
            cinza: Imagem como matriz de inteiros sem sinal.

        Returns:
            Tupla `(pontos, descritores)`. Descritores pode ser None.
        """
        pontos, descritores = self._extrator.detectAndCompute(cinza, None)
        return pontos, descritores

    def _extrair_referencia(self, caminho: str) -> tuple[Any, Any]:
        """Extrai pontos de uma referencia, reaproveitando entre chamadas.

        O cache importa: cada referencia e comparada com muitas regioes ao longo
        de uma execucao, e reextrair a mesma imagem seria a maior parte do custo
        desta camada.

        Args:
            caminho: Caminho da imagem de referencia.

        Returns:
            Tupla `(pontos, descritores)`.
        """
        em_cache = self._descritores_de_referencia.get(caminho)
        if em_cache is not None:
            return em_cache

        self._preparar()
        with Image.open(caminho) as imagem:
            cinza = self._para_cinza(imagem.convert("RGB"))
        extraido = self._extrair(cinza)
        self._descritores_de_referencia[caminho] = extraido
        return extraido

    @staticmethod
    def _para_cinza(imagem: ImagemRgb) -> NDArray[np.uint8]:
        """Converte a imagem para matriz de tons de cinza.

        Args:
            imagem: Imagem em RGB.

        Returns:
            Matriz de inteiros sem sinal.
        """
        return np.asarray(imagem.convert("L"), dtype=np.uint8)

    def _verificar_par(
        self, pontos: Any, descritores: Any, candidato: Candidato
    ) -> VereditoGeometrico:
        """Verifica a regiao contra uma referencia especifica.

        Args:
            pontos: Pontos extraidos da regiao.
            descritores: Descritores da regiao.
            candidato: Candidato cuja referencia sera comparada.

        Returns:
            O veredito, com o motivo preenchido quando nao confirma.
        """
        cv2 = self._cv2()
        try:
            pontos_ref, descritores_ref = self._extrair_referencia(candidato.referencia)
        except Exception as erro:  # noqa: BLE001 - referencia ilegivel nao derruba a analise
            return VereditoGeometrico(candidato.marca, 0, 0, False, f"referencia ilegivel: {erro}")

        if descritores_ref is None or len(pontos_ref) < _PONTOS_MINIMOS:
            return VereditoGeometrico(candidato.marca, 0, 0, False, "referencia com poucos pontos")

        boas = self._filtrar_por_razao(descritores, descritores_ref)
        if len(boas) < _PONTOS_PARA_HOMOGRAFIA:
            return VereditoGeometrico(
                candidato.marca,
                0,
                len(boas),
                False,
                f"{len(boas)} correspondencias, minimo {_PONTOS_PARA_HOMOGRAFIA}",
            )

        origem = np.array([pontos[par.queryIdx].pt for par in boas], dtype=np.float32).reshape(
            -1, 1, 2
        )
        destino = np.array([pontos_ref[par.trainIdx].pt for par in boas], dtype=np.float32).reshape(
            -1, 1, 2
        )
        transformacao, mascara = cv2.findHomography(
            origem, destino, cv2.RANSAC, self._config.erro_de_reprojecao
        )

        if transformacao is None or mascara is None:
            return VereditoGeometrico(
                candidato.marca, 0, len(boas), False, "sem transformacao coerente"
            )

        inliers = int(mascara.sum())
        if self._e_degenerada(transformacao):
            return VereditoGeometrico(
                candidato.marca,
                inliers,
                len(boas),
                False,
                "transformacao degenerada (colapso ou reflexao)",
            )
        if inliers < self._config.inliers_minimos:
            return VereditoGeometrico(
                candidato.marca,
                inliers,
                len(boas),
                False,
                f"{inliers} inliers, minimo {self._config.inliers_minimos}",
            )
        return VereditoGeometrico(candidato.marca, inliers, len(boas), True)

    def _filtrar_por_razao(self, descritores: Any, descritores_ref: Any) -> list[Any]:
        """Mantem apenas correspondencias claramente melhores que a segunda opcao.

        Sem este filtro o ajuste robusto recebe ruido demais e encontra
        transformacao coerente em praticamente qualquer par.

        Args:
            descritores: Descritores da regiao.
            descritores_ref: Descritores da referencia.

        Returns:
            Correspondencias que passaram no teste de razao.
        """
        cv2 = self._cv2()
        try:
            vizinhos = self._casador.knnMatch(descritores, descritores_ref, k=2)
        except cv2.error:
            return []

        razao = self._config.razao_de_lowe
        return [
            melhor
            for par in vizinhos
            if len(par) == 2
            for melhor, segundo in [par]
            if melhor.distance < razao * segundo.distance
        ]

    @staticmethod
    def _e_degenerada(transformacao: NDArray[np.float64]) -> bool:
        """Rejeita transformacao sem sentido fisico.

        Determinante proximo de zero significa que a transformacao colapsa o
        plano numa linha; determinante negativo significa reflexao. Nenhum dos
        dois corresponde a "mesmo desenho visto de outro angulo", mas o ajuste
        robusto produz ambos com prazer quando os pontos sao ruido.

        Args:
            transformacao: Matriz de homografia.

        Returns:
            True se a transformacao deve ser descartada.
        """
        try:
            determinante = float(np.linalg.det(transformacao[:2, :2]))
        except (ValueError, np.linalg.LinAlgError):
            return True
        return not np.isfinite(determinante) or abs(determinante) < 1e-4 or determinante < 0
