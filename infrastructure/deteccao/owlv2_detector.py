"""Detector de vocabulario aberto — encontra ONDE ha marca grafica.

Os prompts sao **conceitos** (`logo`, `emblem`, `wordmark`), nunca nomes de
marca. Nome de marca no prompt reintroduz exatamente o acoplamento que a
arquitetura existe para eliminar: marca nova voltaria a exigir mudanca aqui.

Duas caracteristicas deste tipo de detector, medidas e nao supostas, mudam como
ele deve ser configurado:

**A escala de confianca e comprimida e nao separa bem.** A distribuicao de
confianca em regiao com logo e em regiao sem logo se sobrepoe quase inteiramente.
Consequencia pratica: subir o limiar corta logo real na mesma taxa que corta
ruido — ele nao e um bom filtro de qualidade, so um controle de volume.

**Por isso o limiar padrao e baixo.** O detector so precisa acertar *onde*; a
busca vetorial e a verificacao geometrica filtram *quem*. Limiar alto troca
recall — irrecuperavel, porque a regiao nunca chega ao codificador — por
precisao, que e recuperavel adiante.

Typical usage:
    detector = Owlv2Detector(config, dispositivo="cuda:0")
    detector.preparar()
    deteccoes = detector.detectar(imagem)
"""

from typing import Any

from application.ports.i_detector import IDetector
from application.ports.i_fonte_de_imagens import ImagemRgb
from config.settings import DetectorConfig
from domain.entities.deteccao import Deteccao
from domain.exceptions.domain_exceptions import CaixaInvalidaError
from domain.value_objects.caixa import Caixa


class Owlv2Detector(IDetector):
    """Detector agnostico de marca sobre um modelo de vocabulario aberto."""

    def __init__(self, config: DetectorConfig, dispositivo: str, precisao: str) -> None:
        """Guarda a configuracao sem carregar pesos.

        Args:
            config: Parametros de deteccao.
            dispositivo: Onde o modelo roda, por exemplo `cuda:0` ou `cpu`.
            precisao: Precisao numerica, por exemplo `float16`.
        """
        self._config = config
        self._dispositivo = dispositivo
        self._precisao = precisao
        self._modelo: Any = None
        self._processador: Any = None
        # Modelos de vocabulario aberto respondem melhor a frase que a palavra solta.
        self._consultas = [f"a photo of a {conceito}" for conceito in config.conceitos]

    def preparar(self) -> None:
        """Carrega o modelo e o processador. Idempotente."""
        if self._modelo is not None:
            return

        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._processador = Owlv2Processor.from_pretrained(self._config.identificador)
        modelo: Any = Owlv2ForObjectDetection.from_pretrained(
            self._config.identificador, dtype=getattr(torch, self._precisao)
        )
        modelo = modelo.to(self._dispositivo)
        self._modelo = modelo.eval()
        self._torch = torch

    def detectar(self, imagem: ImagemRgb) -> tuple[Deteccao, ...]:
        """Encontra as regioes da imagem que contem marca grafica.

        Args:
            imagem: Imagem completa em RGB.

        Returns:
            Deteccoes em coordenadas da imagem original, filtradas e suprimidas,
            em ordem decrescente de confianca.
        """
        self.preparar()

        largura, altura = imagem.size
        reduzida, escala = self._reduzir(imagem)
        brutas = self._inferir(reduzida)

        if escala != 1.0:
            inverso = 1.0 / escala
            brutas = [
                (self._reescalar(caixa, inverso), confianca, conceito)
                for caixa, confianca, conceito in brutas
            ]

        return self._pos_processar(brutas, largura, altura)

    # -- etapas internas ---------------------------------------------------

    def _reduzir(self, imagem: ImagemRgb) -> tuple[ImagemRgb, float]:
        """Reduz a imagem para a resolucao de inferencia.

        Reduzir demais faz simbolos pequenos deixarem de existir antes de
        qualquer modelo opinar — por isso o padrao e alto.

        Args:
            imagem: Imagem original.

        Returns:
            Tupla `(imagem_reduzida, escala)`. A escala traz as caixas de volta
            para as coordenadas originais.
        """
        from PIL import Image

        largura, altura = imagem.size
        maior_lado = max(largura, altura)
        alvo = self._config.lado_de_inferencia
        if maior_lado <= alvo:
            return imagem, 1.0

        escala = alvo / maior_lado
        reduzida = imagem.resize(
            (max(1, int(largura * escala)), max(1, int(altura * escala))),
            Image.Resampling.BICUBIC,
        )
        return reduzida, escala

    def _inferir(
        self, imagem: ImagemRgb
    ) -> list[tuple[tuple[float, float, float, float], float, str | None]]:
        """Roda o modelo e devolve caixas brutas na escala da imagem recebida.

        Args:
            imagem: Imagem ja reduzida.

        Returns:
            Lista de `(caixa_xyxy, confianca, conceito)`.
        """
        entradas = self._processador(text=[self._consultas], images=imagem, return_tensors="pt")
        entradas = {chave: valor.to(self._dispositivo) for chave, valor in entradas.items()}

        with self._torch.inference_mode():
            saidas = self._modelo(**entradas)

        largura, altura = imagem.size
        resultados = self._processador.post_process_grounded_object_detection(
            outputs=saidas,
            target_sizes=self._torch.tensor([[altura, largura]], device=self._dispositivo),
            threshold=self._config.limiar_de_confianca,
        )[0]

        brutas = []
        for caixa, confianca, rotulo in zip(
            resultados["boxes"], resultados["scores"], resultados["labels"], strict=True
        ):
            coordenadas = tuple(float(valor) for valor in caixa.tolist())
            indice = int(rotulo)
            conceito = (
                self._config.conceitos[indice] if indice < len(self._config.conceitos) else None
            )
            brutas.append((coordenadas, float(confianca), conceito))
        return brutas  # type: ignore[return-value]

    @staticmethod
    def _reescalar(
        caixa: tuple[float, float, float, float], fator: float
    ) -> tuple[float, float, float, float]:
        """Multiplica as coordenadas pelo fator de volta a escala original.

        Args:
            caixa: Coordenadas xyxy na escala de inferencia.
            fator: Inverso da escala aplicada na reducao.

        Returns:
            Coordenadas na escala da imagem original.
        """
        return (caixa[0] * fator, caixa[1] * fator, caixa[2] * fator, caixa[3] * fator)

    def _pos_processar(
        self,
        brutas: list[tuple[tuple[float, float, float, float], float, str | None]],
        largura: int,
        altura: int,
    ) -> tuple[Deteccao, ...]:
        """Filtra por area e lado, suprime sobreposicoes e limita a quantidade.

        Args:
            brutas: Caixas na escala original.
            largura: Largura da imagem.
            altura: Altura da imagem.

        Returns:
            Deteccoes validas em ordem decrescente de confianca.
        """
        area_da_imagem = float(largura * altura)
        aceitas: list[Deteccao] = []

        for coordenadas, confianca, conceito in brutas:
            caixa = self._prender(coordenadas, largura, altura)
            if caixa is None:
                continue
            fracao = caixa.area / max(1.0, area_da_imagem)
            if fracao < self._config.area_minima_relativa:
                continue
            if fracao > self._config.area_maxima_relativa:
                continue
            if caixa.menor_lado < self._config.lado_minimo:
                continue
            aceitas.append(Deteccao(caixa=caixa, confianca=min(1.0, confianca), conceito=conceito))

        aceitas.sort(key=lambda deteccao: deteccao.confianca, reverse=True)
        suprimidas = self._suprimir(aceitas)
        return tuple(suprimidas[: self._config.maximo_de_regioes])

    @staticmethod
    def _prender(
        coordenadas: tuple[float, float, float, float], largura: int, altura: int
    ) -> Caixa | None:
        """Arredonda e prende as coordenadas aos limites da imagem.

        Args:
            coordenadas: Caixa xyxy, possivelmente fora dos limites.
            largura: Largura da imagem.
            altura: Altura da imagem.

        Returns:
            A caixa valida, ou None se ela degenerar apos o corte.
        """
        x1, y1, x2, y2 = coordenadas
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        try:
            return Caixa(
                x1=int(max(0, min(largura - 1, round(x1)))),
                y1=int(max(0, min(altura - 1, round(y1)))),
                x2=int(max(1, min(largura, round(x2)))),
                y2=int(max(1, min(altura, round(y2)))),
            )
        except CaixaInvalidaError:
            return None

    def _suprimir(self, deteccoes: list[Deteccao]) -> list[Deteccao]:
        """Remove caixas sobrepostas, **sem olhar o conceito que as gerou**.

        A supressao e agnostica de proposito: modelos de vocabulario aberto
        disparam varios conceitos sobre o mesmo pixel — `logo` e `wordmark` no
        mesmo simbolo, por exemplo. Suprimir por conceito deixaria as duas
        caixas passarem, e a mesma regiao seria codificada e cobrada duas vezes.

        Args:
            deteccoes: Deteccoes ja ordenadas por confianca decrescente.

        Returns:
            As que sobreviveram, na mesma ordem.
        """
        if len(deteccoes) <= 1:
            return deteccoes

        mantidas: list[Deteccao] = []
        for candidata in deteccoes:
            sobrepoe = any(
                candidata.caixa.intersecao_sobre_uniao(mantida.caixa)
                > self._config.iou_de_supressao
                for mantida in mantidas
            )
            if not sobrepoe:
                mantidas.append(candidata)
        return mantidas
