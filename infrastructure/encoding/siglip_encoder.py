"""Codificador com supervisao de texto - descreve a MARCA, nao a superficie.

--------------------------------------------------------------------------
O PROBLEMA QUE ISTO RESOLVE, E ELE ERA A CAUSA RAIZ
--------------------------------------------------------------------------
O DINOv2 e auto-supervisionado: aprende o que faz uma imagem parecer com outra
sem ninguem dizer o que importa. Para logo, ele decidiu que o que importa e a
**superficie** - o painel, a iluminacao, a moldura, a textura do tecido.

O experimento que revelou isso estava embutido nos proprios dados: varias
referencias vem da MESMA foto de backdrop, recortadas em marcas diferentes. Da
para montar dois conjuntos sem rotular nada:

    POSITIVO  mesma marca, fotos de origem diferentes
    NEGATIVO  MESMA foto de origem, marcas diferentes

Um codificador que descreve a marca separa isso com folga. Medido, com 1134
pares positivos e 288 negativos:

    codificador                                AUC     pos     neg
    dinov2-base  media       (imagem toda)    0.397   0.668   0.743
    dinov2-base  global      (imagem toda)    0.396   0.527   0.605
    dinov2-base  centro      (o que rodava)   0.515   0.671   0.676
    dinov2-base  centro, recorte 50%          0.655   0.692   0.594
    clip-vit-large-patch14                    0.713   0.763   0.669
    siglip2-base-patch16-224                  0.812   0.868   0.728

**0.515 e moeda.** O vetor antigo nao distinguia "mesma marca" de "mesma foto":
dois logos DIFERENTES do mesmo painel pontuavam 0.676, mais alto que duas fotos
da MESMA marca (0.671). Isso explica de uma vez `amazon x azul` em 0.903,
`sportv x tvglobo` em 0.883, `itau` lido como `sadia`, os 238 pares confundiveis
do banco, e o consenso desabando em backdrop - os 25 vizinhos eram outros
paineis, nao outras aparicoes da marca.

`media` e `global` ficarem ABAIXO de 0.5 e o mesmo fato pelo avesso: usar a
imagem inteira e pior que chutar, porque o fundo domina o vetor.

--------------------------------------------------------------------------
POR QUE SUPERVISAO DE TEXTO MUDA ISSO
--------------------------------------------------------------------------
SigLIP2 foi treinado casando imagem com legenda. Legenda fala de **marca**, nao
de "retangulo branco com moldura escura". O sinal de treino empurra o vetor para
o que uma pessoa nomearia na imagem - que e exatamente a pergunta do produto.

E ha uma consequencia pratica que confirma o mecanismo: apertar o recorte AJUDA
o DINOv2 (0.515 -> 0.655, tirando painel do quadro) e ATRAPALHA o SigLIP2
(0.812 -> 0.778). O SigLIP2 usa o contexto de forma produtiva porque entende o
que esta olhando; o DINOv2 precisava que o contexto fosse escondido dele.

--------------------------------------------------------------------------
TROCAR ISTO INVALIDA O BANCO E OS LIMIARES
--------------------------------------------------------------------------
`signature()` inclui o identificador do modelo, entao um indice construido com o
DINOv2 e recusado na carga com `IncompatibleEncoderError` - nao degrada em
silencio. **Reconstrua o banco e recalibre.** A escala muda: onde o DINOv2 dava
0.671 de mediana entre acertos, o SigLIP2 da 0.868.

Typical usage:
    encoder = SiglipEncoder(config, device="cuda:0", precision="float16")
    vectors = encoder.encode(crops)
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from application.ports.i_encoder import IEncoder
from application.ports.i_image_source import RgbImage
from config.settings import EncoderConfig


class SiglipEncoder(IEncoder):
    """Transforma recortes em vetores L2-normalizados com um modelo imagem-texto."""

    def __init__(self, config: EncoderConfig, device: str, precision: str) -> None:
        """Guarda a configuracao sem carregar pesos.

        Args:
            config: Parametros de codificacao. `aggregation` e ignorada - o
                modelo tem uma cabeca de pooling propria, treinada junto com o
                resto, e substitui-la por media de retalhos desperdicaria
                justamente a parte que aprendeu a resumir a imagem.
            device: Onde o modelo roda.
            precision: Precisao numerica dos pesos.
        """
        self._config = config
        self._device = device
        self._precision = precision
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._dimension = 0

    def prepare(self) -> None:
        """Carrega o modelo e o processador. Idempotente."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(self._config.identifier)
        model: Any = AutoModel.from_pretrained(
            self._config.identifier, dtype=getattr(torch, self._precision)
        )
        # So a torre visual: a de texto nao e usada e ocuparia VRAM a toa.
        self._model = getattr(model, "vision_model", model).to(self._device).eval()
        self._torch = torch
        self._dimension = int(self._model.config.hidden_size)

    def encode(self, crops: Sequence[RgbImage]) -> NDArray[np.float32]:
        """Converte recortes em vetores L2-normalizados.

        Args:
            crops: Regioes ja recortadas.

        Returns:
            Matriz `(len(crops), dimensao)` em float32, cada linha com norma
            unitaria. Matriz vazia quando `crops` e vazio.
        """
        self.prepare()
        items = list(crops)
        if not items:
            return np.zeros((0, max(1, self._dimension)), dtype=np.float32)

        blocks: list[NDArray[np.float32]] = []
        size = max(1, self._config.batch_size)
        for start in range(0, len(items), size):
            blocks.append(self._encode_batch(items[start : start + size]))

        matrix = np.vstack(blocks)
        # NaN aqui e sempre estouro numerico, e quase sempre float16 num modelo
        # grande demais para ele. Sem esta checagem o banco e gravado inteiro de
        # NaN e a falha so aparece muito depois, na primeira busca, como
        # "similaridade fora de [-1, 1]" - medido: 753 de 753 referencias.
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"{self._config.identifier!r} produziu vetor nao finito em "
                f"{self._precision}. Rode com STRATOSPHERE_PRECISION=float32."
            )
        self._dimension = int(matrix.shape[1])
        return self._normalize(matrix)

    def signature(self) -> str:
        """Retorna a assinatura estavel do codificador.

        Returns:
            Texto no formato `modelo|pooler|lado`. O `pooler` fixo registra que
            a agregacao nao vem de `EncoderConfig` - quem ler a assinatura de um
            indice antigo consegue saber com o que ele foi construido.
        """
        return f"{self._config.identifier}|pooler|{self._config.crop_side}"

    def dimension(self) -> int:
        """Retorna a dimensao dos vetores produzidos.

        Returns:
            A dimensao, ou 0 antes da primeira codificacao.
        """
        return self._dimension

    # -- etapas internas ---------------------------------------------------

    def _encode_batch(self, crops: list[RgbImage]) -> NDArray[np.float32]:
        """Codifica um lote.

        Args:
            crops: Recortes do lote.

        Returns:
            Matriz `(len(crops), dimensao)` ainda sem normalizar.
        """
        inputs = self._processor(images=crops, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self._model(**inputs)

        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            # Sem cabeca de pooling, media dos retalhos e o resumo disponivel.
            pooled = output.last_hidden_state.mean(dim=1)
        result: NDArray[np.float32] = pooled.float().cpu().numpy()
        return result

    @staticmethod
    def _normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normaliza cada linha para norma unitaria.

        E o que permite a busca ser um produto interno - ver `IEncoder`.

        Args:
            matrix: Vetores brutos.

        Returns:
            Os mesmos vetores com norma 1. Linha nula fica nula.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result: NDArray[np.float32] = (matrix / norms).astype(np.float32)
        return result
