"""Codificador de imagem — a peca que identifica a marca sem treinar por marca.

Usa um modelo auto-supervisionado, que aprendeu correspondencia de forma e
textura sem rotulo. Isso casa melhor com "e o mesmo desenho?" do que um modelo
alinhado a texto, que puxa para semantica ("isto e roupa esportiva") em vez de
aparencia ("esta forma e aquela forma").

--------------------------------------------------------------------------
A AGREGACAO E O PARAMETRO MAIS IMPORTANTE DESTE ARQUIVO
--------------------------------------------------------------------------
O modelo produz um token global e um token por retalho da imagem. Como reduzir
isso a um vetor muda o resultado mais do que trocar de modelo:

- `global` — so o token global. Perde detalhe fino de forma.
- `media` — media de todos os retalhos. **Dilui o logo no fundo**: um recorte de
  logo em painel de patrocinio e majoritariamente parede lisa, e a media descreve
  a parede.
- `concatenado` — global mais media. O compromisso usual, e herda o problema da
  media.
- `centro` — media apenas do quarto central dos retalhos. **O padrao.** O
  detector ja centra a caixa no logo, entao a borda do recorte e contexto por
  construcao. Ignorar a borda faz o vetor descrever a marca em vez da cena.

A diferenca aparece no tipo de erro: com agregacao que inclui a borda, regioes
de marcas diferentes no MESMO tipo de painel ficam proximas — o vetor descreve o
painel, nao o logo.

Os nomes das agregacoes seguem em portugues de proposito: eles entram na
assinatura gravada no manifesto do banco, e renomea-los invalidaria todo indice
ja construido.

Typical usage:
    encoder = Dinov2Encoder(config, device="cuda:0", precision="float16")
    vectors = encoder.encode([crop_a, crop_b])
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from application.ports.i_encoder import IEncoder
from application.ports.i_image_source import RgbImage
from config.settings import EncoderConfig

VALID_AGGREGATIONS: frozenset[str] = frozenset({"global", "media", "concatenado", "centro"})


class Dinov2Encoder(IEncoder):
    """Transforma recortes em vetores L2-normalizados comparaveis por cosseno."""

    def __init__(self, config: EncoderConfig, device: str, precision: str) -> None:
        """Guarda a configuracao sem carregar pesos.

        Args:
            config: Parametros de codificacao, incluindo a agregacao.
            device: Onde o modelo roda.
            precision: Precisao numerica dos pesos.

        Raises:
            ValueError: Se a agregacao configurada nao existir.
        """
        if config.aggregation not in VALID_AGGREGATIONS:
            raise ValueError(
                f"agregacao {config.aggregation!r} desconhecida. "
                f"Validas: {sorted(VALID_AGGREGATIONS)}"
            )
        self._config = config
        self._device = device
        self._precision = precision
        self._model: Any = None
        self._processor: Any = None
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
        model = model.to(self._device)
        self._model = model.eval()
        self._torch = torch
        hidden_width = int(self._model.config.hidden_size)
        self._dimension = (
            hidden_width * 2 if self._config.aggregation == "concatenado" else hidden_width
        )

    def encode(self, crops: Sequence[RgbImage]) -> NDArray[np.float32]:
        """Converte recortes em vetores L2-normalizados.

        Args:
            crops: Regioes ja recortadas e no tamanho esperado.

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
        self._dimension = int(matrix.shape[1])
        return self._normalize(matrix)

    def signature(self) -> str:
        """Retorna a assinatura estavel do codificador.

        Inclui a agregacao porque o mesmo modelo com agregacoes diferentes
        produz espacos vetoriais distintos — e as dimensoes podem coincidir por
        acaso, o que tornaria o erro invisivel.

        Returns:
            Texto no formato `modelo|agregacao|lado`.
        """
        return (
            f"{self._config.identifier}" f"|{self._config.aggregation}" f"|{self._config.crop_side}"
        )

    def dimension(self) -> int:
        """Retorna a dimensao dos vetores produzidos."""
        return self._dimension

    # -- etapas internas ---------------------------------------------------

    def _encode_batch(self, crops: list[RgbImage]) -> NDArray[np.float32]:
        """Codifica um lote de recortes.

        Args:
            crops: Recortes do lote.

        Returns:
            Matriz `(len(crops), dimensao)` ainda sem normalizar.
        """
        inputs = self._processor(images=crops, return_tensors="pt")
        tensors = {
            key: (
                value.to(self._device, dtype=getattr(self._torch, self._precision))
                if value.is_floating_point()
                else value.to(self._device)
            )
            for key, value in inputs.items()
        }

        with self._torch.inference_mode():
            output = self._model(**tensors)

        return self._aggregate(output.last_hidden_state).float().cpu().numpy()

    def _aggregate(self, states: Any) -> Any:
        """Reduz os tokens do modelo a um vetor por imagem.

        Args:
            states: Tensor `(lote, 1 + retalhos, dimensao)`.

        Returns:
            Tensor `(lote, dimensao)` conforme a agregacao configurada.
        """
        global_ = states[:, 0]
        patches = states[:, 1:]

        if self._config.aggregation == "global":
            return global_
        if self._config.aggregation == "media":
            return patches.mean(dim=1)
        if self._config.aggregation == "concatenado":
            return self._torch.cat([global_, patches.mean(dim=1)], dim=-1)
        return self._center_mean(patches)

    def _center_mean(self, patches: Any) -> Any:
        """Media apenas do quarto central dos retalhos.

        Args:
            patches: Tensor `(lote, retalhos, dimensao)`.

        Returns:
            Tensor `(lote, dimensao)`. Cai para a media completa quando os
            retalhos nao formam uma grade quadrada — situacao possivel em
            modelos com tokens extras, e melhor degradar que quebrar.
        """
        quantity = int(patches.shape[1])
        side = int(round(quantity**0.5))
        if side * side != quantity:
            return patches.mean(dim=1)

        grid = patches.reshape(patches.shape[0], side, side, -1)
        start = side // 4
        end = side - side // 4
        return grid[:, start:end, start:end].mean(dim=(1, 2))

    @staticmethod
    def _normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normaliza cada linha para norma unitaria.

        Args:
            matrix: Vetores brutos.

        Returns:
            A mesma matriz com linhas de norma 1. Linha nula permanece nula em
            vez de virar NaN.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
