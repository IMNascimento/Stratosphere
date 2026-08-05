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

Typical usage:
    codificador = Dinov2Codificador(config, dispositivo="cuda:0", precisao="float16")
    vetores = codificador.codificar([recorte_a, recorte_b])
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from application.ports.i_codificador import ICodificador
from application.ports.i_fonte_de_imagens import ImagemRgb
from config.settings import CodificadorConfig

AGREGACOES_VALIDAS: frozenset[str] = frozenset({"global", "media", "concatenado", "centro"})


class Dinov2Codificador(ICodificador):
    """Transforma recortes em vetores L2-normalizados comparaveis por cosseno."""

    def __init__(self, config: CodificadorConfig, dispositivo: str, precisao: str) -> None:
        """Guarda a configuracao sem carregar pesos.

        Args:
            config: Parametros de codificacao, incluindo a agregacao.
            dispositivo: Onde o modelo roda.
            precisao: Precisao numerica dos pesos.

        Raises:
            ValueError: Se a agregacao configurada nao existir.
        """
        if config.agregacao not in AGREGACOES_VALIDAS:
            raise ValueError(
                f"agregacao {config.agregacao!r} desconhecida. "
                f"Validas: {sorted(AGREGACOES_VALIDAS)}"
            )
        self._config = config
        self._dispositivo = dispositivo
        self._precisao = precisao
        self._modelo: Any = None
        self._processador: Any = None
        self._dimensao = 0

    def preparar(self) -> None:
        """Carrega o modelo e o processador. Idempotente."""
        if self._modelo is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._processador = AutoImageProcessor.from_pretrained(self._config.identificador)
        modelo: Any = AutoModel.from_pretrained(
            self._config.identificador, dtype=getattr(torch, self._precisao)
        )
        modelo = modelo.to(self._dispositivo)
        self._modelo = modelo.eval()
        self._torch = torch
        largura_oculta = int(self._modelo.config.hidden_size)
        self._dimensao = (
            largura_oculta * 2 if self._config.agregacao == "concatenado" else largura_oculta
        )

    def codificar(self, recortes: Sequence[ImagemRgb]) -> NDArray[np.float32]:
        """Converte recortes em vetores L2-normalizados.

        Args:
            recortes: Regioes ja recortadas e no tamanho esperado.

        Returns:
            Matriz `(len(recortes), dimensao)` em float32, cada linha com norma
            unitaria. Matriz vazia quando `recortes` e vazio.
        """
        self.preparar()
        itens = list(recortes)
        if not itens:
            return np.zeros((0, max(1, self._dimensao)), dtype=np.float32)

        blocos: list[NDArray[np.float32]] = []
        tamanho = max(1, self._config.tamanho_do_lote)
        for inicio in range(0, len(itens), tamanho):
            blocos.append(self._codificar_lote(itens[inicio : inicio + tamanho]))

        matriz = np.vstack(blocos)
        self._dimensao = int(matriz.shape[1])
        return self._normalizar(matriz)

    def identificacao(self) -> str:
        """Retorna a assinatura estavel do codificador.

        Inclui a agregacao porque o mesmo modelo com agregacoes diferentes
        produz espacos vetoriais distintos — e as dimensoes podem coincidir por
        acaso, o que tornaria o erro invisivel.

        Returns:
            Texto no formato `modelo|agregacao|lado`.
        """
        return (
            f"{self._config.identificador}"
            f"|{self._config.agregacao}"
            f"|{self._config.lado_do_recorte}"
        )

    def dimensao(self) -> int:
        """Retorna a dimensao dos vetores produzidos."""
        return self._dimensao

    # -- etapas internas ---------------------------------------------------

    def _codificar_lote(self, recortes: list[ImagemRgb]) -> NDArray[np.float32]:
        """Codifica um lote de recortes.

        Args:
            recortes: Recortes do lote.

        Returns:
            Matriz `(len(recortes), dimensao)` ainda sem normalizar.
        """
        entradas = self._processador(images=recortes, return_tensors="pt")
        tensores = {
            chave: (
                valor.to(self._dispositivo, dtype=getattr(self._torch, self._precisao))
                if valor.is_floating_point()
                else valor.to(self._dispositivo)
            )
            for chave, valor in entradas.items()
        }

        with self._torch.inference_mode():
            saida = self._modelo(**tensores)

        return self._agregar(saida.last_hidden_state).float().cpu().numpy()

    def _agregar(self, estados: Any) -> Any:
        """Reduz os tokens do modelo a um vetor por imagem.

        Args:
            estados: Tensor `(lote, 1 + retalhos, dimensao)`.

        Returns:
            Tensor `(lote, dimensao)` conforme a agregacao configurada.
        """
        global_ = estados[:, 0]
        retalhos = estados[:, 1:]

        if self._config.agregacao == "global":
            return global_
        if self._config.agregacao == "media":
            return retalhos.mean(dim=1)
        if self._config.agregacao == "concatenado":
            return self._torch.cat([global_, retalhos.mean(dim=1)], dim=-1)
        return self._media_central(retalhos)

    def _media_central(self, retalhos: Any) -> Any:
        """Media apenas do quarto central dos retalhos.

        Args:
            retalhos: Tensor `(lote, retalhos, dimensao)`.

        Returns:
            Tensor `(lote, dimensao)`. Cai para a media completa quando os
            retalhos nao formam uma grade quadrada — situacao possivel em
            modelos com tokens extras, e melhor degradar que quebrar.
        """
        quantidade = int(retalhos.shape[1])
        lado = int(round(quantidade**0.5))
        if lado * lado != quantidade:
            return retalhos.mean(dim=1)

        grade = retalhos.reshape(retalhos.shape[0], lado, lado, -1)
        inicio = lado // 4
        fim = lado - lado // 4
        return grade[:, inicio:fim, inicio:fim].mean(dim=(1, 2))

    @staticmethod
    def _normalizar(matriz: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normaliza cada linha para norma unitaria.

        Args:
            matriz: Vetores brutos.

        Returns:
            A mesma matriz com linhas de norma 1. Linha nula permanece nula em
            vez de virar NaN.
        """
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        return (matriz / np.maximum(normas, 1e-12)).astype(np.float32)
