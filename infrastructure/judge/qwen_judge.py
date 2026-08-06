"""Juiz visual local, sobre um modelo aberto rodando na propria GPU.

Zero custo por chamada e nenhum dado saindo da maquina — a diferenca em relacao
a um juiz de API nao e so preco, e tambem que recorte de imagem de cliente nao
atravessa a rede.

--------------------------------------------------------------------------
POR QUE ELE LE LOGIT EM VEZ DE LER TEXTO
--------------------------------------------------------------------------
O modelo nao **gera** a resposta: a pergunta e feita de modo que a proxima
palavra so possa ser "sim" ou "nao", e o juiz compara a probabilidade das duas.
Isso resolve de uma vez os tres problemas de usar um modelo pequeno como juiz:

- **Nao ha o que parsear.** Modelo de 2B nao segue formato de saida com
  confiabilidade; pedir JSON e depois tentar extrair produz falha aleatoria.
- **A confianca vira numero de verdade.** E a probabilidade que o proprio
  modelo atribuiu, e nao uma nota que ele inventou sobre si mesmo. E o que faz
  o piso de `min_confidence` significar alguma coisa.
- **Uma passada, sem laco de geracao.** Muito mais rapido, e determinista.

--------------------------------------------------------------------------
ELE CONFIRMA OU NEGA — NAO PROPOE OUTRA MARCA
--------------------------------------------------------------------------
Um modelo local pequeno nao sabe o que e `divino_fogao`, `menzoil` ou
`souza_lima`, e perguntar o nome so produziria alucinacao confiante. Entao ele
responde apenas a pergunta que consegue responder olhando as duas imagens: **sao
a mesma marca?**

O roteador ja trata isso — a saida "discordo e sei o nome" simplesmente nunca
dispara com este adaptador, e a regiao segue para revisao humana como antes.

Typical usage:
    judge = QwenJudge(config, device="cuda:0", precision="float16")
    verdict = judge.judge(crop, reference, brand="nike")
"""

from typing import Any

from application.ports.i_image_source import RgbImage
from application.ports.i_judge import IJudge
from config.settings import JudgeConfig
from domain.value_objects.judge_verdict import JudgeVerdict
from shared.logging.logger import get_logger

log = get_logger(__name__)

# Variantes de resposta afirmativa e negativa. Sao varias porque o primeiro
# token de "sim" depende do tokenizador, e errar essa escolha silenciosamente
# transformaria o juiz num gerador de numeros aleatorios.
_YES = ("Yes", " Yes", "yes", "Sim", " Sim", "sim", "SIM")
_NO = ("No", " No", "no", "Nao", " Nao", "nao", "NAO", "Não", "não")

# A pergunta e em ingles porque foi medida melhor: nos mesmos 60 pares, ela
# separa acerto de erro com AUC 0.809 contra 0.699 da versao em portugues. Um
# modelo de 2B tem muito mais treino em ingles, e a diferenca aparece justamente
# no que aqui importa — a ordem entre as respostas, nao o texto delas.
_QUESTION = (
    "Image 1 is a crop from a photo. Image 2 is a reference logo of the brand {brand!r}.\n"
    "Do both images show the logo of the SAME company? Judge by the symbol shape and the "
    "letterforms, ignoring colour, blur and viewing angle.\n"
    "Answer with a single word: Yes or No."
)


class QwenJudge(IJudge):
    """Segunda opiniao visual com um modelo aberto na GPU local."""

    def __init__(self, config: JudgeConfig, device: str, precision: str) -> None:
        """Guarda a configuracao sem carregar pesos.

        Args:
            config: Parametros do juiz, incluindo o modelo a carregar.
            device: Onde o modelo roda.
            precision: Precisao numerica dos pesos.
        """
        self._config = config
        self._device = device
        self._precision = precision
        self._model: Any = None
        self._processor: Any = None
        self._yes: list[int] = []
        self._no: list[int] = []

    def judge(self, crop: RgbImage, reference: RgbImage, brand: str) -> JudgeVerdict | None:
        """Compara a regiao com a referencia e devolve o parecer.

        Args:
            crop: Regiao recortada.
            reference: Imagem de referencia da marca proposta.
            brand: Marca que o banco propos.

        Returns:
            O parecer, ou None em dois casos que dao no mesmo para quem chama:
            o modelo falhou, ou o modelo olhou e ficou na faixa de duvida. Nos
            dois a regiao fica na fila humana — nunca vira rejeicao.
        """
        try:
            self._prepare()
            probability = self._probability_of_yes(crop, reference, brand)
        except Exception as error:  # noqa: BLE001 - juiz indisponivel nao derruba a analise
            log.warning("juiz local indisponivel, regiao segue para revisao humana: %s", error)
            return None

        if self._config.deny_below < probability < self._config.confirm_above:
            return None

        agrees = probability >= self._config.confirm_above
        # A confianca e a distancia relativa dentro da faixa decidida, e nao a
        # probabilidade crua: crua ela e sempre alta e nao diria nada.
        confidence = (
            (probability - self._config.confirm_above) / max(1e-6, 1.0 - self._config.confirm_above)
            if agrees
            else (self._config.deny_below - probability) / max(1e-6, self._config.deny_below)
        )
        return JudgeVerdict(
            agrees=agrees,
            brand=brand if agrees else None,
            confidence=min(1.0, max(0.0, confidence)),
            reason=(
                f"juiz local comparou a regiao com a referencia e "
                f"{'confirmou' if agrees else 'negou'} (p={probability:.3f}, cortes "
                f"{self._config.deny_below:.2f}/{self._config.confirm_above:.2f})"
            ),
        )

    # -- etapas internas ---------------------------------------------------

    def _prepare(self) -> None:
        """Carrega o modelo, o processador e os tokens de resposta. Idempotente."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._config.model)
        model: Any = AutoModelForImageTextToText.from_pretrained(
            self._config.model, dtype=getattr(torch, self._precision)
        )
        self._model = model.to(self._device).eval()
        self._torch = torch

        tokenizer = self._processor.tokenizer
        self._yes = self._first_tokens(tokenizer, _YES)
        self._no = self._first_tokens(tokenizer, _NO)
        if not self._yes or not self._no:
            raise RuntimeError(
                f"nao foi possivel mapear os tokens de sim/nao em {self._config.model!r}"
            )

    @staticmethod
    def _first_tokens(tokenizer: Any, words: tuple[str, ...]) -> list[int]:
        """Mapeia cada variante de resposta no primeiro token que ela produz.

        Args:
            tokenizer: Tokenizador do modelo.
            words: Variantes de escrita da resposta.

        Returns:
            Identificadores de token, sem repeticao.
        """
        found: list[int] = []
        for word in words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids and ids[0] not in found:
                found.append(ids[0])
        return found

    def _chat_prompt(self, conversation: list[dict[str, Any]]) -> str:
        """Monta o prompt garantindo que a proxima palavra seja a RESPOSTA.

        Modelo com modo de raciocinio abre um bloco de pensamento sozinho: o
        template termina em `<think>` e o proximo token e o comeco do raciocinio,
        nao a resposta. Ler logit ali compara `Yes` contra `No` num ponto em que
        o modelo ia escrever "The images show...".

        MEDIDO no Qwen3.5-4B: sem desligar o pensamento, o token mais provavel e
        `The` com 100.0% e o AUC cai para 0.664 — que nao mede nada. No
        Qwen2-VL-2B, que nao tem o modo, o topo e `Yes` com 91.9%.

        Args:
            conversation: Mensagem no formato do template do modelo.

        Returns:
            O prompt pronto para a passada.
        """
        try:
            prompt = self._processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Template sem modo de raciocinio nao aceita o parametro.
            prompt = self._processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        # Rede de seguranca: alguns templates abrem o bloco mesmo com o parametro
        # desligado, e outros ja o abrem E fecham. Contar os dois lados evita
        # tanto deixar aberto quanto fechar duas vezes.
        text = str(prompt)
        if text.count("<think>") > text.count("</think>"):
            text += "</think>\n\n"
        return text

    def _probability_of_yes(self, crop: RgbImage, reference: RgbImage, brand: str) -> float:
        """Calcula a probabilidade de o modelo responder "sim".

        Args:
            crop: Regiao recortada.
            reference: Imagem de referencia.
            brand: Marca proposta pelo banco.

        Returns:
            Probabilidade entre 0.0 e 1.0, normalizada apenas entre as duas
            respostas possiveis — o resto do vocabulario e ignorado de
            proposito, porque a pergunta so admite duas respostas.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": _QUESTION.format(brand=brand)},
                ],
            }
        ]
        prompt = self._chat_prompt(conversation)
        inputs = self._processor(
            text=[prompt], images=[crop, reference], return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            logits = self._model(**inputs).logits[0, -1].float()

        yes = self._torch.logsumexp(logits[self._yes], dim=0)
        no = self._torch.logsumexp(logits[self._no], dim=0)
        return float(self._torch.softmax(self._torch.stack([no, yes]), dim=0)[1])
