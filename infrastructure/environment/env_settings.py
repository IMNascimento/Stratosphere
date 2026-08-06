"""Leitura do `.env` e do ambiente do processo.

**Este e o unico modulo que le variavel de ambiente.** `config/settings.py` segue
sem conhecer ambiente e sem conter credencial: ele so declara dataclasses com
default. Quem quiser sobrescrever passa por aqui.

--------------------------------------------------------------------------
PRECEDENCIA
--------------------------------------------------------------------------
Do mais fraco para o mais forte:

    default do dataclass  <  .env  <  variavel ja exportada no shell  <  flag da CLI

O `.env` **nao sobrescreve** variavel ja presente no ambiente — e o comportamento
padrao do `dotenv`, e o que permite CI e container injetarem valor sem apagar o
arquivo. As flags da CLI vem por ultimo porque sao a decisao mais explicita que
alguem pode tomar.

--------------------------------------------------------------------------
CREDENCIAL NAO VIRA CONFIGURACAO
--------------------------------------------------------------------------
`HF_TOKEN` nao entra em `AppConfig` e nao e lido por nenhuma camada: ele so
precisa existir em `os.environ` para que o `huggingface_hub` o use ao baixar peso
de modelo com acesso restrito. Carregar e nao guardar mantem a credencial fora de
qualquer objeto que possa acabar num log ou num relatorio.

Typical usage:
    load_env_file()
    config = apply_env_overrides(AppConfig())
"""

import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from config.settings import AppConfig

# Arquivo lido por default, relativo ao diretorio de trabalho.
ENV_FILE = Path(".env")

# Nome canonico do token no `huggingface_hub`. Nao e lido aqui — so precisa
# chegar a `os.environ` para o download de peso restrito funcionar.
HF_TOKEN_VARIABLE = "HF_TOKEN"

# Modo offline do hub. Ligado, nenhuma requisicao sai — o que estiver em cache e
# usado como esta, e o que faltar vira erro em vez de download.
HF_OFFLINE_VARIABLE = "HF_HUB_OFFLINE"

_DEVICE = "STRATOSPHERE_DEVICE"
_PRECISION = "STRATOSPHERE_PRECISION"
_DETECTOR_MODEL = "STRATOSPHERE_DETECTOR_MODEL"
_ENCODER_MODEL = "STRATOSPHERE_ENCODER_MODEL"
_JUDGE_MODEL = "STRATOSPHERE_JUDGE_MODEL"


def load_env_file(path: Path = ENV_FILE) -> bool:
    """Carrega o `.env` para o ambiente do processo, se ele existir.

    Args:
        path: Arquivo a carregar. Ausencia nao e erro — rodar sem `.env` e o
            caso normal em maquina que ja exporta as variaveis.

    Returns:
        True se o arquivo existia e foi lido.
    """
    if not path.is_file():
        return False
    load_dotenv(dotenv_path=path, override=False)
    return True


def apply_env_overrides(config: AppConfig) -> AppConfig:
    """Devolve a configuracao com o que o ambiente sobrescreveu.

    Args:
        config: Configuracao com os defaults do projeto.

    Returns:
        Nova instancia. Variavel ausente ou vazia nao sobrescreve nada — string
        vazia num `.env` significa "nao configurei", nunca "quero vazio".
    """
    device = _text(_DEVICE)
    precision = _text(_PRECISION)
    detector_model = _text(_DETECTOR_MODEL)
    encoder_model = _text(_ENCODER_MODEL)
    judge_model = _text(_JUDGE_MODEL)

    if detector_model is not None:
        config = replace(config, detector=replace(config.detector, identifier=detector_model))
    if encoder_model is not None:
        config = replace(config, encoder=replace(config.encoder, identifier=encoder_model))
    if judge_model is not None:
        config = replace(config, judge=replace(config.judge, model=judge_model))
    if device is not None:
        config = replace(config, device=device)
    if precision is not None:
        config = replace(config, precision=precision)
    return config


def has_hf_token() -> bool:
    """Indica se ha token do Hugging Face no ambiente.

    Serve ao diagnostico do subcomando `ambiente`: modelo de acesso restrito
    falha no download, depois de baixar o resto, quando o token nao esta la.

    Returns:
        True se a variavel esta preenchida. O valor nunca e devolvido nem
        registrado em log.
    """
    return bool(os.environ.get(HF_TOKEN_VARIABLE, "").strip())


def hub_cache_state() -> tuple[Path, float, bool]:
    """Descreve onde os pesos ja baixados estao e se o hub sai a rede.

    Peso de modelo e baixado **uma vez** e fica em disco. O que se repete a cada
    execucao e uma revalidacao de metadado — barata, mas visivel, e impossivel
    sem rede. Ligar o modo offline pula essa ida e usa o que ja esta la.

    Returns:
        A pasta de cache, quanto ela ocupa em GB, e se o modo offline esta
        ligado. O tamanho e 0.0 enquanto nada foi baixado.
    """
    default = Path.home() / ".cache" / "huggingface"
    folder = Path(_text("HF_HOME") or _text("HUGGINGFACE_HUB_CACHE") or default)
    size = 0.0
    if folder.is_dir():
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1024**3
    return folder, size, (_text(HF_OFFLINE_VARIABLE) or "0") not in ("0", "false", "False")


def _text(variable: str) -> str | None:
    """Le uma variavel de ambiente de texto.

    Args:
        variable: Nome da variavel.

    Returns:
        O valor sem espacos nas pontas, ou None se ausente ou vazio.
    """
    value = os.environ.get(variable, "").strip()
    return value or None
