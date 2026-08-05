# Stratosphere

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Deteccao de marcas em imagens, em escala, com marcas entrando continuamente no
portfolio.

A ideia central cabe numa frase:

> **Separar *onde tem logo* de *qual logo e*.**
> Um detector agnostico de marca acha as regioes candidatas. Uma busca vetorial
> num banco de referencias diz de quem sao. **Marca nova e uma pasta de imagens
> a mais, nao um ciclo de retreino.**

Analogia: detector de rosto — que acha qualquer rosto, inclusive de quem nunca
viu — somado a reconhecimento facial, que compara com um banco cadastrado.

---

## Por que nao um classificador por marca

Porque o requisito e "marca nova sem retreino", e ele descarta a abordagem
tradicional por dois motivos que se somam:

| | classificador por marca | retrieval (este projeto) |
|---|---|---|
| custo de marca nova | um ciclo de treino | copiar imagens numa pasta |
| custo de inferencia | cresce com o numero de marcas | constante em relacao as marcas |
| quem sabe o que e "Nike" | o modelo | **o banco** |

O detector deste projeto **nunca** e informado de nome de marca. A entidade
`Detection` nao tem campo `brand`, e essa ausencia e estrutural: e o que impede o
acoplamento de voltar por descuido.

---

## A pipeline

```
imagem
  │
  ├─ 1  PRE-FILTRO ............ descarta imagem sem estrutura
  │       Permissivo de proposito. O que se descarta aqui nunca mais volta.
  │
  ├─ 2  DETECTOR AGNOSTICO .... ONDE ha marca grafica
  │       Prompts de CONCEITO ("logo", "emblem"), nunca nomes de marca.
  │
  ├─ 3a CODIFICADOR ........... regiao -> vetor
  ├─ 3b BUSCA VETORIAL ........ vetor -> marcas candidatas
  │       E aqui que a marca aparece pela primeira vez.
  │
  ├─ 4  VERIFICACAO GEOMETRICA  "e o mesmo desenho?"
  │       Falha de forma diferente da camada 3 — por isso as duas convivem.
  │
  └─ 5  ROTEADOR .............. funde os sinais -> fila de destino
```

Cada camada e mais cara que a anterior e so ve o que a anterior deixou passar. A
verificacao geometrica, em especial, roda apenas nos melhores candidatos de cada
regiao — se rodasse em tudo, seria a camada dominante do custo.

### As filas de saida

| fila | significado | exige humano |
|---|---|:--:|
| `auto_aceite` | evidencia suficiente, entra no relatorio | nao |
| `revisao` | ha um palpite e nao ha confianca | **sim** |
| `confusao` | empate entre marcas do mesmo grupo declarado | **sim** |
| `orfao` | ha logo e o banco nao reconheceu — **a referencia que falta** | **sim** |
| `negativa` | logo de marca fora do portfolio. Acerto, nao rejeicao | nao |
| `auto_rejeicao` | nao ha evidencia de marca | nao |

**`orfao` e o mais valioso.** Ele nao e um erro: e o sistema dizendo *"o banco
tem esta marca e nao tem esta variacao dela"*. Promover essas regioes de volta
para o banco e o que faz o sistema melhorar sozinho com o uso.

---

## Instalacao

Todo comando roda via Poetry. Chamar `python` direto nao funciona — as
dependencias vivem no ambiente do Poetry.

```bash
poetry install
cp .env.example .env      # opcional — so precisa para modelo de acesso restrito
poetry run stratosphere ambiente
```

`ambiente` confere dependencias, token, aceleracao e banco **antes** de qualquer
download de peso de modelo. Rode primeiro: ele transforma meia hora de espera
seguida de erro num diagnostico de um segundo.

### Variaveis de ambiente

`.env` e ignorado pelo git; `.env.example` documenta cada variavel e e o arquivo
versionado. Nada disso e obrigatorio — sem `.env` o projeto roda com os defaults
de `config/settings.py` e baixa modelo de acesso livre.

| variavel | para que serve |
|---|---|
| `HF_TOKEN` | baixar peso com **acesso restrito** no Hugging Face. Gere em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) e aceite os termos na pagina do modelo — token valido sem termos aceitos tambem recebe 403 |
| `HF_HOME` | onde o hub guarda os pesos. Util quando o disco do usuario nao cabe varios modelos de visao |
| `STRATOSPHERE_DETECTOR_MODEL` · `STRATOSPHERE_ENCODER_MODEL` | trocar o peso que os adaptadores carregam |
| `STRATOSPHERE_DEVICE` · `STRATOSPHERE_PRECISION` | `cuda:0`/`cpu` e `float16`/`float32` |

Precedencia: **default do codigo < `.env` < variavel exportada no shell < flag da
CLI**. O `.env` nao sobrescreve variavel que ja esta no processo, e variavel
vazia nao sobrescreve nada.

O token nunca entra em `AppConfig` nem em log: ele so e carregado para o
ambiente, que e de onde o `huggingface_hub` o le. O `ambiente` reporta apenas
`definido` ou `ausente`.

---

## Uso

### 1. Montar o banco de referencia

```
referencias/
  nike/
    simbolo/     o simbolo isolado
    aplicado/    em tecido, em embalagem — ja deformado
    degradado/   pequeno, borrado, cortado
  cimed/
    wordmark/
    ...
```

A **marca** e o primeiro nivel — e o rotulo que o sistema devolve. A **variante**
e o segundo, e serve para tornar visivel no `ls` qual tipo de aplicacao esta
sub-representado.

```bash
poetry run stratosphere banco --referencias referencias/ --destino indice/
```

### 2. Auditar antes de usar

```bash
poetry run stratosphere auditar --banco indice/
```

Lista pares de marcas **diferentes** cujas referencias se parecem demais. **Cada
par e um falso positivo agendado**, ou um grupo de confusao ainda nao declarado.
Resolver isso agora custa minutos; descobrir depois custa um relatorio errado na
mao de um cliente.

Tambem avisa quais marcas tem poucas referencias — elas vao produzir orfaos em
vez de acertos, e saber disso de antemao muda a leitura de qualquer metrica.

### 3. Analisar

```bash
# uma imagem
poetry run stratosphere analisar --entrada foto.jpg --banco indice/

# uma pasta inteira, com o detalhe por regiao em json
poetry run stratosphere analisar --entrada fotos/ --banco indice/ --saida resultado.json

# sem GPU
poetry run stratosphere --cpu analisar --entrada fotos/

# conferir no olho: grava copias com as caixas e os rotulos desenhados
poetry run stratosphere analisar --entrada fotos/ --anotar runs/anotadas/
```

`--anotar` grava uma copia de cada imagem com a caixa de cada regiao e um rotulo
`marca · fila · pontuacao`. **A saida e separada por fila:**

```
runs/anotadas/
  auto_aceite/foto.jpg      so as caixas aceitas
  revisao/foto.jpg          so as que precisam de conferencia
  orfao/foto.jpg            so as referencias que faltam
  ...
```

Cada copia leva **apenas as caixas daquela fila**. Quem abre `orfao/` esta
decidindo promocao para o banco, e caixa de outra fila no meio so atrapalha. A
mesma imagem aparece em mais de uma pasta quando tem regioes de filas diferentes
— o que e a informacao certa: ela exige duas acoes distintas. A estrutura de
subpastas da entrada e espelhada dentro de cada fila, entao `nike/01.jpg` e
`itau/01.jpg` nao se sobrescrevem.

A cor tambem vem da fila: verde entra sozinho, ambar e azul pedem humano, roxo e
a referencia que falta, cinza e marca fora do portfolio, vermelho e descarte.

**Um logo, uma caixa.** Recortes aninhados da mesma marca sao colapsados antes do
relatorio — ver `NestedRegionResolver` em `.claude/doc/CONTRACTS.md`. Sem isso o
mesmo escudo aparece tres vezes, em tres filas diferentes, e infla a fila humana
com recortes internos de algo que ja foi aceito.

Por padrao as regioes em `auto_rejeicao` ficam de fora — do json e do desenho.
Para ver tudo que o detector achou, incluindo o descartado, acrescente `--tudo`.

O json de saida traz, por regiao: a caixa, a fila, a marca, a pontuacao, os
sinais que a produziram (similaridade, margem, inliers) e **os motivos em ordem
de aplicacao**. Com isso, "por que esta regiao caiu nesta fila?" tem resposta sem
reexecutar nada.

### 4. Reenquadrar as referencias (opcional, mas recomendado)

Referencia que inclui a placa em volta ensina a placa, nao a marca. Medido no par
`amazon x azul` recortado da mesma foto de backdrop: com a moldura, 0.903 de
similaridade entre **logos diferentes**; so o wordmark, 0.630.

```bash
poetry run python tools/tighten_references.py --origem referencias/ --destino referencias_cortadas/
poetry run stratosphere banco --referencias referencias_cortadas/ --destino indice/
```

O script usa o **proprio detector da pipeline** para achar a marca dentro de cada
referencia, entao o enquadramento da referencia fica igual ao da consulta. Quem
nao tem caixa utilizavel e copiado como esta — perder referencia e pior que
manter uma folgada.

### 5. Calibrar os limiares com dado proprio

Os defaults de `config/settings.py` sao ponto de partida, nao verdade. Com um
conjunto rotulado por pasta (`<marca>/arquivo.jpg`), da para medir onde eles
deveriam estar:

```bash
poetry run python tools/calibrate_thresholds.py --rotuladas rotuladas/ --banco indice/
```

Ele imprime a distribuicao de similaridade, consenso e inliers dos acertos contra
a dos erros, e sugere `min_similarity`, `max_similarity`, `confident_inliers` e
`orphan_max_similarity`. Tambem mede o consenso como regra sozinha — quanto dele
e preciso para o aceite nao errar.

### Todos os comandos

| comando | o que faz |
|---|---|
| `stratosphere ambiente` | confere dependencias, GPU e banco |
| `stratosphere banco --referencias <pasta> --destino <pasta>` | constroi o indice vetorial |
| `stratosphere auditar [--limite N] [--minimo N]` | marcas do banco parecidas demais |
| `stratosphere analisar --entrada <arq\|pasta> [--saida json] [--anotar pasta] [--limite N] [--tudo]` | roda a pipeline |
| `python tools/tighten_references.py --origem <pasta> --destino <pasta>` | reenquadra as referencias no logo |
| `python tools/calibrate_thresholds.py --rotuladas <pasta> --banco <pasta>` | mede onde os limiares deveriam estar |

Opcoes globais: `--banco <pasta>` (default `indice`), `--cpu`, `-v`.

**As globais vem antes do subcomando** — `stratosphere --banco outro/ analisar ...`,
nao `stratosphere analisar --banco outro/`. O argparse rejeita a segunda forma.

---

## Como adicionar uma marca

```bash
mkdir -p referencias/marca_nova/aplicado
# copie as imagens de referencia
poetry run stratosphere banco --referencias referencias/ --destino indice/
poetry run stratosphere auditar --banco indice/
```

Sem treino. Sem GPU alem da codificacao das imagens novas.

**O que funciona melhor:** crop real da superficie onde a marca costuma
aparecer. Logo de press kit e arte vetorial limpa; a regiao real e um recorte
pequeno, borrado e torto de um painel filmado de longe. Os dois vetores nao
ficam proximos. A logo oficial vale como ponto de partida para uma marca que
acabou de entrar e ainda nao tem material real — e so isso.

**O caminho natural:** rode em sombra por alguns dias, olhe a fila `orfao`, e
promova as regioes recorrentes para o banco. E o loop que faz o sistema melhorar
com o uso.

---

## Estrutura

Arquitetura em camadas, com a dependencia sempre apontando para dentro.

```
config/          dataclasses de configuracao, sem logica e sem ambiente
domain/          entidades, value objects e AS REGRAS DE DECISAO
  services/queue_router.py        <- o nucleo. Puro, sem I/O, sem GPU
application/     portas e casos de uso — orquestra, nao decide
infrastructure/  os adaptadores concretos e o container
entrypoints/cli/ a linha de comando
```

```
entrypoints ──> application ──> domain
      │              │
      └──────> infrastructure ──> (domain, application)
```

- `domain/` nao importa nada alem da biblioteca padrao e `numpy`.
- `application/` fala com as portas, **nunca** com `infrastructure/`.
- `infrastructure/container/container.py` e o **unico** lugar que instancia
  tecnologia concreta. Trocar de detector ou de codificador e uma mudanca la, e
  em lugar nenhum mais.
- `infrastructure/environment/env_settings.py` e o **unico** lugar que le
  variavel de ambiente. `config/settings.py` nao le ambiente e nao guarda
  credencial.

Documentacao de projeto em `.claude/doc/`: objetivos, arquitetura, contratos,
entidades, decisoes e o grafo de dependencias.

---

## Configuracao

Tudo em `config/settings.py`, em dataclasses.

**Dois valores merecem atencao especial**, porque falham em silencio quando
importados de outro contexto:

| parametro | por que nao transfere |
|---|---|
| `DetectorConfig.confidence_threshold` | esta na escala **daquele** detector. Um valor razoavel para um modelo treinado costuma zerar o recall de um detector de vocabulario aberto, cuja distribuicao de confianca e muito mais comprimida |
| `RoutingConfig.min_similarity` | esta na escala **daquele** codificador. Vetores de regioes nao relacionadas raramente ficam proximos de zero — mapear a partir de zero faz parede lisa parecer evidencia |

Recalibre com dado proprio antes de operar. Os defaults sao ponto de partida.

### Uma armadilha do scorer

Os pesos de `RoutingConfig` somam 1. Quando a verificacao geometrica **nao
opina** — logo chapado ou pequeno demais para casar pontos — os pesos restantes
sao **renormalizados**. Sem isso, a regiao seria punida por uma evidencia que
nunca teve chance de existir, e o teto da pontuacao cairia abaixo do limiar de
aceite para uma classe inteira de casos.

---

## Qualidade

```bash
poetry run ruff check . --fix
poetry run ruff format .
poetry run mypy . --ignore-missing-imports
```

**Esta versao nao tem suite de testes**, por decisao de escopo registrada em
`.claude/doc/DECISIONS.md`. A consequencia esta anotada la: nao ha rede de
seguranca contra regressao.

O dominio foi escrito puro e sem I/O justamente para que a suite possa ser
acrescentada depois sem refatoracao. O ponto de partida obvio e
`tests/unit/test_queue_router.py`, cobrindo a ordem das regras descrita em
`.claude/doc/CONTRACTS.md` — e o arquivo de maior risco e menor custo de teste.

---

## Licenca

MIT. Ver [LICENSE](LICENSE).
