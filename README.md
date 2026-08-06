# Stratosphere

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

Deteccao de marcas em imagens, em escala, com marcas entrando continuamente no
portfolio.

A ideia central cabe numa frase:

> **Separar *onde tem logo* de *qual logo e*.**
> Um detector agnostico de marca acha as regioes candidatas. Uma busca vetorial
> num banco de referencias diz de quem sao. **Marca nova e uma pasta de imagens
> a mais, nao um ciclo de retreino.**

Analogia: detector de rosto - que acha qualquer rosto, inclusive de quem nunca
viu - somado a reconhecimento facial, que compara com um banco cadastrado.

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
  │       Modelo com supervisao de TEXTO. Ver "Por que nao um encoder
  │       auto-supervisionado" abaixo - a escolha errada aqui custou
  │       27 pontos de recall.
  ├─ 3b BUSCA VETORIAL ........ vetor -> marcas candidatas
  │       E aqui que a marca aparece pela primeira vez.
  │
  ├─ 4  VERIFICACAO GEOMETRICA  "e o mesmo desenho?"
  │       Falha de forma diferente da camada 3 - por isso as duas convivem.
  │       So roda no que a camada 3 NAO resolveu: 0.7s por imagem
  │       contra 3.0s verificando tudo.
  │
  └─ 5  ROTEADOR .............. funde os sinais -> fila de destino
```

Cada camada e mais cara que a anterior e so ve o que a anterior deixou passar. A
verificacao geometrica, em especial, roda apenas nos melhores candidatos de cada
regiao - se rodasse em tudo, seria a camada dominante do custo.

### As filas de saida

| fila | significado | exige humano |
|---|---|:--:|
| `auto_aceite` | evidencia suficiente, entra no relatorio | nao |
| `revisao` | ha um palpite e nao ha confianca | **sim** |
| `confusao` | empate entre marcas do mesmo grupo declarado | **sim** |
| `orfao` | ha logo e o banco nao reconheceu - **a referencia que falta** | **sim** |
| `negativa` | logo de marca fora do portfolio. Acerto, nao rejeicao | nao |
| `auto_rejeicao` | nao ha evidencia de marca | nao |

**`orfao` e o mais valioso.** Ele nao e um erro: e o sistema dizendo *"o banco
tem esta marca e nao tem esta variacao dela"*. Promover essas regioes de volta
para o banco e o que faz o sistema melhorar sozinho com o uso.

---

## Modelos: o que foi medido

Todo modelo aqui foi comparado no mesmo conjunto e com a mesma metrica. Numero de
conjunto diferente nao compara - foi assim que o SIFT quase pareceu melhor que o
LightGlue, e que o SigLIP2 quase pareceu pior que o DINOv2.

### Codificador: a escolha que mais importa

**O teste nao precisa de rotulo.** Varias referencias vem da MESMA foto de
backdrop, recortadas em marcas diferentes, e o nome do arquivo carrega o hash da
origem. Isso da dois conjuntos de graca:

| conjunto | esperado |
|---|---|
| **positivo** - mesma marca, fotos de origem diferentes | proximos |
| **negativo** - MESMA foto de origem, marcas diferentes | distantes |

AUC mede a ORDEM, nao o valor absoluto. 0.5 e moeda.

Medido em 588 referencias, 1107 pares positivos, 270 negativos:

| codificador | melhor AUC | agregacao | licenca | ms/img |
|---|---|---|---|---|
| **siglip2-so400m-patch16-256** | **0.898** | `centro` | apache-2.0 | 13 |
| siglip2-large-patch16-256 | 0.883 | `centro` | apache-2.0 | 10 |
| metaclip-2-worldwide-giant | 0.882 | `media` | **cc-by-nc** | 48 |
| **siglip2-base-patch16-224** (em uso) | 0.801 | `pooler` | apache-2.0 | - |
| dinov3-vitl16 | 0.769 | `centro` | **other**, restrito | - |
| clip-vit-large-patch14 | 0.713 | `pooler` | apache-2.0 | - |
| dinov3-vitb16 | 0.694 | `centro` | **other**, restrito | - |
| onevision-encoder-large | 0.557 | `centro` | apache-2.0 | 61 |
| dinov2-base (o original) | 0.515 | `centro` | apache-2.0 | - |

**0.515 e moeda, e era o que rodava em producao.** O DINOv2 nao distinguia
"mesma marca" de "mesma foto": dois logos DIFERENTES do mesmo painel pontuavam
0.676, acima de duas fotos da MESMA marca (0.671). Ele e auto-supervisionado e
aprendeu que o que faz duas imagens parecidas e a **superficie** - painel, luz,
moldura. Isso explicava sozinho `amazon x azul` em 0.903, `itau` lido como
`sadia`, os 238 pares confundiveis do banco (hoje 32) e o consenso desabando em
backdrop.

Tres coisas que a tabela ensina alem do vencedor:

- **Supervisao de texto ganha de auto-supervisao.** Legenda fala de marca;
  auto-supervisao aprende textura. As duas geracoes da familia DINO ficam abaixo
  de qualquer CLIP.
- **Escala ainda entrega.** O `so400m` bate o `base` em 10 pontos. Nao havia
  plato.
- **O `onevision-encoder` colapsa.** Todos os pares dao 0.999 - positivo,
  negativo, tanto faz. E encoder para alimentar LLM, e a saida agregada nunca foi
  treinada para ter geometria de cosseno.

**Trocar o codificador invalida o indice E todos os limiares de similaridade.**
A assinatura protege o indice - a carga recusa a combinacao errada. Os limiares
nao tem essa protecao e falham em **silencio**: `min_similarity`,
`max_similarity`, `redundancy_similarity`, `alert_similarity`,
`entry_similarity` e `orphan_max_similarity` vivem todos na escala do
codificador. O de deduplicacao passou despercebido e cortou o banco de 753 para
524 referencias antes de alguem notar.

### Verificacao geometrica

Medido em 60 pares certos contra 60 errados, a 448px:

| verificador | AUC | mudo em par certo | recall com precisao 100% |
|---|---|---|---|
| **DISK+LightGlue** | 0.796 | **3%** | **63%** |
| LoFTR | 0.793 | 5% | 60% |
| SIFT (o original) | 0.711 | **45%** | **0%** |
| contorno por momentos de Hu | 0.601 | - | - |

A coluna que decide e a ultima. **No SIFT ela e zero**: nao existe limiar em que
ele confirme alguma coisa sem deixar passar par errado. Ele nunca teve como tirar
regiao da fila humana; so tinha como concordar com quem ja estava decidido.

Contorno parecia a resposta obvia para forma solida sem textura - swoosh,
wordmark - e nao e. A binarizacao captura o contorno da placa em volta, nao do
logo. SuperPoint ficou de fora porque os pesos sao non-commercial.

### Juiz visual (`--vlm`)

Medido em 40 pares certos contra 40 errados:

| juiz | AUC | confirma com precisao 100% | s/par | VRAM |
|---|---|---|---|---|
| **Qwen3.5-4B** | **0.995** | **38 de 40** | 0.82 | ~9.3 GB |
| Qwen2-VL-2B (em uso) | 0.700 | 7 de 40 | 0.19 | ~4.4 GB |

O 2B responde "sim" para quase tudo - 0.842 em par certo contra 0.789 em par
errado, quase sem separacao. O Qwen3.5 responde **0.950 no certo e 0.075 no
errado**: ele discrimina em vez de concordar.

**Atencao ao ler logit em modelo com modo de raciocinio.** O template do
Qwen3.5 abre o bloco de pensamento sozinho, e o primeiro token vira o comeco do
raciocinio, nao a resposta. Medindo no lugar errado, a AUC dele cai para 0.664 -
numero plausivel que teria condenado o modelo. Ver `QwenJudge._chat_prompt`.

---

## Instalacao

Todo comando roda via Poetry. Chamar `python` direto nao funciona - as
dependencias vivem no ambiente do Poetry.

```bash
poetry install
cp .env.example .env      # opcional - so precisa para modelo de acesso restrito
poetry run stratosphere ambiente
```

`ambiente` confere dependencias, token, aceleracao e banco **antes** de qualquer
download de peso de modelo. Rode primeiro: ele transforma meia hora de espera
seguida de erro num diagnostico de um segundo.

### Variaveis de ambiente

`.env` e ignorado pelo git; `.env.example` documenta cada variavel e e o arquivo
versionado. Nada disso e obrigatorio - sem `.env` o projeto roda com os defaults
de `config/settings.py` e baixa modelo de acesso livre.

| variavel | para que serve |
|---|---|
| `HF_TOKEN` | baixar peso com **acesso restrito** no Hugging Face. Gere em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) e aceite os termos na pagina do modelo - token valido sem termos aceitos tambem recebe 403 |
| `HF_HOME` | onde o hub guarda os pesos. Util quando o disco do usuario nao cabe varios modelos de visao |
| `HF_HUB_OFFLINE` | `=1` para nao sair a rede. Os pesos ja ficam em disco; o que se repete a cada execucao e so uma revalidacao de metadado. `ambiente` mostra o que ja esta em cache |
| `STRATOSPHERE_DETECTOR_MODEL` | trocar o peso do detector |
| `STRATOSPHERE_ENCODER_MODEL` · `_BACKEND` · `_AGGREGATION` | trocar o codificador. **As tres andam juntas** - modelo de outra familia precisa de outro adaptador, e a agregacao errada produz espaco vetorial ruim em silencio |
| `STRATOSPHERE_JUDGE_MODEL` | trocar o VLM do `--vlm`. **Invalida os cortes calibrados do juiz** - refaca com `tools/calibrate_judge.py` |
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
    aplicado/    em tecido, em embalagem - ja deformado
    degradado/   pequeno, borrado, cortado
  cimed/
    wordmark/
    ...
```

A **marca** e o primeiro nivel - e o rotulo que o sistema devolve. A **variante**
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

Tambem avisa quais marcas tem poucas referencias - elas vao produzir orfaos em
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
- o que e a informacao certa: ela exige duas acoes distintas. A estrutura de
subpastas da entrada e espelhada dentro de cada fila, entao `nike/01.jpg` e
`itau/01.jpg` nao se sobrescrevem.

A cor tambem vem da fila: verde entra sozinho, ambar e azul pedem humano, roxo e
a referencia que falta, cinza e marca fora do portfolio, vermelho e descarte.

**Um logo, uma caixa.** Recortes aninhados sao colapsados antes do
relatorio - ver `NestedRegionResolver` em `.claude/doc/CONTRACTS.md`. Sem isso o
mesmo escudo aparece tres vezes, em tres filas diferentes, e infla a fila humana
com recortes internos de algo que ja foi aceito.

Por padrao as regioes em `auto_rejeicao` ficam de fora - do json e do desenho.
Para ver tudo que o detector achou, incluindo o descartado, acrescente `--tudo`.

### 4. Segunda opiniao visual (opcional)

```bash
poetry run stratosphere --vlm analisar --entrada fotos/ --anotar runs/anotadas/
```

`--vlm` liga um juiz visual sobre **apenas as regioes que cairam em revisao**.
Ele recebe duas imagens - o recorte e a referencia do banco - e responde se sao a
mesma marca. Tres saidas:

| veredito | destino |
|---|---|
| confirma | `auto_aceite` |
| nega | `auto_rejeicao` |
| se abstem, ou juiz indisponivel | `revisao`, exatamente onde ja estava |

O modelo e **aberto e roda local** (`Qwen/Qwen2-VL-2B-Instruct`, apache-2.0,
~4.4 GB), na mesma GPU do resto da pipeline. Nao ha API, nao ha custo por
chamada, e nenhum recorte de imagem sai da maquina. A primeira execucao baixa os
pesos; depois disso o cache resolve.

**A pergunta e de comparacao, nunca "que marca e essa?".** Metade deste
portfolio e marca regional que nenhum modelo conhece de treino; perguntar o nome
funciona onde o sistema ja acerta e inventa resposta onde ele precisa de ajuda.
Comparar duas imagens vale igual para marca famosa e desconhecida - e mantem o
banco como quem diz qual marca e.

**O juiz nao le a propria resposta em texto.** A pergunta e montada para que a
proxima palavra so possa ser `Yes` ou `No`, e o que se usa e a probabilidade que
o modelo deu a cada uma. Modelo de 2B nao segue formato de saida com
confiabilidade, e a confianca que ele declara sobre si mesmo nao vale nada - a
probabilidade vale, e sai numa passada so, sem laco de geracao.

**Os dois cortes de decisao sao medidos, nao arbitrados.** Perguntado "sao a
mesma marca?", o modelo concorda com quase tudo: medido, acerto e erro caem os
dois na casa dos 0.6. O que tem sinal e a ordem, entao os cortes vem de
percentil das duas distribuicoes:

```bash
poetry run python tools/calibrate_judge.py --rotuladas dataset/por_marca --banco indice
```

Trocar o modelo do juiz invalida os dois cortes - cada modelo tem seu vies.

Juiz indisponivel ou em duvida **nao decide**: a regiao fica na fila humana.

O json de saida traz, por regiao: a caixa, a fila, a marca, a pontuacao, os
sinais que a produziram (similaridade, margem, inliers) e **os motivos em ordem
de aplicacao**. Com isso, "por que esta regiao caiu nesta fila?" tem resposta sem
reexecutar nada.

### 5. Reenquadrar as referencias (opcional, mas recomendado)

Referencia que inclui a placa em volta ensina a placa, nao a marca. Medido no par
`amazon x azul` recortado da mesma foto de backdrop: com a moldura, 0.903 de
similaridade entre **logos diferentes**; so o wordmark, 0.630.

```bash
poetry run python tools/tighten_references.py --origem referencias/ --destino referencias_cortadas/
poetry run stratosphere banco --referencias referencias_cortadas/ --destino indice/
```

O script usa o **proprio detector da pipeline** para achar a marca dentro de cada
referencia, entao o enquadramento da referencia fica igual ao da consulta. Quem
nao tem caixa utilizavel e copiado como esta - perder referencia e pior que
manter uma folgada.

### 6. Calibrar os limiares com dado proprio

Os defaults de `config/settings.py` sao ponto de partida, nao verdade. Com um
conjunto rotulado por pasta (`<marca>/arquivo.jpg`), da para medir onde eles
deveriam estar:

```bash
poetry run python tools/calibrate_thresholds.py --rotuladas rotuladas/ --banco indice/
```

Ele imprime a distribuicao de similaridade, consenso e inliers dos acertos contra
a dos erros, e sugere `min_similarity`, `max_similarity`, `confident_inliers` e
`orphan_max_similarity`. Tambem mede o consenso como regra sozinha - quanto dele
e preciso para o aceite nao errar.

### Todos os comandos

| comando | o que faz |
|---|---|
| `stratosphere ambiente` | confere dependencias, GPU, cache do hub e banco |
| `stratosphere banco --referencias <pasta> --destino <pasta>` | constroi o indice vetorial |
| `stratosphere auditar [--limite N] [--minimo N]` | marcas do banco parecidas demais |
| `stratosphere analisar --entrada <arq\|pasta> [--saida json] [--anotar pasta] [--limite N] [--tudo]` | roda a pipeline |
| `stratosphere --vlm analisar ...` | idem, com segunda opiniao visual na fila de revisao |
| `python tools/tighten_references.py --origem <pasta> --destino <pasta>` | reenquadra as referencias no logo |
| `python tools/mine_references.py --entrada <pasta> --destino <pasta> [--marcas a,b] [--excluir arq...]` | garimpa candidatas a novas referencias |
| `python tools/dump_signals.py --rotuladas <pasta> --saida sinais.jsonl` | grava os sinais crus, para iterar sem GPU |
| `python tools/sweep_routing.py --sinais sinais.jsonl [--varredura]` | explora limiares em cima do dump |
| `python tools/calibrate_thresholds.py --rotuladas <pasta> --banco <pasta>` | mede onde os limiares deveriam estar |
| `python tools/calibrate_judge.py --rotuladas <pasta> --banco <pasta> [--limite N]` | mede o juiz visual e sugere os dois cortes dele |

**Opcoes globais vem ANTES do subcomando:**

| flag | efeito |
|---|---|
| `--banco <pasta>` | qual indice usar. Default `indice` |
| `--cpu` | forca CPU. Vence `STRATOSPHERE_DEVICE` |
| `--vlm` | liga o juiz visual sobre a fila de revisao |
| `-v` | log em DEBUG, incluindo o trafego de rede |

```bash
poetry run stratosphere --banco outro/ analisar --entrada fotos/   # certo
poetry run stratosphere analisar --banco outro/ --entrada fotos/   # argparse recusa
```

### Rodar uma pasta inteira

```bash
poetry run stratosphere analisar   --entrada /caminho/das/fotos   --saida runs/resultado.json   --anotar runs/anotadas
```

Percorre a pasta **recursivamente**, em ordem estavel. A estrutura de subpastas da
entrada e espelhada dentro de cada fila da saida, entao `nike/01.jpg` e
`itau/01.jpg` nao se sobrescrevem.

```
runs/anotadas/
  auto_aceite/...    so as caixas aceitas
  revisao/...        so as que precisam de conferencia
  orfao/...          so as referencias que faltam
```

**Cada copia leva apenas as caixas daquela fila.** Quem abre `orfao/` esta
decidindo promocao para o banco, e caixa de outra fila no meio so atrapalha. A
mesma imagem aparece em mais de uma pasta quando tem regioes de filas diferentes
- o que e a informacao certa: ela exige duas acoes distintas.

Variacoes uteis:

```bash
# so as N primeiras, para calibrar antes de rodar tudo
poetry run stratosphere analisar --entrada fotos/ --limite 40 --anotar runs/amostra

# incluindo o que foi descartado, para investigar o que o detector achou
poetry run stratosphere analisar --entrada fotos/ --tudo --saida runs/tudo.json

# sem GPU
poetry run stratosphere --cpu analisar --entrada fotos/

# com o juiz visual sobre a fila de revisao
poetry run stratosphere --vlm analisar --entrada fotos/ --anotar runs/com_juiz
```

**Custo:** ~0,7 s por imagem numa GPU de 12 GB, com a geometria rodando so onde
pode mudar o destino (`GeometryConfig.only_when_uncertain`). Verificando tudo sao
3,0 s pelo mesmo resultado. O `--vlm` acrescenta uma passada de VLM por regiao em
revisao, com teto em `JudgeConfig.max_regions`.

**O json de saida** traz, por regiao: a caixa, a fila, a marca, a pontuacao, os
sinais que a produziram (similaridade, margem, inliers) e **os motivos em ordem
de aplicacao**. Com isso, "por que esta regiao caiu nesta fila?" tem resposta sem
reexecutar nada. Regioes em `auto_rejeicao` ficam de fora por padrao - use
`--tudo` para incluir.

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
acabou de entrar e ainda nao tem material real - e so isso.

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
application/     portas e casos de uso - orquestra, nao decide
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
| `RoutingConfig.min_similarity` | esta na escala **daquele** codificador. Vetores de regioes nao relacionadas raramente ficam proximos de zero - mapear a partir de zero faz parede lisa parecer evidencia |

Recalibre com dado proprio antes de operar. Os defaults sao ponto de partida.

### Uma armadilha do scorer

Os pesos de `RoutingConfig` somam 1. Quando a verificacao geometrica **nao
opina** - logo chapado ou pequeno demais para casar pontos - os pesos restantes
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
`.claude/doc/CONTRACTS.md` - e o arquivo de maior risco e menor custo de teste.

---

## Licenca

**Apache License 2.0.** Ver [LICENSE](LICENSE) e [NOTICE](NOTICE).

Uso, modificacao e redistribuicao sao permitidos, inclusive comercialmente,
desde que a atribuicao de autoria e o aviso NOTICE sejam mantidos. O software e
fornecido no estado em que se encontra, **sem garantia**, e os autores **nao
respondem por danos nem por uso indevido**.

Dois avisos que este projeto em particular exige:

- **Marcas de terceiros.** O software detecta e compara marcas graficas. Os
  logotipos e nomes comerciais processados pertencem aos seus titulares e **nao
  sao licenciados por este software**. Quem opera responde por ter direito de uso
  sobre as imagens que processa e sobre as referencias que cadastra.
- **Modelos de terceiros.** Os pesos baixados em tempo de execucao tem licencas
  proprias, independentes desta. Os defaults do projeto sao apache-2.0; ver a
  tabela em "Modelos: o que foi medido" antes de trocar qualquer um.

---

## Como fazer um bom banco de referencia

Tudo aqui saiu de medicao neste projeto, e cada item cita o caso que o produziu.
O banco e a peca que mais determina o resultado - mais que o detector, mais que
os limiares. **Marca que o banco cobre mal nao e recuperavel por regra nenhuma.**

### 1. Cobertura de variante vale mais que quantidade

O que decide nao e quantas referencias a marca tem, e sim se ela tem a
**aplicacao** que aparece na foto. Medido numa coletiva de imprensa, com a marca
correta em primeiro lugar na busca em todos os casos:

| marca | refs de backdrop | resultado |
|---|---|---|
| volkswagen | 11 | aceita 3x |
| ifood | 11 | aceita 2x |
| amazon | 9 | aceita |
| sadia | 3 | revisao |
| **nike** | **2** | **rejeitada** - e a nike tinha 35 referencias no total |
| **vivo** | **1** | **rejeitada** - de 38 no total |

A nike tinha 27 referencias de uniforme. Um swoosh gigante num painel procurava
vizinho entre elas e nao achava nenhuma parecida: o consenso desabava para 2 de
25 e a regiao morria. **Nao era duvida sobre a marca; era aritmetica do banco.**

Na pratica: para cada marca, tenha referencia de **backdrop, uniforme, digital e
o logo oficial**. Quatro boas de tipos diferentes valem mais que trinta do mesmo
tipo.

### 2. Enquadre no logo, nao na placa

Medido no par `amazon x azul` recortado da MESMA foto de backdrop:

```
recorte com placa, moldura e fundo ... 0.903   <- duas marcas DIFERENTES
tirando o fundo e a moldura .......... 0.801
so o wordmark ........................ 0.630
```

Mas ha um piso: cortando a 40% a similaridade **sobe** de novo para 0.699 - o
corte comeu letras, e fragmento mutilado volta a parecer com outro fragmento
mutilado. Por isso o `prepare_reference` usa a caixa do detector com folga
pequena, e nao uma fracao fixa.

### 3. Uma marca por referencia

Backdrop tem varios patrocinadores lado a lado. Referencia que pega dois logos
ensina os dois juntos, e a busca passa a casar "painel com dois logos" em vez de
qualquer um deles.

### 4. PNG com transparencia precisa ser composto, nao achatado

Alpha descartado vira **retangulo preto**. Quatro referencias do banco original
eram exatamente isso, e uma delas colocava `cimed x nike` a **0.951** - o pior
par do banco inteiro, entre duas marcas sem nenhuma semelhanca. O
`prepare_reference` e o carregamento da pipeline ja compoem sobre o fundo neutro.

### 5. Nao infle com quase-copias

Referencia redundante nao acrescenta cobertura, ocupa o topo da busca com copia e
**desloca vizinho util do top-k** - que e justamente o que alimenta o consenso.
`prepare_reference` recusa acima de `SearchConfig.redundancy_similarity`.

### 6. Desconfie de marca com muitas referencias

A `cbf` tem 65 referencias, mais que qualquer outra. Resultado: quase todo top-25
era cbf, e a **bandeira do Brasil** era aceita como cbf com pontuacao 0.44 - o
escudo tem um circulo azul com estrelas sobre verde e amarelo. Marca
super-representada domina a vizinhanca e vaza para onde nao deve.

### 7. Audite depois de cada mudanca

```bash
poetry run stratosphere auditar
```

Cada par de marcas diferentes acima do limiar de alerta e um falso positivo
agendado. Referencia nova que cria par assim custa mais do que entrega.

### 8. Promova o que a operacao encontrar

A fila `orfao` e o sistema dizendo *"tenho esta marca e nao tenho esta variacao
dela"*. Promover essas regioes e o que faz o banco melhorar com o uso - a `cbf`
ja tem 23 referencias vindas dai.

```bash
poetry run python tools/mine_references.py --entrada fotos/ --destino runs/candidatas     --marcas nike,vivo --excluir imagem_de_avaliacao.jpg
```

**Nunca garimpe da imagem que voce usa para medir.** Promover a partir dela
fabrica metrica: o sistema passa a reconhecer um recorte de si mesmo. Por isso o
`--excluir` existe e o log registra o que foi excluido.

E confira uma a uma. Medido: das 20 candidatas com similaridade acima de 0.82, so
**11 eram a marca certa** - `itau` lido como `sadia`, `Bolsonaro` lido como
`vivo`, `Caze` lido como `uber`.

### Preparando uma imagem para o banco

```bash
poetry run python tools/prepare_reference.py     --entrada logo_recortado.png --marca nike --variante backdrop
```

A ferramenta compoe a transparencia, reenquadra no logo com o detector, amplia o
que for pequeno, e **so entao** compara com o banco. Ela responde quatro
perguntas antes de gravar:

| verdicto | significado |
|---|---|
| `REDUNDANTE` | quase igual a uma que ja esta la |
| `PERIGOSA` | parecida demais com OUTRA marca |
| `ROTULO SUSPEITO` | o banco prefere outra marca a que voce declarou |
| `pequena demais` | menos de 48px de menor lado |

Ela **nao** escreve no banco. Grava a copia preparada em
`<destino>/<marca>/<variante>/`, e reconstruir o indice continua sendo uma
decisao sua:

```bash
poetry run python tools/prepare_reference.py --entrada logos/ --marca vivo --so-relatorio
poetry run stratosphere banco --referencias referencias_v3 --destino indice
poetry run stratosphere auditar
```

