"""Limiares de dominio com o significado de cada valor.

Estes sao **defaults de partida**, nao verdades. Todo limiar aqui deve ser
recalibrado com dado proprio antes de operar: a escala de similaridade depende
do codificador, e a escala de confianca depende do detector. Um valor importado
de outro projeto costuma estar em outra escala e falha em silencio.

O lugar de sobrescrever e `config/settings.py`, nunca este arquivo.
"""

# Similaridade abaixo da qual a evidencia do banco conta como nula.
# MEDIDO em 84 imagens rotuladas (158 regioes): e o percentil 90 dos ERROS. O
# valor anterior, 0.68, ficava abaixo disso - o sistema pagava credito de
# similaridade para ruido.
DEFAULT_MIN_SIMILARITY: float = 0.86

# Similaridade a partir da qual a evidencia do banco conta como maxima.
# MEDIDO: percentil 75 dos acertos. Os acertos tem mediana 0.930 e p90 0.970;
# os erros, mediana 0.539 e p90 0.755 - a similaridade separa bem, e a escala
# agora comeca onde o ruido acaba.
DEFAULT_MAX_SIMILARITY: float = 0.99

# Margem (topo menos rival de outra marca) que ja caracteriza escolha confiante.
DEFAULT_CONFIDENT_MARGIN: float = 0.15

# Inliers de RANSAC que ja caracterizam geometria confirmada.
# ATENCAO: este valor so vale para o matcher em uso. As escalas nao sao
# comparaveis - no mesmo conjunto de 60 pares certos, SIFT devolve mediana 2 e
# DISK+LightGlue devolve 86.
# MEDIDO para o LightGlue, em pares de referencia com rotulo limpo: par de
# marcas DIFERENTES chega no maximo a 52 inliers, e par da mesma marca tem
# mediana 86. Em 90 a evidencia esta acima de todo par errado observado.
# O valor anterior (17) era do SIFT e, na escala nova, era alcancado por ruido.
DEFAULT_CONFIDENT_INLIERS: float = 90.0

# Pontuacao a partir da qual a regiao e aceita sem humano.
DEFAULT_ACCEPT: float = 0.80

# Pontuacao abaixo da qual a regiao e descartada sem humano.
DEFAULT_REJECT: float = 0.60

# Margem abaixo da qual duas marcas do mesmo grupo sao consideradas empatadas.
DEFAULT_TIE_MARGIN: float = 0.06

# Inliers minimos para o orfao geometrico disparar. Abaixo disso o casamento de
# pontos encontra coerencia em ruido, e a fila de orfaos vira lixo.
DEFAULT_ORPHAN_MIN_INLIERS: int = 20

# Similaridade abaixo da qual, havendo confirmacao geometrica, a regiao e orfa:
# o banco tem a marca e nao tem esta variacao dela.
# MEDIDO: percentil 10 dos acertos. Em 0.88 a regra disparava no MEIO da faixa
# normal - uma regiao com 27 inliers e similaridade 0.869 virava orfa por onze
# milesimos, sem faltar variacao nenhuma no banco.
DEFAULT_ORPHAN_MAX_SIMILARITY: float = 0.90

# Densidade de borda abaixo da qual a imagem nao tem estrutura suficiente para
# valer uma passada de detector. Permissivo de proposito - o que se descarta
# aqui nunca mais volta, e o recall do detector e o teto do sistema.
DEFAULT_MIN_EDGE_DENSITY: float = 0.004

# Fracao da menor caixa coberta pela maior a partir da qual dois recortes sao o
# MESMO logo, e nao dois logos vizinhos. Nao e IoU: IoU e cego para aninhamento,
# e recorte dentro de recorte sai com IoU baixo mesmo estando 100% contido.
DEFAULT_NESTED_CONTAINMENT: float = 0.80


# Consenso a partir do qual a regiao e aceita sem humano, independentemente da
# pontuacao. MEDIDO em 84 imagens rotuladas: acima de 0.80 nao houve um unico
# erro em 15 regioes. E o sinal que resgata logo chapado - swoosh, wordmark -
# cuja geometria nao tem canto para dar ponto e cuja similaridade absoluta cai
# na faixa do ruido.
DEFAULT_CONSENSUS_ACCEPT: float = 0.80

# Concordantes ABSOLUTOS exigidos junto com o consenso. Sem isto, marca com 2
# referencias no banco alcanca consenso 1.0 trivialmente - o teto do consenso e
# `min(top-k, referencias da marca)`, e unanimidade de 2 nao vale o mesmo que
# unanimidade de 25.
DEFAULT_CONSENSUS_MIN_AGREEING: int = 8


# Probabilidade de "sim" do juiz visual a partir da qual ele CONFIRMA a marca,
# e ate a qual ele NEGA. Entre os dois ele se abstem e a regiao vai para o humano.
#
# MEDIDO no Qwen3.5-4B, 60 pares certos contra 60 errados:
#
#     certos   min=0.060  p25=0.834  mediana=0.936  max=0.996
#     errados  min=0.004  p25=0.039  mediana=0.112  max=0.666
#
# AS DUAS MARGENS SAO ASSIMETRICAS DE PROPOSITO, porque os dois erros custam
# coisas diferentes.
#
# CONFIRMAR: o maior par errado chegou a 0.666. Em 0.70 ja sao 54 de 60 certos
# com zero erros; 0.75 mantem 100% de precisao com folga acima do teto
# observado. Confirmar errado poe marca errada no relatorio do cliente, e 60
# pares nao garantem que 0.666 seja o teto real.
#
# NEGAR: o menor par certo foi 0.060. Em 0.05 caem 19 dos 60 errados sem perder
# nenhum certo, mas a margem fica em 0.01. Em 0.04 sobra margem de verdade.
# Negar errado APAGA DETECCAO REAL EM SILENCIO - o pior erro do sistema - e o
# ganho e pequeno, porque a fila humana ja esta em 1% das regioes.
#
# Trocar o modelo do juiz invalida os dois valores. O anterior, Qwen2-VL-2B,
# vivia em 0.92/0.60: ele respondia "sim" para quase tudo e a faixa util ficava
# no topo. O Qwen3.5 discrimina, e a faixa util se abriu.
DEFAULT_JUDGE_CONFIRM_ABOVE: float = 0.75
DEFAULT_JUDGE_DENY_BELOW: float = 0.04


# Inliers a partir dos quais a geometria aceita a regiao sozinha, sem passar
# pelo limiar de pontuacao. Regra simetrica a do consenso: aquela resgata logo
# chapado, esta resgata logo de desenho rico em marca com poucas referencias no
# banco, onde o consenso e baixo por aritmetica e nao por duvida.
# MEDIDO com rotulo limpo em 60 pares de marcas diferentes: o maximo alcancado
# foi 52 inliers, contra mediana 86 nos pares da mesma marca. Acima de 90 nao
# houve par errado.
# Depende do matcher - trocar o verificador invalida este valor.
DEFAULT_GEOMETRY_ACCEPT_INLIERS: float = 90.0


# Inliers abaixo dos quais a contagem e ruido e o veredito geometrico conta como
# SILENCIO - peso redistribuido - em vez de evidencia fraca.
# MEDIDO: par de marcas diferentes tem mediana 8 inliers e chega a 52; par da
# mesma marca tem mediana 86. Nessa faixa de baixo as duas distribuicoes se
# sobrepoem, entao a contagem nao diz nada - e tratar "nao diz nada" como "diz
# pouco a favor" derruba a nota de quem nunca teve chance.
# Depende do matcher - trocar o verificador invalida este valor.
DEFAULT_INFORMATIVE_INLIERS: int = 20


# Similaridade propria minima para uma regiao em revisao ser promovida a aceite
# porque a mesma imagem ja confirmou aquela marca em outra caixa.
# MEDIDO: regiao errada tem similaridade mediana 0.539 e p90 de 0.755. Em 0.85 a
# promocao fica bem longe dessa distribuicao - a corroboracao decide entre "e
# esta marca" e "nao sei", nunca entre "e logo" e "e parede".
DEFAULT_CORROBORATION_MIN_SIMILARITY: float = 0.85


# Contencao a partir da qual duas caixas da MESMA marca sao o mesmo logo.
# Mais folgado que o limiar entre marcas diferentes por assimetria de risco:
# colapsar duas caixas da mesma marca custa, no pior caso, uma ocorrencia a
# menos de uma marca que a imagem ja reporta; colapsar duas de marcas diferentes
# apaga uma marca inteira do relatorio.
# CASO MEDIDO: o escudo da CBF gerava duas caixas - o escudo inteiro e a parte de
# cima dele - com contencao 0.76, e as duas passavam pelo limiar de 0.80.
DEFAULT_NESTED_SAME_BRAND_CONTAINMENT: float = 0.60


# --------------------------------------------------------------------------
# ESCALA DO SIGLIP2 - os valores acima foram REMEDIDOS na troca de codificador
# --------------------------------------------------------------------------
# `min_similarity` e `max_similarity` viviam na escala do DINOv2 (0.755 / 0.959)
# e nao sao portateis. MEDIDO na imagem de referencia, com as 15 marcacoes
# conferidas a olho: regiao certa tem similaridade 0.922 a 0.994; falso positivo
# vai ate 0.948. O piso subiu para 0.86 porque abaixo disso, na escala nova, nao
# ha acerto nenhum - manter 0.755 fazia parede e cadeira entrarem na conta.
