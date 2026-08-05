"""Limiares de dominio com o significado de cada valor.

Estes sao **defaults de partida**, nao verdades. Todo limiar aqui deve ser
recalibrado com dado proprio antes de operar: a escala de similaridade depende
do codificador, e a escala de confianca depende do detector. Um valor importado
de outro projeto costuma estar em outra escala e falha em silencio.

O lugar de sobrescrever e `config/settings.py`, nunca este arquivo.
"""

# Similaridade abaixo da qual a evidencia do banco conta como nula.
# MEDIDO em 84 imagens rotuladas (158 regioes): e o percentil 90 dos ERROS. O
# valor anterior, 0.68, ficava abaixo disso — o sistema pagava credito de
# similaridade para ruido.
DEFAULT_MIN_SIMILARITY: float = 0.755

# Similaridade a partir da qual a evidencia do banco conta como maxima.
# MEDIDO: percentil 75 dos acertos. Os acertos tem mediana 0.930 e p90 0.970;
# os erros, mediana 0.539 e p90 0.755 — a similaridade separa bem, e a escala
# agora comeca onde o ruido acaba.
DEFAULT_MAX_SIMILARITY: float = 0.959

# Margem (topo menos rival de outra marca) que ja caracteriza escolha confiante.
DEFAULT_CONFIDENT_MARGIN: float = 0.15

# Inliers de RANSAC que ja caracterizam geometria confirmada.
# MEDIDO: os erros tem mediana 4 inliers e p90 de 8; os acertos, mediana 27 e
# p25 de 17. Em 17 a evidencia ja esta muito acima de qualquer ruido — exigir o
# p75 (49) puniria logo chapado, que rende poucos pontos por natureza.
DEFAULT_CONFIDENT_INLIERS: float = 17.0

# Pontuacao a partir da qual a regiao e aceita sem humano.
DEFAULT_ACCEPT: float = 0.80

# Pontuacao abaixo da qual a regiao e descartada sem humano.
DEFAULT_REJECT: float = 0.40

# Margem abaixo da qual duas marcas do mesmo grupo sao consideradas empatadas.
DEFAULT_TIE_MARGIN: float = 0.06

# Inliers minimos para o orfao geometrico disparar. Abaixo disso o casamento de
# pontos encontra coerencia em ruido, e a fila de orfaos vira lixo.
DEFAULT_ORPHAN_MIN_INLIERS: int = 20

# Similaridade abaixo da qual, havendo confirmacao geometrica, a regiao e orfa:
# o banco tem a marca e nao tem esta variacao dela.
# MEDIDO: percentil 10 dos acertos. Em 0.88 a regra disparava no MEIO da faixa
# normal — uma regiao com 27 inliers e similaridade 0.869 virava orfa por onze
# milesimos, sem faltar variacao nenhuma no banco.
DEFAULT_ORPHAN_MAX_SIMILARITY: float = 0.664

# Densidade de borda abaixo da qual a imagem nao tem estrutura suficiente para
# valer uma passada de detector. Permissivo de proposito — o que se descarta
# aqui nunca mais volta, e o recall do detector e o teto do sistema.
DEFAULT_MIN_EDGE_DENSITY: float = 0.004

# Fracao da menor caixa coberta pela maior a partir da qual dois recortes sao o
# MESMO logo, e nao dois logos vizinhos. Nao e IoU: IoU e cego para aninhamento,
# e recorte dentro de recorte sai com IoU baixo mesmo estando 100% contido.
DEFAULT_NESTED_CONTAINMENT: float = 0.80


# Consenso a partir do qual a regiao e aceita sem humano, independentemente da
# pontuacao. MEDIDO em 84 imagens rotuladas: acima de 0.80 nao houve um unico
# erro em 15 regioes. E o sinal que resgata logo chapado — swoosh, wordmark —
# cuja geometria nao tem canto para dar ponto e cuja similaridade absoluta cai
# na faixa do ruido.
DEFAULT_CONSENSUS_ACCEPT: float = 0.80

# Concordantes ABSOLUTOS exigidos junto com o consenso. Sem isto, marca com 2
# referencias no banco alcanca consenso 1.0 trivialmente — o teto do consenso e
# `min(top-k, referencias da marca)`, e unanimidade de 2 nao vale o mesmo que
# unanimidade de 25.
DEFAULT_CONSENSUS_MIN_AGREEING: int = 8
