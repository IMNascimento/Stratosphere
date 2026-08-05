"""Limiares de dominio com o significado de cada valor.

Estes sao **defaults de partida**, nao verdades. Todo limiar aqui deve ser
recalibrado com dado proprio antes de operar: a escala de similaridade depende
do codificador, e a escala de confianca depende do detector. Um valor importado
de outro projeto costuma estar em outra escala e falha em silencio.

O lugar de sobrescrever e `config/settings.py`, nunca este arquivo.
"""

# Similaridade abaixo da qual a evidencia do banco conta como nula.
# Deve ficar proximo do percentil alto da distribuicao de regioes SEM logo: e o
# ponto em que o banco para de distinguir sinal de fundo.
DEFAULT_MIN_SIMILARITY: float = 0.68

# Similaridade a partir da qual a evidencia do banco conta como maxima.
# Proximo da mediana das correspondencias corretas.
DEFAULT_MAX_SIMILARITY: float = 0.97

# Margem (topo menos rival de outra marca) que ja caracteriza escolha confiante.
DEFAULT_CONFIDENT_MARGIN: float = 0.15

# Inliers de RANSAC que ja caracterizam geometria confirmada.
DEFAULT_CONFIDENT_INLIERS: float = 25.0

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
DEFAULT_ORPHAN_MAX_SIMILARITY: float = 0.88

# Densidade de borda abaixo da qual a imagem nao tem estrutura suficiente para
# valer uma passada de detector. Permissivo de proposito — o que se descarta
# aqui nunca mais volta, e o recall do detector e o teto do sistema.
DEFAULT_MIN_EDGE_DENSITY: float = 0.004
