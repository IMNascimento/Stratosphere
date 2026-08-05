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
SIMILARIDADE_MINIMA_PADRAO: float = 0.68

# Similaridade a partir da qual a evidencia do banco conta como maxima.
# Proximo da mediana das correspondencias corretas.
SIMILARIDADE_MAXIMA_PADRAO: float = 0.97

# Margem (topo menos rival de outra marca) que ja caracteriza escolha confiante.
MARGEM_CONFIANTE_PADRAO: float = 0.15

# Inliers de RANSAC que ja caracterizam geometria confirmada.
INLIERS_CONFIANTES_PADRAO: float = 25.0

# Pontuacao a partir da qual a regiao e aceita sem humano.
ACEITE_PADRAO: float = 0.80

# Pontuacao abaixo da qual a regiao e descartada sem humano.
REJEICAO_PADRAO: float = 0.40

# Margem abaixo da qual duas marcas do mesmo grupo sao consideradas empatadas.
MARGEM_EMPATE_PADRAO: float = 0.06

# Inliers minimos para o orfao geometrico disparar. Abaixo disso o casamento de
# pontos encontra coerencia em ruido, e a fila de orfaos vira lixo.
ORFAO_INLIERS_MINIMOS_PADRAO: int = 20

# Similaridade abaixo da qual, havendo confirmacao geometrica, a regiao e orfa:
# o banco tem a marca e nao tem esta variacao dela.
ORFAO_SIMILARIDADE_MAXIMA_PADRAO: float = 0.88

# Densidade de borda abaixo da qual a imagem nao tem estrutura suficiente para
# valer uma passada de detector. Permissivo de proposito — o que se descarta
# aqui nunca mais volta, e o recall do detector e o teto do sistema.
DENSIDADE_BORDAS_MINIMA_PADRAO: float = 0.004
