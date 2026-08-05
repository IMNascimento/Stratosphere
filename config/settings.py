"""Configuracao da aplicacao — apenas dataclasses com defaults, sem logica.

Este modulo nao le variavel de ambiente e nao contem credencial. Leitura de
ambiente e responsabilidade do container de infraestrutura.

**Sobre os defaults numericos.** Todos os limiares aqui sao pontos de partida,
nao verdades. Dois deles falham de forma especialmente traicoeira quando
importados de outro contexto:

- `DetectorConfig.limiar_de_confianca` esta na escala do detector em uso. Um
  valor que faz sentido para um detector treinado costuma zerar o recall de um
  detector de vocabulario aberto, cuja distribuicao de confianca e muito mais
  comprimida.
- `RoteamentoConfig.similaridade_minima` esta na escala do codificador. Vetores
  de crops nao relacionados raramente ficam proximos de zero — mapear a partir
  de zero faz parede lisa parecer evidencia.

Recalibre com dado proprio antes de operar.

Typical usage:
    config = AppConfig()
    config = replace(config, detector=replace(config.detector, limiar_de_confianca=0.08))
"""

from dataclasses import dataclass, field

from domain.constants import limiares

# Prompts de CONCEITO para detectores de vocabulario aberto.
# NUNCA colocar nome de marca aqui: isso reintroduz o acoplamento que a
# arquitetura existe para eliminar, e marca nova volta a exigir retreino.
CONCEITOS_PADRAO: tuple[str, ...] = (
    "logo",
    "brand logo",
    "emblem",
    "wordmark",
    "printed label on packaging",
    "logo on clothing",
    "sign with text",
)


@dataclass(frozen=True)
class DetectorConfig:
    """Parametros da camada que encontra ONDE ha marca grafica.

    Attributes:
        identificador: Modelo a carregar.
        limiar_de_confianca: Confianca minima para a regiao seguir adiante.
            **Baixo de proposito.** O detector so precisa acertar onde; a busca
            vetorial filtra quem. Limiar alto troca recall — que e
            irrecuperavel, porque a regiao nunca chega ao codificador — por
            precisao, que e recuperavel adiante.
        maximo_de_regioes: Teto de regioes por imagem, apos ordenar por
            confianca.
        area_minima_relativa: Fracao minima da area da imagem. Um simbolo em
            manga de camisa pode ocupar menos de 0,01% de uma foto grande.
        area_maxima_relativa: Acima disto a caixa e a imagem, nao um logo.
        lado_minimo: Menor lado aceitavel, em pixels.
        iou_de_supressao: IoU acima do qual duas caixas sao consideradas a
            mesma. A supressao e **agnostica de conceito**: detectores de
            vocabulario aberto disparam varios conceitos sobre o mesmo pixel, e
            suprimir por conceito deixaria a mesma regiao passar duas vezes.
        lado_de_inferencia: Maior lado ao qual a imagem e reduzida antes da
            deteccao. Reduzir demais faz simbolos pequenos deixarem de existir
            antes de qualquer modelo opinar.
        conceitos: Prompts de conceito. Nunca nomes de marca.
    """

    identificador: str = "google/owlv2-base-patch16-ensemble"
    limiar_de_confianca: float = 0.05
    maximo_de_regioes: int = 300
    area_minima_relativa: float = 0.00004
    area_maxima_relativa: float = 0.60
    lado_minimo: int = 12
    iou_de_supressao: float = 0.60
    lado_de_inferencia: int = 1024
    conceitos: tuple[str, ...] = CONCEITOS_PADRAO


@dataclass(frozen=True)
class CodificadorConfig:
    """Parametros da camada que transforma regiao em vetor.

    Attributes:
        identificador: Modelo a carregar.
        agregacao: Como reduzir os tokens do modelo a um vetor. `centro` usa
            apenas o quarto central dos patches — o detector ja centra a caixa
            no logo, entao a borda do recorte e contexto por construcao, e
            inclui-la faz o vetor descrever o fundo em vez da marca.
        tamanho_do_lote: Quantas regioes por passada.
        margem_do_recorte: Contexto proporcional ao redor da caixa.
        lado_do_recorte: Lado do quadrado final, com letterbox.
        lado_minimo_para_ampliar: Recorte menor que isto e ampliado antes de
            codificar.
    """

    identificador: str = "facebook/dinov2-base"
    agregacao: str = "centro"
    tamanho_do_lote: int = 32
    margem_do_recorte: float = 0.12
    lado_do_recorte: int = 224
    lado_minimo_para_ampliar: int = 64


@dataclass(frozen=True)
class BuscaConfig:
    """Parametros da busca vetorial no banco de referencia.

    Attributes:
        vizinhos: Quantos candidatos retornar por regiao. **Precisa ser grande o
            bastante para uma segunda marca aparecer**: com dezenas de
            referencias por marca, um valor pequeno devolve so a propria marca e
            a margem calculada a partir disso perde o sentido.
        similaridade_de_redundancia: Referencias da MESMA marca acima disto sao
            redundantes — nao acrescentam cobertura e ocupam o topo da busca com
            copias.
        similaridade_de_alerta: Referencias de marcas DIFERENTES acima disto sao
            reportadas na auditoria. Cada par e um falso positivo agendado.
    """

    vizinhos: int = 25
    similaridade_de_redundancia: float = 0.95
    similaridade_de_alerta: float = 0.85


@dataclass(frozen=True)
class GeometriaConfig:
    """Parametros da verificacao de que e o mesmo desenho.

    Attributes:
        ativa: Se a camada roda.
        similaridade_de_entrada: Similaridade minima para um candidato ser
            verificado. **Deliberadamente permissiva.** A busca vetorial e fraca
            exatamente onde o casamento de pontos e forte — mudanca de ponto de
            vista. Um portao alto so deixa passar o que ja estava decidido, e a
            verificacao vira enfeite.
        maximo_de_referencias: Quantas marcas verificar por regiao. Uma
            referencia por marca: o objetivo e decidir ENTRE marcas.
        inliers_minimos: Pontos coerentes para confirmar.
        razao_de_lowe: Corte do teste de razao entre os dois melhores pares.
        erro_de_reprojecao: Tolerancia do ajuste robusto, em pixels.
        maximo_de_pontos: Teto de pontos extraidos por imagem.
    """

    ativa: bool = True
    similaridade_de_entrada: float = 0.45
    maximo_de_referencias: int = 4
    inliers_minimos: int = 8
    razao_de_lowe: float = 0.75
    erro_de_reprojecao: float = 5.0
    maximo_de_pontos: int = 800


@dataclass(frozen=True)
class RoteamentoConfig:
    """Pesos e limiares da decisao final.

    Os pesos devem somar 1. O peso de geometria e redistribuido automaticamente
    quando a verificacao nao opina — ver `RoteadorDeFila`.

    Attributes:
        peso_similaridade: Contribuicao da semelhanca com a melhor referencia.
        peso_margem: Contribuicao da vantagem sobre a rival mais proxima.
        peso_geometria: Contribuicao da confirmacao de mesmo desenho.
        peso_deteccao: Contribuicao da confianca do detector. Pequeno de
            proposito: essa confianca raramente separa logo de fundo.
        similaridade_minima: Abaixo disto a evidencia do banco conta como nula.
        similaridade_maxima: A partir disto conta como maxima.
        margem_confiante: Margem que ja caracteriza escolha confiante.
        inliers_confiantes: Inliers que ja caracterizam geometria confirmada.
        aceite: Pontuacao a partir da qual a regiao e aceita sem humano.
        rejeicao: Pontuacao abaixo da qual a regiao e descartada sem humano.
        orfao_inliers_minimos: Inliers minimos para o orfao geometrico disparar.
        orfao_similaridade_maxima: Similaridade abaixo da qual, havendo
            confirmacao geometrica, a regiao e orfa.
    """

    peso_similaridade: float = 0.36
    peso_margem: float = 0.16
    peso_geometria: float = 0.40
    peso_deteccao: float = 0.08

    similaridade_minima: float = limiares.SIMILARIDADE_MINIMA_PADRAO
    similaridade_maxima: float = limiares.SIMILARIDADE_MAXIMA_PADRAO
    margem_confiante: float = limiares.MARGEM_CONFIANTE_PADRAO
    inliers_confiantes: float = limiares.INLIERS_CONFIANTES_PADRAO
    aceite: float = limiares.ACEITE_PADRAO
    rejeicao: float = limiares.REJEICAO_PADRAO
    orfao_inliers_minimos: int = limiares.ORFAO_INLIERS_MINIMOS_PADRAO
    orfao_similaridade_maxima: float = limiares.ORFAO_SIMILARIDADE_MAXIMA_PADRAO


@dataclass(frozen=True)
class ConfusaoConfig:
    """Politica de marcas confundiveis e de marcas fora do portfolio.

    Attributes:
        grupos: Conjuntos de marcas que compartilham linguagem visual. Empate
            dentro de um grupo forca desempate, independentemente da pontuacao.
        negativas: Marcas cadastradas que nao sao clientes. Sem elas o sistema e
            obrigado a escolher entre as marcas do portfolio e sempre escolhe
            alguma.
        margem_de_empate: Diferenca de similaridade abaixo da qual duas marcas
            do mesmo grupo empatam.
    """

    grupos: tuple[tuple[str, ...], ...] = ()
    negativas: tuple[str, ...] = ()
    margem_de_empate: float = limiares.MARGEM_EMPATE_PADRAO


@dataclass(frozen=True)
class PreFiltroConfig:
    """Descarte barato de imagem sem estrutura, antes de gastar detector.

    Permissivo de proposito: o recall do detector e o teto do sistema, e o que
    se descarta aqui nunca mais volta.

    Attributes:
        ativo: Se o pre-filtro roda.
        lado_minimo: Imagem com lado menor que isto e miniatura ou icone.
        densidade_de_bordas_minima: Fracao minima de pixels com gradiente forte.
    """

    ativo: bool = True
    lado_minimo: int = 96
    densidade_de_bordas_minima: float = limiares.DENSIDADE_BORDAS_MINIMA_PADRAO


@dataclass(frozen=True)
class AppConfig:
    """Configuracao completa da aplicacao.

    Attributes:
        detector: Camada que encontra onde ha marca.
        codificador: Camada que transforma regiao em vetor.
        busca: Parametros da busca vetorial.
        geometria: Camada que confirma se e o mesmo desenho.
        roteamento: Pesos e limiares da decisao.
        confusao: Politica de marcas confundiveis e negativas.
        pre_filtro: Descarte de imagem sem estrutura.
        dispositivo: Onde os modelos rodam. `cuda:0` ou `cpu`.
        precisao: Precisao numerica dos modelos.
    """

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    codificador: CodificadorConfig = field(default_factory=CodificadorConfig)
    busca: BuscaConfig = field(default_factory=BuscaConfig)
    geometria: GeometriaConfig = field(default_factory=GeometriaConfig)
    roteamento: RoteamentoConfig = field(default_factory=RoteamentoConfig)
    confusao: ConfusaoConfig = field(default_factory=ConfusaoConfig)
    pre_filtro: PreFiltroConfig = field(default_factory=PreFiltroConfig)
    dispositivo: str = "cuda:0"
    precisao: str = "float16"
