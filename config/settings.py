"""Configuracao da aplicacao - apenas dataclasses com defaults, sem logica.

Este modulo nao le variavel de ambiente e nao contem credencial. Leitura de
ambiente e responsabilidade do container de infraestrutura.

**Sobre os defaults numericos.** Todos os limiares aqui sao pontos de partida,
nao verdades. Dois deles falham de forma especialmente traicoeira quando
importados de outro contexto:

- `DetectorConfig.confidence_threshold` esta na escala do detector em uso. Um
  valor que faz sentido para um detector treinado costuma zerar o recall de um
  detector de vocabulario aberto, cuja distribuicao de confianca e muito mais
  comprimida.
- `RoutingConfig.min_similarity` esta na escala do codificador. Vetores de
  crops nao relacionados raramente ficam proximos de zero - mapear a partir de
  zero faz parede lisa parecer evidencia.

Recalibre com dado proprio antes de operar.

Typical usage:
    config = AppConfig()
    config = replace(config, detector=replace(config.detector, confidence_threshold=0.08))
"""

from dataclasses import dataclass, field

from domain.constants import thresholds

# Prompts de CONCEITO para detectores de vocabulario aberto.
# NUNCA colocar nome de marca aqui: isso reintroduz o acoplamento que a
# arquitetura existe para eliminar, e marca nova volta a exigir retreino.
DEFAULT_CONCEPTS: tuple[str, ...] = (
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
        identifier: Modelo a carregar.
        confidence_threshold: Confianca minima para a regiao seguir adiante.
            **Baixo de proposito.** O detector so precisa acertar onde; a busca
            vetorial filtra quem. Limiar alto troca recall - que e
            irrecuperavel, porque a regiao nunca chega ao codificador - por
            precisao, que e recuperavel adiante.
        max_regions: Teto de regioes por imagem, apos ordenar por confianca.
        min_relative_area: Fracao minima da area da imagem. Um simbolo em
            manga de camisa pode ocupar menos de 0,01% de uma foto grande.
        max_relative_area: Acima disto a caixa e a imagem, nao um logo.
        min_side: Menor lado aceitavel, em pixels.
        suppression_iou: IoU acima do qual duas caixas sao consideradas a
            mesma. A supressao e **agnostica de conceito**: detectores de
            vocabulario aberto disparam varios conceitos sobre o mesmo pixel, e
            suprimir por conceito deixaria a mesma regiao passar duas vezes.
        inference_side: Maior lado ao qual a imagem e reduzida antes da
            deteccao. Reduzir demais faz simbolos pequenos deixarem de existir
            antes de qualquer modelo opinar.
        concepts: Prompts de conceito. Nunca nomes de marca.
    """

    identifier: str = "google/owlv2-base-patch16-ensemble"
    confidence_threshold: float = 0.05
    max_regions: int = 300
    min_relative_area: float = 0.00004
    max_relative_area: float = 0.60
    min_side: int = 12
    suppression_iou: float = 0.60
    inference_side: int = 1024
    concepts: tuple[str, ...] = DEFAULT_CONCEPTS


@dataclass(frozen=True)
class EncoderConfig:
    """Parametros da camada que transforma regiao em vetor.

    Attributes:
        identifier: Modelo a carregar.
        aggregation: Como reduzir os tokens do modelo a um vetor. `centro` usa
            apenas o quarto central dos patches - o detector ja centra a caixa
            no logo, entao a borda do recorte e contexto por construcao, e
            inclui-la faz o vetor descrever o fundo em vez da marca.
        batch_size: Quantas regioes por passada.
        crop_margin: Contexto proporcional ao redor da caixa.
        crop_side: Lado do quadrado final, com letterbox.
        min_side_to_upscale: Recorte menor que isto e ampliado antes de
            codificar.
        backend: Qual adaptador roda. `siglip` usa a cabeca de pooling do
            modelo; `dinov2` agrega retalhos conforme `aggregation`.

    **O default e `siglip` por medicao, nao por preferencia.** Com o proprio
    banco como conjunto de teste - pares da mesma marca em fotos diferentes
    contra pares de marcas diferentes na MESMA foto - o DINOv2 separa com AUC
    0.515 (moeda) e o SigLIP2 com 0.812. O DINOv2 estava codificando a
    superficie, nao a marca. Ver `infrastructure/encoding/siglip_encoder.py`.

    Trocar `identifier` ou `backend` **invalida o indice e todos os limiares de
    similaridade**. A assinatura do codificador entra no indice e a carga
    recusa a combinacao errada, entao o erro aparece - mas os limiares de
    `RoutingConfig` nao tem essa protecao e precisam ser refeitos a mao.
    """

    identifier: str = "google/siglip2-base-patch16-224"
    backend: str = "siglip"
    aggregation: str = "pooler"
    batch_size: int = 32
    crop_margin: float = 0.12
    crop_side: int = 224
    min_side_to_upscale: int = 64


@dataclass(frozen=True)
class SearchConfig:
    """Parametros da busca vetorial no banco de referencia.

    Attributes:
        neighbors: Quantos candidatos retornar por regiao. **Precisa ser grande
            o bastante para uma segunda marca aparecer**: com dezenas de
            referencias por marca, um valor pequeno devolve so a propria marca e
            a margem calculada a partir disso perde o sentido.
        redundancy_similarity: Referencias da MESMA marca acima disto sao
            redundantes - nao acrescentam cobertura e ocupam o topo da busca com
            copias.
        alert_similarity: Referencias de marcas DIFERENTES acima disto sao
            reportadas na auditoria. Cada par e um falso positivo agendado.

    **Os dois limiares vivem na escala do codificador e nao sao portateis.**
    Medido: no DINOv2, pares da mesma marca em fotos diferentes tinham mediana
    0.671; no SigLIP2, 0.840 com p90 em 0.919. Manter 0.95 apos a troca fez a
    deduplicacao comer referencia legitima - `amazon` caiu de 26 para 3 - porque
    o que era "quase copia" numa escala e "mesma marca, outra foto" na outra.
    """

    neighbors: int = 25
    redundancy_similarity: float = 0.985
    alert_similarity: float = 0.92


@dataclass(frozen=True)
class GeometryConfig:
    """Parametros da verificacao de que e o mesmo desenho.

    Attributes:
        enabled: Se a camada roda.
        entry_similarity: Similaridade minima para um candidato ser
            verificado, e tambem o portao que decide onde a geometria roda
            quando `only_when_uncertain` esta ligado. **Vive na escala do
            codificador**: 0.45 valia para o DINOv2 e, na escala do SigLIP2,
            deixava passar ate parede lisa - o portao existia e nao filtrava
            nada. Permissiva dentro da escala certa, e nao fora dela. A busca vetorial e fraca
            exatamente onde o casamento de pontos e forte - mudanca de ponto de
            vista. Um portao alto so deixa passar o que ja estava decidido, e a
            verificacao vira enfeite.
        max_references: Quantas marcas verificar por regiao. Uma referencia por
            marca: o objetivo e decidir ENTRE marcas.
        min_inliers: Pontos coerentes para confirmar.
        lowe_ratio: Corte do teste de razao entre os dois melhores pares. So o
            `SiftVerifier` usa - o LightGlue casa por atencao, vendo os dois
            conjuntos de pontos juntos, e nao precisa desse desempate.
        reprojection_error: Tolerancia do ajuste robusto, em pixels.
        max_points: Teto de pontos extraidos por imagem.
        matcher: Qual implementacao roda. `lightglue` usa pontos aprendidos;
            `sift` e a implementacao anterior, mantida para comparacao.
        only_when_uncertain: Se a verificacao roda **apenas** nas regioes que a
            pontuacao nao resolveu. E a ordem que a arquitetura sempre prometeu
            - camada cara so ve o que a barata deixou passar - e que a
            implementacao anterior nao cumpria: a geometria rodava nas ~55
            regioes de cada imagem, 4 referencias cada, inclusive nas ja
            decididas. Regiao rejeitada com similaridade acima de
            `entry_similarity` continua sendo verificada, senao o orfao
            geometrico deixaria de existir.
        match_side: Lado, em pixels, para o qual a imagem e reamostrada antes de
            extrair ponto. **Nao e o lado do recorte** (224px): ponto aprendido
            em 224px e escasso, e a mesma comparacao a 448px rende varias vezes
            mais correspondencia. So o `lightglue` usa.

    **`min_inliers` depende do matcher.** As escalas nao sao comparaveis: medido
    em 60 pares certos, SIFT devolve mediana de 2 inliers e LightGlue devolve
    86. Trocar `matcher` sem refazer `tools/calibrate_thresholds.py` faz a
    camada confirmar tudo ou nada.
    """

    enabled: bool = True
    entry_similarity: float = 0.86
    max_references: int = 4
    min_inliers: int = 57
    lowe_ratio: float = 0.75
    reprojection_error: float = 4.0
    max_points: int = 1024
    matcher: str = "lightglue"
    match_side: int = 448
    only_when_uncertain: bool = True


@dataclass(frozen=True)
class RoutingConfig:
    """Pesos e limiares da decisao final.

    Os pesos devem somar 1. O peso de geometria e redistribuido automaticamente
    quando a verificacao nao opina - ver `QueueRouter`.

    Attributes:
        similarity_weight: Contribuicao da semelhanca com a melhor referencia.
        consensus_weight: Contribuicao de quanto o top-k concorda com a marca
            escolhida. Peso alto de proposito: medido em imagem real, consenso
            separa logo verdadeiro de ruido melhor que similaridade absoluta.
        margin_weight: Contribuicao da vantagem sobre a rival mais proxima.
        geometry_weight: Contribuicao da confirmacao de mesmo desenho.
        detection_weight: Contribuicao da confianca do detector. Pequeno de
            proposito: essa confianca raramente separa logo de fundo.
        min_similarity: Abaixo disto a evidencia do banco conta como nula.
        max_similarity: A partir disto conta como maxima.
        confident_margin: Margem que ja caracteriza escolha confiante.
        confident_inliers: Inliers que ja caracterizam geometria confirmada.
        accept: Pontuacao a partir da qual a regiao e aceita sem humano.
        reject: Pontuacao abaixo da qual a regiao e descartada sem humano.
        orphan_min_inliers: Inliers minimos para o orfao geometrico disparar.
        orphan_max_similarity: Similaridade abaixo da qual, havendo confirmacao
            geometrica, a regiao e orfa.
        consensus_accept: Consenso a partir do qual a regiao e aceita sem humano.
        consensus_min_agreeing: Concordantes absolutos exigidos junto.
        geometry_accept_inliers: Inliers a partir dos quais a geometria
            aceita sozinha, sem passar pelo limiar de pontuacao.
        informative_inliers: Abaixo disto o veredito conta como silencio.
        corroboration_enabled: Se regiao em revisao pode ser aceita quando a
            propria imagem ja confirmou aquela marca em outra caixa.
        corroboration_min_similarity: Similaridade propria minima para a
            regiao ser promovida por corroboracao.
        nested_containment: Fracao da menor caixa coberta pela maior a partir da
            qual dois recortes da MESMA marca sao o mesmo logo, e so o de melhor
            pontuacao entra no relatorio. Nao e IoU - ver `NestedRegionResolver`.
    """

    similarity_weight: float = 0.30
    consensus_weight: float = 0.20
    margin_weight: float = 0.12
    geometry_weight: float = 0.32
    detection_weight: float = 0.06

    min_similarity: float = thresholds.DEFAULT_MIN_SIMILARITY
    max_similarity: float = thresholds.DEFAULT_MAX_SIMILARITY
    confident_margin: float = thresholds.DEFAULT_CONFIDENT_MARGIN
    confident_inliers: float = thresholds.DEFAULT_CONFIDENT_INLIERS
    accept: float = thresholds.DEFAULT_ACCEPT
    reject: float = thresholds.DEFAULT_REJECT
    orphan_min_inliers: int = thresholds.DEFAULT_ORPHAN_MIN_INLIERS
    orphan_max_similarity: float = thresholds.DEFAULT_ORPHAN_MAX_SIMILARITY
    consensus_accept: float = thresholds.DEFAULT_CONSENSUS_ACCEPT
    consensus_min_agreeing: int = thresholds.DEFAULT_CONSENSUS_MIN_AGREEING
    geometry_accept_inliers: float = thresholds.DEFAULT_GEOMETRY_ACCEPT_INLIERS
    informative_inliers: int = thresholds.DEFAULT_INFORMATIVE_INLIERS
    nested_same_brand_containment: float = thresholds.DEFAULT_NESTED_SAME_BRAND_CONTAINMENT
    corroboration_enabled: bool = True
    corroboration_min_similarity: float = thresholds.DEFAULT_CORROBORATION_MIN_SIMILARITY
    nested_containment: float = thresholds.DEFAULT_NESTED_CONTAINMENT


@dataclass(frozen=True)
class ConfusionConfig:
    """Politica de marcas confundiveis e de marcas fora do portfolio.

    Attributes:
        groups: Conjuntos de marcas que compartilham linguagem visual. Empate
            dentro de um grupo forca desempate, independentemente da pontuacao.
        negatives: Marcas cadastradas que nao sao clientes. Sem elas o sistema e
            obrigado a escolher entre as marcas do portfolio e sempre escolhe
            alguma.
        tie_margin: Diferenca de similaridade abaixo da qual duas marcas do
            mesmo grupo empatam.
    """

    groups: tuple[tuple[str, ...], ...] = ()
    negatives: tuple[str, ...] = ()
    tie_margin: float = thresholds.DEFAULT_TIE_MARGIN


@dataclass(frozen=True)
class PreFilterConfig:
    """Descarte barato de imagem sem estrutura, antes de gastar detector.

    Permissivo de proposito: o recall do detector e o teto do sistema, e o que
    se descarta aqui nunca mais volta.

    Attributes:
        enabled: Se o pre-filtro roda.
        min_side: Imagem com lado menor que isto e miniatura ou icone.
        min_edge_density: Fracao minima de pixels com gradiente forte.
    """

    enabled: bool = True
    min_side: int = 96
    min_edge_density: float = thresholds.DEFAULT_MIN_EDGE_DENSITY


@dataclass(frozen=True)
class JudgeConfig:
    """Parametros da segunda opiniao visual sobre o que ficou em revisao.

    **Desligada por default.** Ligar carrega um terceiro modelo na GPU, ao lado
    do detector e do codificador. Quem opera decide se tem VRAM e tempo para
    isso; o default nao decide por ele.

    Attributes:
        enabled: Se o juiz roda. A CLI liga com `--vlm`.
        model: Peso a carregar. O default e aberto (apache-2.0) e cabe em 12 GB
            junto com o resto da pipeline. Trocar por um maior melhora o
            parecer e custa VRAM - escolha de quem opera, nao default.
        max_regions: Teto de regioes julgadas por imagem, das de maior
            pontuacao para as de menor. E o freio de tempo: sem ele, uma imagem
            com 40 regioes em revisao vira 40 passadas de VLM.
        confirm_above: Probabilidade de "sim" a partir da qual o juiz confirma.
        deny_below: Probabilidade de "sim" ate a qual o juiz nega.

    A faixa entre `deny_below` e `confirm_above` e a abstencao - o juiz olhou e
    nao se decidiu, e a regiao segue para a fila humana. Ela e larga de
    proposito.

    **Os dois cortes sao medidos, nao arbitrados.** A probabilidade que um
    modelo de instrucao devolve nao vem calibrada: perguntado "sao a mesma
    marca?", ele concorda quase sempre. Medido neste projeto, acerto e erro
    caem os dois na casa dos 0.6 com a pergunta em portugues. O que tem sinal e
    a **ordem**, e por isso os cortes saem de percentil das duas distribuicoes:

        poetry run python tools/calibrate_judge.py --rotuladas <pasta> --banco indice

    Trocar o modelo em `model` **invalida os dois cortes** - cada modelo tem seu
    proprio vies. Recalibre antes de confiar no resultado.
    """

    enabled: bool = False
    model: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_regions: int = 12
    confirm_above: float = thresholds.DEFAULT_JUDGE_CONFIRM_ABOVE
    deny_below: float = thresholds.DEFAULT_JUDGE_DENY_BELOW


@dataclass(frozen=True)
class AppConfig:
    """Configuracao completa da aplicacao.

    Attributes:
        detector: Camada que encontra onde ha marca.
        encoder: Camada que transforma regiao em vetor.
        search: Parametros da busca vetorial.
        geometry: Camada que confirma se e o mesmo desenho.
        routing: Pesos e limiares da decisao.
        confusion: Politica de marcas confundiveis e negativas.
        pre_filter: Descarte de imagem sem estrutura.
        judge: Segunda opiniao visual sobre o que ficou em revisao.
        device: Onde os modelos rodam. `cuda:0` ou `cpu`.
        precision: Precisao numerica dos modelos.
    """

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    confusion: ConfusionConfig = field(default_factory=ConfusionConfig)
    pre_filter: PreFilterConfig = field(default_factory=PreFilterConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    device: str = "cuda:0"
    precision: str = "float16"
