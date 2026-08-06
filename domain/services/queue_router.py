"""O servico que decide o destino de cada regiao - o nucleo do sistema.

E dominio puro: recebe tudo por parametro, nao faz I/O, nao conhece detector nem
codificador. Isso e proposital. A decisao "aceito sozinho / mando para humano /
registro como referencia faltante" e politica de negocio, vale igual com qualquer
tecnologia plugada, e precisa ser verificavel sem GPU.

--------------------------------------------------------------------------
A ORDEM DAS REGRAS E A POLITICA
--------------------------------------------------------------------------
1. **Sem candidatos** - o banco nao respondeu nada.
2. **Marca negativa** - logo de quem nao interessa. Fila propria, porque contar
   como rejeicao mascara a qualidade real do sistema.
3. **Grupo de confusao** - empate entre marcas do mesmo grupo. Politica de
   negocio: forca desempate mesmo com similaridade alta.
4. **Orfao geometrico** - a geometria confirma e o banco nao reconhece.
5. **Limiares** - aceite, revisao ou rejeicao.

Reordenar isso muda o comportamento do produto, nao so do codigo. Duas
precedencias que parecem detalhe e nao sao:

- **Negativa e confusao vem ANTES do orfao geometrico.** O orfao geometrico ja
  carrega uma marca confiavel, entao e questao de evidencia; negativa e confusao
  sao classificacao e politica. Invertido, um empate gatorade x powerade com
  geometria forte viraria orfao e escaparia do desempate obrigatorio.
- **O empate so vale com evidencia.** Sem esse guarda, ruido puro cujos dois
  melhores palpites sao lixo empatado - e ambos do mesmo setor, que e justamente
  o que o codificador aproxima quando NAO ha sinal - inunda a fila humana.

--------------------------------------------------------------------------
POR QUE OS PESOS SAO RENORMALIZADOS
--------------------------------------------------------------------------
A verificacao geometrica **nao opina em toda regiao**: logo chapado, pequeno ou
vetorial demais nao tem pontos para casar. Com peso fixo, essas regioes sao
punidas por uma evidencia que nunca teve chance de existir - o teto da pontuacao
cai e o limiar de aceite fica inalcancavel para uma classe inteira de casos.
Quando nao ha veredito, os demais pesos sao renormalizados para somar 1.

Typical usage:
    router = QueueRouter(weights=..., calibration=..., groups=...)
    decision = router.route(region)
"""

from dataclasses import dataclass

from domain.entities.analyzed_region import AnalyzedRegion
from domain.entities.decision import Decision
from domain.enums.queue import Queue
from domain.services.confusion_groups import ConfusionGroups

_TERM_SIMILARITY = "similarity"
_TERM_CONSENSUS = "consensus"
_TERM_MARGIN = "margin"
_TERM_GEOMETRY = "geometry"
_TERM_DETECTION = "detection"


@dataclass(frozen=True)
class EvidenceWeights:
    """Quanto cada sinal vale na pontuacao final.

    Attributes:
        similarity: Peso do quanto a melhor referencia se parece com a regiao.
        consensus: Peso de quanto o top-k concorda com a marca escolhida. Vale
            muito porque separa casamento real de vizinho por acaso melhor que
            a similaridade absoluta - ver `AnalyzedRegion.brand_consensus`.
        margin: Peso do quanto a marca escolhida supera a rival mais proxima.
        geometry: Peso da confirmacao de que e o mesmo desenho.
        detection: Peso da confianca do detector. Costuma ser pequeno: a escala
            varia por detector e raramente separa logo de fundo.
    """

    similarity: float
    consensus: float
    margin: float
    geometry: float
    detection: float

    def __post_init__(self) -> None:
        """Valida pesos nao negativos que somem 1.

        Raises:
            ValueError: Se algum peso for negativo ou a soma nao for 1.
        """
        values = (
            self.similarity,
            self.consensus,
            self.margin,
            self.geometry,
            self.detection,
        )
        if any(weight < 0 for weight in values):
            raise ValueError(f"pesos nao podem ser negativos: {values}")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"pesos devem somar 1.0, somam {sum(values)}")


@dataclass(frozen=True)
class Calibration:
    """Como converter cada sinal bruto para a escala [0, 1].

    Estes valores dependem do codificador e do detector em uso. **Nao ha default
    universal** - a distribuicao de similaridade de um codificador nao vale para
    outro, e um limiar importado falha em silencio.

    Attributes:
        min_similarity: Abaixo disto a evidencia do banco conta como nula.
        max_similarity: A partir disto conta como maxima.
        confident_margin: Margem que ja caracteriza escolha confiante.
        confident_inliers: Inliers que ja caracterizam geometria confirmada.
        accept: Pontuacao a partir da qual a regiao e aceita sem humano.
        reject: Pontuacao abaixo da qual a regiao e descartada sem humano.
        orphan_min_inliers: Inliers minimos para o orfao geometrico disparar.
        orphan_max_similarity: Similaridade abaixo da qual, havendo
            confirmacao geometrica, a regiao e orfa.
        consensus_accept: Consenso a partir do qual a regiao e aceita sem
            humano, independentemente da pontuacao.
        informative_inliers: Abaixo disto a contagem de inliers e ruido, e o
            veredito conta como silencio - peso redistribuido - em vez de
            evidencia fraca. **Depende do matcher.**
        geometry_accept_inliers: Inliers a partir dos quais a geometria
            aceita sozinha. Medido com rotulo limpo, e nao derivado da
            pontuacao. **Depende do matcher** - trocar o verificador
            invalida este valor.
        consensus_min_agreeing: Concordantes absolutos exigidos junto com o
            consenso, para que unanimidade de marca pouco coberta nao valha o
            mesmo que unanimidade de marca bem coberta.
    """

    min_similarity: float
    max_similarity: float
    confident_margin: float
    confident_inliers: float
    accept: float
    reject: float
    orphan_min_inliers: int
    orphan_max_similarity: float
    consensus_accept: float
    consensus_min_agreeing: int
    geometry_accept_inliers: float
    informative_inliers: int

    def __post_init__(self) -> None:
        """Valida a coerencia entre os limiares.

        Raises:
            ValueError: Se algum intervalo estiver invertido ou degenerado.
        """
        if self.max_similarity <= self.min_similarity:
            raise ValueError(
                f"max_similarity deve superar a minima: "
                f"{self.max_similarity} <= {self.min_similarity}"
            )
        if self.accept <= self.reject:
            raise ValueError(f"aceite deve superar rejeicao: {self.accept} <= {self.reject}")
        if self.confident_margin <= 0 or self.confident_inliers <= 0:
            raise ValueError("confident_margin e confident_inliers devem ser positivos")
        if not 0.0 < self.consensus_accept <= 1.0:
            raise ValueError(f"consensus_accept deve estar em (0, 1]: {self.consensus_accept}")
        if self.consensus_min_agreeing < 1:
            raise ValueError(
                f"consensus_min_agreeing deve ser ao menos 1: {self.consensus_min_agreeing}"
            )


def _clamp_to_unit(value: float) -> float:
    """Prende um valor ao intervalo [0, 1].

    Args:
        value: Numero a limitar.

    Returns:
        O proprio valor, ou a borda mais proxima se estiver fora.
    """
    return max(0.0, min(1.0, value))


class QueueRouter:
    """Decide a fila de destino de uma regiao a partir das evidencias reunidas.

    Servico de dominio puro: sem I/O, sem estado entre chamadas, sem dependencia
    de framework. Duas chamadas com a mesma regiao produzem a mesma decisao.

    Attributes:
        weights: Quanto cada sinal vale na pontuacao.
        calibration: Como converter sinal bruto para escala [0, 1].
    """

    def __init__(
        self,
        weights: EvidenceWeights,
        calibration: Calibration,
        groups: ConfusionGroups,
    ) -> None:
        """Inicializa o roteador com pesos, calibragem e politica de marcas.

        Args:
            weights: Contribuicao de cada sinal para a pontuacao.
            calibration: Limiares de conversao e de decisao.
            groups: Politica de grupos de confusao e marcas negativas.
        """
        self.weights = weights
        self.calibration = calibration
        self._groups = groups

    def route(self, region: AnalyzedRegion) -> Decision:
        """Decide o destino da regiao aplicando as regras em ordem.

        Args:
            region: Regiao com deteccao, candidatos do banco e vereditos
                geometricos ja reunidos.

        Returns:
            A decisao, com a fila escolhida, a marca afirmada quando houver, e
            os motivos e contribuicoes que a produziram.
        """
        without_candidates = self._route_without_candidates(region)
        if without_candidates is not None:
            return without_candidates

        score, contributions = self._score(region)
        brand = region.top_brand

        negative = self._route_negative(region, score, contributions)
        if negative is not None:
            return negative

        confusion = self._route_confusion(region, score, contributions)
        if confusion is not None:
            return confusion

        judged = self._route_by_judge(region, brand, score, contributions)
        if judged is not None:
            return judged

        orphan = self._route_geometric_orphan(region, score, contributions)
        if orphan is not None:
            return orphan

        consensus = self._route_by_consensus(region, brand, score, contributions)
        if consensus is not None:
            return consensus

        geometry = self._route_by_geometry(region, brand, score, contributions)
        if geometry is not None:
            return geometry

        return self._route_by_threshold(region, brand, score, contributions)

    # -- regras, na ordem em que sao aplicadas -----------------------------

    def _route_without_candidates(self, region: AnalyzedRegion) -> Decision | None:
        """Decide quando o banco nao devolveu candidato algum.

        Args:
            region: Regiao analisada.

        Returns:
            A decisao, ou None se houver candidatos e a regra nao se aplicar.
        """
        if region.candidates:
            return None
        return Decision(
            queue=Queue.AUTO_REJECT,
            brand=None,
            score=0.0,
            reasons=("o banco de referencia nao devolveu candidato algum",),
        )

    def _route_negative(
        self,
        region: AnalyzedRegion,
        score: float,
        contributions: dict[str, float],
    ) -> Decision | None:
        """Decide quando a melhor resposta e marca fora do portfolio.

        Args:
            region: Regiao analisada.
            score: Pontuacao ja calculada.
            contributions: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        brand = region.top_brand
        if not self._groups.is_negative(brand) or score < self.calibration.reject:
            return None
        return Decision(
            queue=Queue.NEGATIVE,
            brand=brand,
            score=score,
            reasons=(
                f"{brand!r} esta no banco como concorrente, fora do portfolio. "
                "Contabilizar separado de 'sem logo' - aqui o sistema acertou.",
            ),
            contributions=contributions,
        )

    def _route_confusion(
        self,
        region: AnalyzedRegion,
        score: float,
        contributions: dict[str, float],
    ) -> Decision | None:
        """Decide quando ha empate entre marcas do mesmo grupo de confusao.

        O guarda de evidencia (`score >= reject`) nao e opcional: sem ele,
        ruido puro com dois palpites empatados do mesmo setor inunda a fila
        humana e destroi a estimativa de capacidade.

        Args:
            region: Regiao analisada.
            score: Pontuacao ja calculada.
            contributions: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        brand = region.top_brand
        rival = region.rival_brand
        tied = region.margin < self._groups.tie_margin
        if score < self.calibration.reject or not self._groups.same_group(brand, rival) or not tied:
            return None
        return Decision(
            queue=Queue.CONFUSION,
            brand=brand,
            score=score,
            reasons=(
                f"{brand!r} e {rival!r} pertencem ao grupo "
                f"{self._groups.group_of(brand)} e a margem e {region.margin:.3f}. "
                "Desempate obrigatorio com as referencias lado a lado.",
            ),
            contributions=contributions,
        )

    def _route_by_judge(
        self,
        region: AnalyzedRegion,
        brand: str | None,
        score: float,
        contributions: dict[str, float],
    ) -> Decision | None:
        """Decide quando um juiz visual olhou a regiao e opinou.

        O juiz so e consultado sobre o que ja tinha caido em revisao, entao a
        alternativa a este veredito nunca e "aceite" - e "uma pessoa vai olhar".
        Por isso ele decide: quem olhou as duas imagens sabe mais que a
        pontuacao ponderada, que so viu vetores.

        Tres saidas. A terceira - juiz discorda e **sabe nomear outra marca** -
        mantem a regiao na fila humana com a sugestao anexada ao motivo, para
        quem revisa nao comecar do zero. A regra existe para qualquer juiz; o
        adaptador local nao a dispara, porque modelo pequeno nao conhece marca
        regional e nomear so produziria alucinacao. Ver `IJudge`.

        Vem depois das regras de politica - negativa e confusao sao decisao de
        negocio e nao se terceiriza - e antes do orfao geometrico, porque um
        juiz que nega a marca desmonta a premissa do orfao.

        Args:
            region: Regiao analisada.
            brand: Marca do melhor candidato.
            score: Pontuacao ja calculada.
            contributions: Contribuicao de cada termo.

        Returns:
            A decisao, ou None quando nenhum juiz opinou sobre esta regiao.
        """
        verdict = region.judgement
        if verdict is None:
            return None

        if verdict.agrees:
            return Decision(
                queue=Queue.AUTO_ACCEPT,
                brand=verdict.brand,
                score=score,
                reasons=(
                    f"o juiz visual confirmou {verdict.brand!r} comparando a regiao com a "
                    f"referencia do banco (confianca {verdict.confidence:.2f}): {verdict.reason}",
                ),
                contributions=contributions,
            )

        if verdict.proposes_other_brand:
            return Decision(
                queue=Queue.REVIEW,
                brand=brand,
                score=score,
                reasons=(
                    f"o banco propos {brand!r}, mas o juiz visual reconheceu "
                    f"{verdict.brand!r} (confianca {verdict.confidence:.2f}): {verdict.reason}",
                    "conferencia humana com a sugestao do juiz em maos.",
                ),
                contributions=contributions,
            )

        return Decision(
            queue=Queue.AUTO_REJECT,
            brand=None,
            score=score,
            reasons=(
                f"o juiz visual negou {brand!r} comparando a regiao com a referencia "
                f"(confianca {verdict.confidence:.2f}): {verdict.reason}",
            ),
            contributions=contributions,
        )

    def _route_geometric_orphan(
        self,
        region: AnalyzedRegion,
        score: float,
        contributions: dict[str, float],
    ) -> Decision | None:
        """Decide quando a geometria confirma e o banco nao reconhece.

        Traducao do sinal: o banco tem a marca e **nao tem esta variacao dela**.
        E a referencia que falta - o insumo que alimenta o banco de volta.

        Args:
            region: Regiao analisada.
            score: Pontuacao ja calculada.
            contributions: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        verdict = region.confirmed_verdict
        if verdict is None:
            return None
        if verdict.inliers < self.calibration.orphan_min_inliers:
            return None
        if region.top_similarity >= self.calibration.orphan_max_similarity:
            return None
        return Decision(
            queue=Queue.ORPHAN,
            brand=verdict.brand,
            score=score,
            reasons=(
                f"a geometria confirma {verdict.brand!r} com {verdict.inliers} "
                f"inliers, mas a melhor similaridade do banco e "
                f"{region.top_similarity:.3f}. O banco tem a marca e nao tem "
                "esta variacao - promova esta regiao para o banco.",
            ),
            contributions=contributions,
        )

    def _route_by_consensus(
        self,
        region: AnalyzedRegion,
        brand: str | None,
        score: float,
        contributions: dict[str, float],
    ) -> Decision | None:
        """Aceita quando o banco inteiro concorda, mesmo com pontuacao baixa.

        A pontuacao ponderada cobra evidencia geometrica que **nem toda marca
        tem como produzir**: swoosh e wordmark sao lisos e rendem poucos pontos
        por natureza. Medido em 84 imagens rotuladas, o consenso acima de 0.80
        nao errou uma vez em 15 regioes - enquanto regioes com o top-k inteiro
        de uma marca so, e similaridade 0.92, iam para conferencia humana porque
        a geometria devolveu 6 inliers em vez de 25.

        Vem **depois** do orfao geometrico de proposito: quando a geometria
        confirma o desenho e a similaridade esta no rabo de baixo, a regiao e a
        referencia que falta, e promove-la ao banco vale mais que aceita-la.

        **Nao resgata regiao abaixo do limiar de rejeicao.** O consenso promove
        de revisao para aceite; nao promove de descarte para aceite.

        Args:
            region: Regiao analisada.
            brand: Marca do melhor candidato.
            score: Pontuacao ja calculada.
            contributions: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        if brand is None:
            return None
        # Piso: o consenso decide entre REVISAO e ACEITE, nunca entre REJEICAO e
        # aceite. Sem ele, uma regiao com pontuacao 0.37 - abaixo da linha em
        # que a propria politica manda descartar sem humano - entrava direto no
        # relatorio do cliente so porque o top-k era unanime. Consenso forte com
        # todo o resto fraco continua sendo motivo para olhar, nao para afirmar.
        if score < self.calibration.reject:
            return None
        # Consenso sem similaridade nao vale: "muitas referencias concordam" so
        # significa alguma coisa se ALGUMA delas estiver perto. Abaixo do piso de
        # similaridade nao ha evidencia nenhuma para o consenso somar.
        #
        # CASO MEDIDO: a bandeira do Brasil era aceita como 'cbf' com pontuacao
        # 0.44. O escudo da CBF contem um circulo azul com estrelas sobre verde e
        # amarelo, e a marca tem 65 referencias no banco - mais que qualquer
        # outra -, entao quase todo vizinho do top-25 era cbf e o consenso
        # chegava a 1.00 em cima de similaridade 0.75.
        if region.top_similarity < self.calibration.min_similarity:
            return None
        agreeing = region.agreeing_candidates
        if region.brand_consensus < self.calibration.consensus_accept:
            return None
        if agreeing < self.calibration.consensus_min_agreeing:
            return None
        return Decision(
            queue=Queue.AUTO_ACCEPT,
            brand=brand,
            score=score,
            reasons=(
                f"{agreeing} das {len(region.candidates)} respostas do banco sao {brand!r} "
                f"(consenso {region.brand_consensus:.2f}). Referencias independentes "
                "concordando valem mais que a geometria, que fica muda em logo chapado.",
                f"pontuacao={score:.3f} - aceite por consenso, nao por limiar",
            ),
            contributions=contributions,
        )

    def _route_by_geometry(
        self,
        region: AnalyzedRegion,
        brand: str | None,
        score: float,
        contributions: dict[str, float],
    ) -> Decision | None:
        """Aceita quando a geometria da evidencia que par errado nao alcanca.

        Regra simetrica a do consenso, para o caso oposto. O consenso resgata
        logo chapado, que rende poucos pontos e muita concordancia; esta resgata
        logo com desenho rico, que rende muitos pontos e pouca concordancia -
        tipicamente marca com poucas referencias no banco, onde o consenso e
        baixo por aritmetica e nao por duvida.

        **O limiar vem de medicao com rotulo limpo**, e nao da pontuacao: em 60
        pares de marcas diferentes, o maximo observado foi 52 inliers, contra
        mediana 86 nos pares da mesma marca. Acima de
        `geometry_accept_inliers` nenhum par errado chegou.

        Sem esta regra, regiao com 127 inliers e similaridade 0.946 ia para
        conferencia humana por 0.05 de pontuacao - pagando o preco de uma
        calibracao feita quando a geometria era muda em metade dos casos.

        **Nao resgata regiao abaixo do limiar de rejeicao**, pelo mesmo motivo
        que o consenso nao resgata: promover de descarte para aceite e outra
        decisao, e ninguem olhou.

        Args:
            region: Regiao analisada.
            brand: Marca do melhor candidato.
            score: Pontuacao ja calculada.
            contributions: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        if brand is None or score < self.calibration.reject:
            return None
        verdict = region.confirmed_verdict
        # A geometria precisa confirmar **a marca do topo**. Veredito forte de
        # outra marca nao e motivo para aceitar esta - e motivo para duvidar
        # das duas, que e o que a fila de revisao ja faz.
        if verdict is None or verdict.brand != brand:
            return None
        if verdict.inliers < self.calibration.geometry_accept_inliers:
            return None
        return Decision(
            queue=Queue.AUTO_ACCEPT,
            brand=brand,
            score=score,
            reasons=(
                f"geometria confirmou {brand!r} com {verdict.inliers} inliers de "
                f"{verdict.matches} correspondencias - acima de "
                f"{self.calibration.geometry_accept_inliers:.0f}, patamar que nenhum par de "
                "marcas diferentes alcancou na medicao.",
                f"pontuacao={score:.3f} - aceite por geometria, nao por limiar",
            ),
            contributions=contributions,
        )

    def _route_by_threshold(
        self,
        region: AnalyzedRegion,
        brand: str | None,
        score: float,
        contributions: dict[str, float],
    ) -> Decision:
        """Decide pelos limiares de aceite e rejeicao.

        Args:
            region: Regiao analisada.
            brand: Marca do melhor candidato.
            score: Pontuacao ja calculada.
            contributions: Contribuicao de cada termo.

        Returns:
            A decisao final. Regiao rejeitada nunca afirma marca.
        """
        reasons: list[str] = []
        verdict = region.best_verdict
        if verdict is not None:
            reasons.append(
                f"geometria: {verdict.inliers} inliers de "
                f"{verdict.matches} correspondencias vs {verdict.brand!r}"
            )
        else:
            reasons.append(
                "geometria nao opinou - sem pontos suficientes para casar. "
                "Os demais pesos foram renormalizados."
            )
        reasons.append(
            f"pontuacao={score:.3f} "
            f"(similaridade={region.top_similarity:.3f} "
            f"consenso={region.brand_consensus:.2f} "
            f"margem={region.margin:.3f})"
        )

        if score >= self.calibration.accept:
            reasons.append(f"pontuacao >= aceite ({self.calibration.accept})")
            return Decision(
                queue=Queue.AUTO_ACCEPT,
                brand=brand,
                score=score,
                reasons=tuple(reasons),
                contributions=contributions,
            )

        if score < self.calibration.reject:
            reasons.append(f"pontuacao < rejeicao ({self.calibration.reject})")
            return Decision(
                queue=Queue.AUTO_REJECT,
                brand=None,
                score=score,
                reasons=tuple(reasons),
                contributions=contributions,
            )

        reasons.append(
            f"faixa ambigua [{self.calibration.reject}, {self.calibration.accept}) "
            "- vai para conferencia humana"
        )
        return Decision(
            queue=Queue.REVIEW,
            brand=brand,
            score=score,
            reasons=tuple(reasons),
            contributions=contributions,
        )

    # -- pontuacao ---------------------------------------------------------

    def _score(self, region: AnalyzedRegion) -> tuple[float, dict[str, float]]:
        """Converte os sinais em uma pontuacao unica entre 0 e 1.

        Quando a verificacao geometrica nao opinou, o peso dela e redistribuido
        entre os demais em vez de contar como zero. Contar como zero puniria a
        regiao por uma evidencia que nunca teve chance de existir, e derrubaria
        o teto da pontuacao abaixo do limiar de aceite.

        Args:
            region: Regiao com candidatos e vereditos.

        Returns:
            Tupla `(score, contributions)`, onde as contribuicoes somam a
            pontuacao e permitem auditar a decisao sem reexecutar nada.
        """
        calibration = self.calibration
        span = calibration.max_similarity - calibration.min_similarity

        normalized = {
            _TERM_SIMILARITY: _clamp_to_unit(
                (region.top_similarity - calibration.min_similarity) / span
            ),
            _TERM_CONSENSUS: _clamp_to_unit(region.brand_consensus),
            _TERM_MARGIN: _clamp_to_unit(region.margin / calibration.confident_margin),
            _TERM_DETECTION: _clamp_to_unit(region.detection.confidence),
        }
        weights = {
            _TERM_SIMILARITY: self.weights.similarity,
            _TERM_CONSENSUS: self.weights.consensus,
            _TERM_MARGIN: self.weights.margin,
            _TERM_DETECTION: self.weights.detection,
        }

        verdict = region.best_verdict
        # Contagem dentro da faixa de ruido conta como SILENCIO, e nao como
        # evidencia fraca. Nao e a mesma coisa: evidencia fraca puxa a nota para
        # baixo, silencio redistribui o peso.
        #
        # Medido: par de marcas diferentes tem mediana de 8 inliers e chega a
        # 52; par da mesma marca tem mediana 86. Uma contagem de 12 e mais
        # provavel de vir de par errado que de par certo - chamar isso de "13%
        # da evidencia possivel" inventa um sinal que os numeros nao sustentam.
        #
        # O caso real que forcou esta regra: um swoosh com consenso 0.92 e 23
        # das 25 respostas do banco em 'nike' caiu de aceite para descarte
        # porque 12 inliers derrubaram a nota de 0.42 para 0.377, cruzando o
        # piso de rejeicao por 23 milesimos. Com o SIFT a mesma regiao era
        # aceita - nao por acerto, mas porque o SIFT ficava mudo ali e o
        # silencio a protegia.
        informative = verdict is not None and verdict.inliers >= calibration.informative_inliers
        if verdict is not None and informative:
            normalized[_TERM_GEOMETRY] = _clamp_to_unit(
                verdict.inliers / calibration.confident_inliers
            )
            weights[_TERM_GEOMETRY] = self.weights.geometry
        else:
            weights = self._renormalize(weights)

        # Trava final: a geometria so pode SOMAR. Ela e um dispositivo de
        # confirmacao - a pergunta dela e "e o mesmo desenho?", e "nao confirmei"
        # significa "nao sei", nunca "nao e". Uma contagem baixa nao e prova
        # contra: e ausencia de prova a favor, e quem carrega a prova contra sao
        # a similaridade, o consenso e a margem.
        #
        # MEDIDO em 20 imagens de rede social: sem esta trava, trocar SIFT por
        # LightGlue DERRUBAVA o resultado - 10 aceites contra 18, e 4 marcas
        # contra 6. O motivo e contraintuitivo e vale registrar: o SIFT fica
        # mudo em metade dos pares certos, e o silencio dele redistribuia o peso
        # e SUBIA a nota. O LightGlue opina em quase tudo, entao uma regiao com
        # 25 inliers passou a receber 0.28 de geometria onde antes recebia
        # redistribuicao - e caiu abaixo do limiar por ter ganhado um sinal.
        muted = self._renormalize(
            {term: weight for term, weight in weights.items() if term != _TERM_GEOMETRY}
        )
        if _TERM_GEOMETRY in weights and self._total(muted, normalized) > self._total(
            weights, normalized
        ):
            weights = muted

        contributions = {term: round(weights[term] * normalized[term], 4) for term in weights}
        return _clamp_to_unit(sum(contributions.values())), contributions

    @staticmethod
    def _total(weights: dict[str, float], normalized: dict[str, float]) -> float:
        """Soma ponderada dos termos presentes nos pesos.

        Args:
            weights: Peso de cada termo considerado.
            normalized: Valor normalizado de cada termo.

        Returns:
            A pontuacao que esses pesos produziriam.
        """
        return sum(weight * normalized[term] for term, weight in weights.items())

    @staticmethod
    def _renormalize(weights: dict[str, float]) -> dict[str, float]:
        """Reescala os pesos restantes para somarem 1.

        Args:
            weights: Pesos dos termos que produziram evidencia.

        Returns:
            Os mesmos termos com pesos proporcionais somando 1. Se a soma for
            zero, devolve os pesos como estao - nao ha o que reescalar.
        """
        total = sum(weights.values())
        if total <= 0:
            return weights
        return {term: weight / total for term, weight in weights.items()}
