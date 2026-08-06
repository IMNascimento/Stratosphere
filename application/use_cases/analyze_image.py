"""Caso de uso que atravessa a pipeline inteira para uma imagem.

Orquestra, nao decide. Toda regra de negocio esta em `QueueRouter`; aqui ha
apenas a sequencia das camadas e o cuidado de nao gastar camada cara com o que a
camada barata ja descartou.

A ordem importa e nao e arbitraria: cada camada e mais cara que a anterior, e so
ve o que a anterior deixou passar. A verificacao geometrica, em particular, so
roda nos melhores candidatos de cada regiao - se rodasse em tudo, seria a camada
dominante do custo.

Typical usage:
    use_case = AnalyzeImageUseCase(detector, encoder, verifier, source, database, router, config)
    output = use_case.execute(AnalyzeImageCommand(path))
"""

from pathlib import Path

from application.dtos.analysis import AnalysisOutput, AnalyzeImageCommand, RegionOutput
from application.ports.i_detector import IDetector
from application.ports.i_encoder import IEncoder
from application.ports.i_geometric_verifier import IGeometricVerifier
from application.ports.i_image_source import IImageSource, RgbImage
from application.ports.i_judge import IJudge
from config.settings import AppConfig
from domain.entities.analyzed_region import AnalyzedRegion
from domain.entities.decision import Decision
from domain.enums.queue import Queue
from domain.repositories.i_reference_database import IReferenceDatabase
from domain.services.brand_corroboration_resolver import BrandCorroborationResolver
from domain.services.nested_region_resolver import NestedRegionResolver
from domain.services.queue_router import QueueRouter


class AnalyzeImageUseCase:
    """Executa pre-filtro, deteccao, codificacao, busca, geometria e roteamento."""

    def __init__(
        self,
        detector: IDetector,
        encoder: IEncoder,
        verifier: IGeometricVerifier,
        source: IImageSource,
        database: IReferenceDatabase,
        router: QueueRouter,
        resolver: NestedRegionResolver,
        corroborator: BrandCorroborationResolver,
        config: AppConfig,
        judge: IJudge | None = None,
    ) -> None:
        """Recebe as portas e os servicos de dominio ja construidos.

        Args:
            detector: Camada que encontra onde ha marca.
            encoder: Camada que transforma regiao em vetor.
            verifier: Camada que confirma se e o mesmo desenho.
            source: Acesso a imagem e recorte.
            database: Banco de referencia consultado pela busca vetorial.
            router: Servico de dominio que decide a fila.
            resolver: Servico de dominio que colapsa recortes do mesmo logo.
            corroborator: Servico que aceita regiao em revisao quando a
                propria imagem ja confirmou aquela marca em outra caixa.
            config: Parametros de todas as camadas.
            judge: Segunda opiniao visual sobre o que cair em revisao. None
                desliga a camada, e a pipeline se comporta exatamente como
                antes de ela existir.
        """
        self._detector = detector
        self._encoder = encoder
        self._verifier = verifier
        self._source = source
        self._database = database
        self._router = router
        self._resolver = resolver
        self._corroborator = corroborator
        self._config = config
        self._judge = judge

    def execute(self, command: AnalyzeImageCommand) -> AnalysisOutput:
        """Analisa uma imagem e devolve a decisao de cada regiao encontrada.

        Args:
            command: Pedido com o caminho da imagem.

        Returns:
            A analise completa. Quando o pre-filtro descarta a imagem,
            `discarded_by` explica o motivo e `regions` vem vazio.

        Raises:
            OSError: Se a imagem nao puder ser lida.
        """
        image = self._source.load(command.path)

        discard_reason = self._discard(image)
        if discard_reason is not None:
            return AnalysisOutput(path=command.path, discarded_by=discard_reason)

        detections = self._detector.detect(image)
        if not detections:
            return AnalysisOutput(path=command.path, discarded_by=None)

        crops = self._crop_all(image, detections)
        vectors = self._encoder.encode(crops)

        # Uma consulta so por imagem: o consenso precisa saber quantas
        # referencias a marca tem, e a contagem nao muda entre regioes.
        references_by_brand = self._database.references_by_brand()

        decided: list[tuple[AnalyzedRegion, Decision]] = []
        for index, detection in enumerate(detections):
            candidates = self._database.search(vectors[index], self._config.search.neighbors)

            region = AnalyzedRegion(
                identifier=f"{command.path.stem}-{index:03d}",
                detection=detection,
            ).with_candidates(candidates)
            region = region.with_brand_references(
                references_by_brand.get(region.top_brand or "", 0)
            )

            decided.append((region, self._router.route(region)))

        decided = self._verify_uncertain(decided, crops)

        survivors = self._resolver.resolve(decided)
        # Depois do colapso, e nunca antes: caixa que vai ser descartada
        # como recorte interno nao pode servir de testemunha.
        survivors = self._corroborator.resolve(survivors)
        survivors = self._judge_reviews(survivors, crops)
        regions = [self._to_output(region, decision) for region, decision in survivors]
        return AnalysisOutput(path=command.path, discarded_by=None, regions=tuple(regions))

    def _verify_uncertain(
        self,
        decided: list[tuple[AnalyzedRegion, Decision]],
        crops: list[RgbImage],
    ) -> list[tuple[AnalyzedRegion, Decision]]:
        """Roda a verificacao geometrica e redecide as regioes que precisam dela.

        **A camada cara so ve o que a barata nao resolveu.** Esta e a ordem que a
        arquitetura sempre prometeu, e que a implementacao anterior nao cumpria:
        a geometria rodava em toda regiao, inclusive nas que o consenso ja tinha
        decidido com folga e nas que a similaridade ja tinha condenado.

        O custo era o dominante da pipeline. Numa imagem tipica sao ~55 regioes
        e 4 referencias por regiao - 220 comparacoes, a ~87ms cada.

        **O que decide quem e "duvidoso":** a fila que a pontuacao sozinha
        produziu. Aceite e rejeicao ja estao decididos; revisao, confusao e orfao
        nao. Com `only_when_uncertain` desligado, roda em tudo como antes.

        **O custo desta economia, dito claramente:** uma regiao rejeitada pela
        pontuacao nao ganha segunda chance, e o orfao geometrico - a regra que
        acha a referencia que falta - depende justamente de geometria forte com
        similaridade baixa. Por isso o guarda de similaridade: regiao na faixa do
        orfao continua sendo verificada mesmo tendo sido rejeitada.

        Args:
            decided: Pares `(regiao, decisao)` decididos sem geometria.
            crops: Recortes na ordem das deteccoes.

        Returns:
            Os mesmos pares, com as regioes verificadas ja redecididas.
        """
        if not self._config.geometry.enabled:
            return decided

        result: list[tuple[AnalyzedRegion, Decision]] = []
        for index, (region, decision) in enumerate(decided):
            if not region.candidates or not self._needs_geometry(region, decision):
                result.append((region, decision))
                continue
            verified = region.with_verdicts(self._verifier.verify(crops[index], region.candidates))
            result.append((verified, self._router.route(verified)))
        return result

    def _needs_geometry(self, region: AnalyzedRegion, decision: Decision) -> bool:
        """Diz se vale gastar verificacao geometrica nesta regiao.

        Args:
            region: Regiao ja com candidatos.
            decision: Decisao tomada sem geometria.

        Returns:
            True quando a geometria pode mudar o destino da regiao.
        """
        if not self._config.geometry.only_when_uncertain:
            return True
        if decision.queue not in (Queue.AUTO_ACCEPT, Queue.AUTO_REJECT):
            return True
        # Candidata a orfao: o banco reconhece de longe e a geometria e quem
        # decide se ha logo ali. Sem esta excecao, a fila mais valiosa do
        # sistema - a referencia que falta - deixaria de existir.
        return (
            decision.queue is Queue.AUTO_REJECT
            and region.top_similarity >= self._config.geometry.entry_similarity
        )

    def _judge_reviews(
        self,
        survivors: tuple[tuple[AnalyzedRegion, Decision], ...],
        crops: list[RgbImage],
    ) -> tuple[tuple[AnalyzedRegion, Decision], ...]:
        """Pede segunda opiniao apenas sobre o que caiu em revisao.

        **So a fila de revisao**, e por economia: numa imagem tipica sao 7
        regioes contra 50 detectadas, e esta e a unica camada que custa por
        chamada. Julgar tudo multiplicaria a conta por sete sem tocar no que ja
        estava decidido.

        As de maior pontuacao vao primeiro, e o teto corta o resto: numa imagem
        com dezenas de regioes ambiguas, as melhores sao as que mais tem chance
        de virar aceite.

        Args:
            survivors: Pares `(regiao, decisao)` que sobreviveram ao resolvedor.
            crops: Recortes na ordem das deteccoes.

        Returns:
            Os mesmos pares, com as regioes de revisao redecididas quando o juiz
            opinou.
        """
        if self._judge is None:
            return survivors

        pending = sorted(
            (pair for pair in survivors if pair[1].queue is Queue.REVIEW),
            key=lambda pair: pair[1].score,
            reverse=True,
        )[: self._config.judge.max_regions]
        if not pending:
            return survivors

        judged: dict[str, tuple[AnalyzedRegion, Decision]] = {}
        for region, _ in pending:
            candidate = region.candidates[0] if region.candidates else None
            if candidate is None:
                continue
            # O identificador termina no indice da deteccao que gerou a regiao,
            # e e assim que se acha o recorte correspondente. Ver `execute`.
            index = int(region.identifier.rsplit("-", 1)[-1])
            reference = self._source.load(Path(candidate.reference))
            verdict = self._judge.judge(crops[index], reference, candidate.brand)
            # None nao vira decisao: a regiao fica na fila humana, que e
            # exatamente onde ela ja estava. Quem se abstem por duvida tambem
            # devolve None - so o adaptador sabe o que a confianca dele
            # significa, entao e la que essa linha e desenhada.
            if verdict is None:
                continue
            decided = region.with_judgement(verdict)
            judged[region.identifier] = (decided, self._router.route(decided))

        return tuple(
            judged.get(region.identifier, (region, decision)) for region, decision in survivors
        )

    def _discard(self, image: RgbImage) -> str | None:
        """Decide se a imagem nao merece uma passada de detector.

        Deliberadamente permissivo: o recall do detector e o teto do sistema
        inteiro, e o que se descarta aqui nunca mais volta.

        Args:
            image: Imagem carregada.

        Returns:
            Motivo do descarte, ou None se a imagem deve ser processada.
        """
        config = self._config.pre_filter
        if not config.enabled:
            return None

        width, height = self._source.dimensions(image)
        if min(width, height) < config.min_side:
            return f"lado menor que {config.min_side}px - miniatura ou icone"

        density = self._source.edge_density(image)
        if density < config.min_edge_density:
            return (
                f"densidade de bordas {density:.4f} abaixo de "
                f"{config.min_edge_density} - imagem sem estrutura"
            )
        return None

    def _crop_all(self, image: RgbImage, detections: tuple) -> list[RgbImage]:  # type: ignore[type-arg]
        """Recorta todas as regioes detectadas de uma vez.

        Args:
            image: Imagem de origem.
            detections: Regioes encontradas pelo detector.

        Returns:
            Recortes na mesma ordem das deteccoes.
        """
        config = self._config.encoder
        return [
            self._source.crop(
                image,
                detection.box,
                config.crop_margin,
                config.crop_side,
            )
            for detection in detections
        ]

    def _to_output(self, region: AnalyzedRegion, decision: Decision) -> RegionOutput:
        """Converte a regiao analisada e sua decisao no DTO de saida.

        Args:
            region: Regiao com candidatos e vereditos ja reunidos.
            decision: O que o roteador decidiu para ela.

        Returns:
            O DTO correspondente, com os sinais que justificam a decisao.
        """
        verdict = region.best_verdict
        box = region.detection.box
        return RegionOutput(
            identifier=region.identifier,
            box=(box.x1, box.y1, box.x2, box.y2),
            queue=decision.queue,
            brand=decision.brand,
            score=decision.score,
            similarity=region.top_similarity,
            margin=region.margin,
            inliers=verdict.inliers if verdict is not None else 0,
            reasons=decision.reasons,
        )
