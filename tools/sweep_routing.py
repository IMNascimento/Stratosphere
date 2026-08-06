"""Explora configuracoes de roteamento sobre um dump de sinais, sem tocar na GPU.

Le o `.jsonl` do `dump_signals.py`, reconstroi as regioes e roda o **`QueueRouter`
de verdade** — o mesmo objeto que a pipeline usa. Nenhuma regra e reimplementada
aqui: se o roteador mudar, esta ferramenta muda junto sozinha.

--------------------------------------------------------------------------
AS TRES METRICAS, E POR QUE SAO ESSAS
--------------------------------------------------------------------------
O objetivo declarado do projeto e "maxima precisao com minima revisao humana".
Isso sao duas forcas opostas, e uma terceira que costuma ser esquecida:

- **precisao do aceite** — das regioes aceitas sem humano, quantas acertaram a
  marca. E o numero que nao pode cair: aceite errado vai para o relatorio do
  cliente sem ninguem ver.
- **volume de revisao** — quantas regioes exigem pessoa. E o custo.
- **perda por rejeicao** — regioes cuja marca do topo estava CERTA e que foram
  descartadas assim mesmo. E o custo invisivel: nao aparece em nenhuma metrica
  de precisao, e some do relatorio sem deixar rastro.

Otimizar as duas primeiras ignorando a terceira produz um sistema que parece
excelente e nao encontra nada.

--------------------------------------------------------------------------
O ROTULO E POR IMAGEM, NAO POR CAIXA
--------------------------------------------------------------------------
`<raiz>/<marca>/arquivo.jpg` diz que a imagem contem aquela marca — nao que toda
caixa dentro dela seja daquela marca. Num backdrop com varios patrocinadores, as
outras caixas contam como erro sem serem erro.

Isso torna a precisao medida um **piso pessimista**, e e util assim: comparar
configuracoes entre si funciona, porque o vies e o mesmo em todas. Ler o numero
absoluto como "a precisao do sistema" nao funciona.

Typical usage:
    poetry run python tools/sweep_routing.py --sinais sinais.jsonl
    poetry run python tools/sweep_routing.py --sinais a.jsonl --comparar b.jsonl
"""

import argparse
import json
import sys
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

from config.settings import AppConfig, RoutingConfig
from domain.entities.analyzed_region import AnalyzedRegion
from domain.entities.decision import Decision
from domain.entities.detection import Detection
from domain.enums.queue import Queue
from domain.services.nested_region_resolver import NestedRegionResolver
from domain.value_objects.box import Box
from domain.value_objects.candidate import Candidate
from domain.value_objects.geometric_verdict import GeometricVerdict
from infrastructure.container.container import _build_router
from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere.sweep_routing")

_HUMAN = (Queue.REVIEW, Queue.CONFUSION, Queue.ORPHAN)


@dataclass(frozen=True)
class Result:
    """Resultado de uma configuracao sobre o conjunto inteiro.

    Attributes:
        accepted: Regioes aceitas sem humano.
        accepted_right: Dessas, quantas acertaram a marca.
        human: Regioes que exigem pessoa.
        lost: Regioes com a marca certa no topo que foram rejeitadas.
        recoverable: Regioes com a marca certa no topo — o teto do que da para
            aceitar corretamente.
        total: Regioes avaliadas.
    """

    accepted: int
    accepted_right: int
    human: int
    lost: int
    recoverable: int
    total: int

    @property
    def precision(self) -> float:
        """Fracao dos aceites que acertaram a marca.

        Returns:
            Entre 0.0 e 1.0. Vale 1.0 quando nada foi aceito — nenhum erro
            cometido —, e por isso ela nunca e lida sozinha.
        """
        return self.accepted_right / self.accepted if self.accepted else 1.0

    @property
    def recall(self) -> float:
        """Fracao do recuperavel que virou aceite correto.

        Returns:
            Entre 0.0 e 1.0.
        """
        return self.accepted_right / self.recoverable if self.recoverable else 0.0

    @property
    def human_share(self) -> float:
        """Fracao das regioes que exige pessoa.

        Returns:
            Entre 0.0 e 1.0.
        """
        return self.human / self.total if self.total else 0.0


def main(arguments: list[str] | None = None) -> int:
    """Roda a varredura e imprime a comparacao.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 em sucesso, 1 quando o arquivo de sinais nao existe.
    """
    options = _parser().parse_args(arguments)
    configure_logging("WARNING")

    files = [Path(options.sinais)] + [Path(p) for p in options.comparar]
    for path in files:
        if not path.is_file():
            log.error("arquivo de sinais nao existe: %s", path)
            return 1

    base = AppConfig()
    for path in files:
        records = _load(path)
        print(f"\n=== {path.name} — {len(records)} regioes ===")
        _report_header()
        _report("configuracao atual", _evaluate(records, base))

        if options.varredura:
            for routing in _variants(base.routing):
                config = replace(base, routing=routing)
                label = (
                    f"inl={routing.confident_inliers:.0f} "
                    f"aceite={routing.accept:.2f} "
                    f"cons={routing.consensus_accept:.2f}"
                )
                _report(label, _evaluate(records, config))
    return 0


def _load(path: Path) -> list[dict[str, Any]]:
    """Le o dump de sinais.

    Args:
        path: Arquivo `.jsonl`.

    Returns:
        Um dicionario por regiao.
    """
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _evaluate(records: list[dict[str, Any]], config: AppConfig) -> Result:
    """Roteia todas as regioes com uma configuracao e conta os desfechos.

    O resolvedor de aninhamento roda tambem, porque ele muda o que chega ao
    relatorio — medir sem ele mediria uma pipeline que nao existe.

    Args:
        records: Regioes do dump.
        config: Configuracao a avaliar.

    Returns:
        O resultado agregado.
    """
    router = _build_router(config)
    resolver = NestedRegionResolver(containment=config.routing.nested_containment)

    by_image: dict[str, list[tuple[AnalyzedRegion, Decision, str]]] = {}
    for record in records:
        region = _rebuild(record)
        by_image.setdefault(record["image"], []).append(
            (region, router.route(region), record["expected"])
        )

    accepted = accepted_right = human = lost = recoverable = total = 0
    for pairs in by_image.values():
        expected_by_id = {region.identifier: expected for region, _, expected in pairs}
        survivors = resolver.resolve([(region, decision) for region, decision, _ in pairs])
        for region, decision in survivors:
            expected = expected_by_id[region.identifier]
            right = region.top_brand == expected
            total += 1
            recoverable += int(right)
            if decision.queue is Queue.AUTO_ACCEPT:
                accepted += 1
                accepted_right += int(decision.brand == expected)
            elif decision.queue in _HUMAN:
                human += 1
            elif decision.queue is Queue.AUTO_REJECT and right:
                lost += 1

    return Result(accepted, accepted_right, human, lost, recoverable, total)


def _rebuild(record: dict[str, Any]) -> AnalyzedRegion:
    """Reconstroi a regiao a partir do registro gravado.

    Args:
        record: Uma linha do dump.

    Returns:
        A regiao com candidatos, vereditos e contagem de referencias.
    """
    x1, y1, x2, y2 = record["box"]
    region = AnalyzedRegion(
        identifier=f"{Path(record['image']).stem}-{record['index']}",
        detection=Detection(box=Box(x1, y1, x2, y2), confidence=record["detection_score"]),
    )
    region = region.with_candidates(
        tuple(
            Candidate(brand=c["brand"], similarity=c["similarity"], reference=c["reference"])
            for c in record["candidates"]
        )
    )
    region = region.with_brand_references(record["brand_references"])
    if record["verdicts"]:
        region = region.with_verdicts(
            tuple(
                GeometricVerdict(
                    brand=v["brand"],
                    inliers=v["inliers"],
                    matches=v["matches"],
                    confirms=v["confirms"],
                    reason=v["reason"],
                )
                for v in record["verdicts"]
            )
        )
    return region


def _variants(routing: RoutingConfig) -> list[RoutingConfig]:
    """Gera as configuracoes da varredura.

    Varre os tres parametros que a troca de matcher desregulou: o patamar de
    inliers considerado confiante, o limiar de aceite e o corte do consenso.

    Args:
        routing: Configuracao base.

    Returns:
        As variantes, sem repetir a base.
    """
    inliers = (10.0, 20.0, 40.0, 60.0, 90.0)
    accepts = (0.55, 0.60, 0.65, 0.70)
    consensus = (0.70, 0.80, 0.90)
    return [
        replace(routing, confident_inliers=i, accept=a, consensus_accept=c)
        for i, a, c in product(inliers, accepts, consensus)
    ]


def _report_header() -> None:
    """Imprime o cabecalho da tabela."""
    print(
        f"  {'configuracao':<34} {'precisao':>9} {'recall':>7} {'revisao':>8} "
        f"{'aceites':>8} {'perdidas':>9}"
    )


def _report(label: str, result: Result) -> None:
    """Imprime uma linha de resultado.

    Args:
        label: Nome da configuracao.
        result: Resultado medido.
    """
    print(
        f"  {label:<34} {result.precision:>8.1%} {result.recall:>7.1%} "
        f"{result.human_share:>7.1%} {result.accepted:>8} {result.lost:>9}"
    )


def _parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos.

    Returns:
        O parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="explora configuracoes de roteamento sobre um dump de sinais"
    )
    parser.add_argument("--sinais", required=True, help="arquivo .jsonl do dump_signals")
    parser.add_argument("--comparar", nargs="*", default=[], help="outros dumps a medir junto")
    parser.add_argument("--varredura", action="store_true", help="varre limiares alem da base")
    return parser


if __name__ == "__main__":
    sys.exit(main())
