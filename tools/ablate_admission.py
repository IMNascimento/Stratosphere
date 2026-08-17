"""Mede o impacto de uma regra de admissao no RESULTADO FINAL, nao no banco.

--------------------------------------------------------------------------
A PERGUNTA QUE ESTA FERRAMENTA RESPONDE, E QUE `sweep_admission` NAO RESPONDE
--------------------------------------------------------------------------
`sweep_admission.py` diz **quanto** uma regra recusa. E barato e util, mas nao
diz se recusar foi bom: um limiar que remove 30% do banco pode estar removendo
30% de lixo.

Esta ferramenta responde a outra metade. Ela constroi um indice SEM as
referencias que a regra recusaria, roda a pipeline inteira nos dois, e compara
o que sai do outro lado - recall, carga humana e marcas espurias.

E cara: exige GPU e uma passada completa por dataset, por indice. Use em conjunto
pequeno (`cbf_uniforme` tem 256 imagens) para ter sinal em minutos.

--------------------------------------------------------------------------
O RESULTADO QUE JUSTIFICOU ESCREVER ISTO
--------------------------------------------------------------------------
Testando o corte de 48px em `dataset_final/cbf_uniforme/teste`:

                        com as pequenas   com a regra
    recall cbf               29,3%            28,5%
    regioes para humano        40              128
    orfaos                      0               80
    `getv` espurio              0              +72

A regra piora **as duas forcas opostas ao mesmo tempo** - perde recall E
triplica a carga humana - o que e a assinatura de uma regra que remove cobertura
util. Os 80 orfaos sao o sistema dizendo literalmente "tenho esta marca e nao
tenho esta variacao dela". E os 72 `getv` mostram o mecanismo: sem as
referencias pequenas de cbf, o vizinho mais proximo passa a ser outra marca, e a
marca errada entra no relatorio.

Nenhuma contagem de banco mostraria isso. So rodar mostra.

--------------------------------------------------------------------------
O ROTULO
--------------------------------------------------------------------------
`dataset_final/<marca>_<variante>/teste/` diz que toda imagem ali contem aquela
marca. E rotulo de PRESENCA, nao de ausencia: a imagem pode conter outras marcas
legitimamente, e por isso "marcas alem da esperada" e um sinal a interpretar, nao
um erro a contar. Um salto grande numa marca so - como os 72 `getv` - e o padrao
que denuncia substituicao de vizinho.

Typical usage:
    poetry run python tools/ablate_admission.py --regra lado48 \\
        --referencias referencias_v3 --dataset dataset_final/cbf_uniforme/teste \\
        --marca cbf
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from shared.logging.logger import configure_logging, get_logger

log = get_logger("stratosphere.ablate_admission")

_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class Outcome:
    """O que a pipeline produziu sobre um dataset.

    Attributes:
        images: Imagens analisadas.
        found: Imagens em que a marca esperada foi aceita.
        human: Regioes que exigem pessoa.
        orphans: Regioes na fila de orfao - o sinal de cobertura ausente.
        other_brands: Ocorrencias de marcas alem da esperada.
    """

    images: int
    found: int
    human: int
    orphans: int
    other_brands: Counter[str]

    @property
    def recall(self) -> float:
        """Fracao das imagens em que a marca esperada foi encontrada.

        Returns:
            Fracao entre 0.0 e 1.0.
        """
        return self.found / self.images if self.images else 0.0


def main(arguments: list[str] | None = None) -> int:
    """Compara a pipeline com e sem as referencias que uma regra recusaria.

    Args:
        arguments: Argumentos de linha de comando. None usa `sys.argv`.

    Returns:
        0 em sucesso, 1 em erro de entrada.
    """
    options = _parser().parse_args(arguments)
    configure_logging("INFO")

    references = Path(options.referencias)
    dataset = Path(options.dataset)
    if not references.exists() or not dataset.exists():
        log.error("referencias ou dataset ausente")
        return 1

    workspace = Path(options.trabalho)
    workspace.mkdir(parents=True, exist_ok=True)
    filtered = workspace / "referencias_filtradas"
    ablated_index = workspace / "indice_ablado"

    removed = _filter(references, filtered, options.regra, options.valor)
    if not removed:
        log.error("a regra %r nao removeu nenhuma referencia", options.regra)
        return 1
    log.info("regra %r remove %d referencias", options.regra, removed)

    _build(filtered, ablated_index, options.python)
    expected = options.marca.strip().lower()
    baseline = _run(options.indice, dataset, workspace / "base.json", options.python, expected)
    ablated = _run(ablated_index, dataset, workspace / "ablado.json", options.python, expected)

    _report(options, removed, baseline, ablated)
    return 0


def _filter(source: Path, destination: Path, rule: str, value: float) -> int:
    """Copia as referencias que a regra MANTERIA.

    Args:
        source: Arvore original.
        destination: Arvore filtrada, recriada do zero.
        rule: Qual regra aplicar.
        value: Limiar da regra.

    Returns:
        Quantas referencias a regra removeria.
    """
    if destination.exists():
        shutil.rmtree(destination)

    removed = 0
    for path in sorted(source.rglob("*")):
        if not (path.is_file() and path.suffix.lower() in _EXTENSIONS):
            continue
        if _would_reject(path, rule, value):
            removed += 1
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)
    return removed


def _would_reject(path: Path, rule: str, value: float) -> bool:
    """Diz se a regra recusaria esta referencia.

    Args:
        path: Arquivo a avaliar.
        rule: Qual regra aplicar.
        value: Limiar da regra.

    Returns:
        True quando a regra recusaria.
    """
    try:
        image = Image.open(path)
    except OSError:
        return True
    if rule == "lado":
        return min(image.size) < value
    raise ValueError(f"regra desconhecida: {rule!r}")


def _build(references: Path, destination: Path, python: str) -> None:
    """Constroi o indice a partir da arvore filtrada.

    Args:
        references: Arvore filtrada.
        destination: Onde gravar o indice.
        python: Interpretador a usar.
    """
    log.info("construindo indice ablado em %s", destination)
    subprocess.run(  # noqa: S603 - argumentos vem da linha de comando do proprio operador
        [python, "-m", "entrypoints.cli.main", "banco",
         "--referencias", str(references), "--destino", str(destination)],
        check=True,
        capture_output=True,
    )


def _run(
    index: str | Path,
    dataset: Path,
    output: Path,
    python: str,
    expected: str,
) -> Outcome:
    """Roda a pipeline e resume o resultado.

    Args:
        index: Indice a usar.
        dataset: Pasta de imagens rotuladas por presenca.
        output: Onde gravar o json de regioes.
        python: Interpretador a usar.
        expected: Marca que o dataset contem.

    Returns:
        O resumo do que a pipeline produziu.
    """
    log.info("rodando %s sobre %s", index, dataset.name)
    subprocess.run(  # noqa: S603 - argumentos vem da linha de comando do proprio operador
        [python, "-m", "entrypoints.cli.main", "--banco", str(index),
         "analisar", "--entrada", str(dataset), "--saida", str(output)],
        check=True,
        capture_output=True,
    )
    return _summarise(output, expected)


def _summarise(path: Path, expected: str) -> Outcome:
    """Le o json da pipeline e conta o que importa.

    Args:
        path: Arquivo de resultado.
        expected: Marca que o dataset contem.

    Returns:
        O resumo.
    """
    results = json.loads(path.read_text(encoding="utf-8"))
    human = orphans = 0
    others: Counter[str] = Counter()
    found = 0

    for image in results:
        accepted = image.get("marcas_aceitas") or []
        if expected in accepted:
            found += 1
        for brand in accepted:
            if brand != expected:
                others[brand] += 1
        for region in image.get("regioes", []):
            queue = region.get("fila")
            if queue in ("revisao", "confusao", "orfao"):
                human += 1
            if queue == "orfao":
                orphans += 1
    return Outcome(
        images=len(results),
        found=found,
        human=human,
        orphans=orphans,
        other_brands=others,
    )


def _report(
    options: argparse.Namespace,
    removed: int,
    baseline: Outcome,
    ablated: Outcome,
) -> None:
    """Imprime a comparacao.

    Args:
        options: Opcoes de linha de comando.
        removed: Quantas referencias a regra removeu.
        baseline: Resultado com o banco completo.
        ablated: Resultado com o banco filtrado.
    """
    print(f"\n=== regra {options.regra}={options.valor:g} | {removed} referencias removidas ===")
    print(f"    dataset: {options.dataset} ({baseline.images} imagens, marca {options.marca!r})\n")
    print(f"    {'metrica':<28} {'completo':>12} {'com a regra':>13} {'delta':>10}")
    print(f"    {'recall da marca':<28} {baseline.recall:>11.1%} {ablated.recall:>13.1%}"
          f" {(ablated.recall - baseline.recall) * 100:>+9.1f}pt")
    print(f"    {'regioes para humano':<28} {baseline.human:>12} {ablated.human:>13}"
          f" {ablated.human - baseline.human:>+10d}")
    print(f"    {'orfaos':<28} {baseline.orphans:>12} {ablated.orphans:>13}"
          f" {ablated.orphans - baseline.orphans:>+10d}")

    appeared = ablated.other_brands - baseline.other_brands
    if appeared:
        print("\n    marcas que APARECERAM ao remover referencias:")
        for brand, count in appeared.most_common(6):
            print(f"      +{count:<4} {brand}")
        print("    (salto grande numa marca so denuncia substituicao de vizinho:")
        print("     sem a referencia certa, o mais proximo passa a ser outra marca)")

    print()
    if ablated.recall < baseline.recall and ablated.human > baseline.human:
        print("    VEREDITO: a regra piora as DUAS forcas opostas ao mesmo tempo -")
        print("    perde recall E aumenta a carga humana. E a assinatura de uma")
        print("    regra que remove cobertura util, nao lixo.")
    elif ablated.recall >= baseline.recall and ablated.human <= baseline.human:
        print("    VEREDITO: a regra melhora as duas. Removeu lixo.")
    else:
        print("    VEREDITO: troca uma coisa pela outra - decida pelo custo de cada")
        print("    lado no seu contexto.")


def _parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos.

    Returns:
        O parser configurado.
    """
    parser = argparse.ArgumentParser(
        description="mede o impacto de uma regra de admissao no resultado final"
    )
    parser.add_argument("--regra", default="lado", choices=("lado",), help="regra a testar")
    parser.add_argument("--valor", type=float, default=48.0, help="limiar da regra")
    parser.add_argument("--referencias", default="referencias_v3", help="arvore de referencias")
    parser.add_argument("--indice", default="indice", help="indice completo, para o baseline")
    parser.add_argument("--dataset", required=True, help="pasta de imagens rotuladas")
    parser.add_argument("--marca", required=True, help="marca que o dataset contem")
    parser.add_argument("--trabalho", default="runs/ablacao", help="pasta de trabalho")
    parser.add_argument("--python", default=sys.executable, help="interpretador a usar")
    return parser


if __name__ == "__main__":
    sys.exit(main())
