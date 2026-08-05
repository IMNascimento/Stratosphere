"""O servico que decide o destino de cada regiao — o nucleo do sistema.

E dominio puro: recebe tudo por parametro, nao faz I/O, nao conhece detector nem
codificador. Isso e proposital. A decisao "aceito sozinho / mando para humano /
registro como referencia faltante" e politica de negocio, vale igual com qualquer
tecnologia plugada, e precisa ser verificavel sem GPU.

--------------------------------------------------------------------------
A ORDEM DAS REGRAS E A POLITICA
--------------------------------------------------------------------------
1. **Sem candidatos** — o banco nao respondeu nada.
2. **Marca negativa** — logo de quem nao interessa. Fila propria, porque contar
   como rejeicao mascara a qualidade real do sistema.
3. **Grupo de confusao** — empate entre marcas do mesmo grupo. Politica de
   negocio: forca desempate mesmo com similaridade alta.
4. **Orfao geometrico** — a geometria confirma e o banco nao reconhece.
5. **Limiares** — aceite, revisao ou rejeicao.

Reordenar isso muda o comportamento do produto, nao so do codigo. Duas
precedencias que parecem detalhe e nao sao:

- **Negativa e confusao vem ANTES do orfao geometrico.** O orfao geometrico ja
  carrega uma marca confiavel, entao e questao de evidencia; negativa e confusao
  sao classificacao e politica. Invertido, um empate gatorade x powerade com
  geometria forte viraria orfao e escaparia do desempate obrigatorio.
- **O empate so vale com evidencia.** Sem esse guarda, ruido puro cujos dois
  melhores palpites sao lixo empatado — e ambos do mesmo setor, que e justamente
  o que o codificador aproxima quando NAO ha sinal — inunda a fila humana.

--------------------------------------------------------------------------
POR QUE OS PESOS SAO RENORMALIZADOS
--------------------------------------------------------------------------
A verificacao geometrica **nao opina em toda regiao**: logo chapado, pequeno ou
vetorial demais nao tem pontos para casar. Com peso fixo, essas regioes sao
punidas por uma evidencia que nunca teve chance de existir — o teto da pontuacao
cai e o limiar de aceite fica inalcancavel para uma classe inteira de casos.
Quando nao ha veredito, os demais pesos sao renormalizados para somar 1.

Typical usage:
    roteador = RoteadorDeFila(pesos=..., calibragem=..., grupos=...)
    decisao = roteador.rotear(regiao)
"""

from dataclasses import dataclass

from domain.entities.decisao import Decisao
from domain.entities.regiao_analisada import RegiaoAnalisada
from domain.enums.fila import Fila
from domain.services.grupos_de_confusao import GruposDeConfusao

_TERMO_SIMILARIDADE = "similaridade"
_TERMO_MARGEM = "margem"
_TERMO_GEOMETRIA = "geometria"
_TERMO_DETECCAO = "deteccao"


@dataclass(frozen=True)
class PesosDeEvidencia:
    """Quanto cada sinal vale na pontuacao final.

    Attributes:
        similaridade: Peso do quanto a melhor referencia se parece com a regiao.
        margem: Peso do quanto a marca escolhida supera a rival mais proxima.
        geometria: Peso da confirmacao de que e o mesmo desenho.
        deteccao: Peso da confianca do detector. Costuma ser pequeno: a escala
            varia por detector e raramente separa logo de fundo.
    """

    similaridade: float
    margem: float
    geometria: float
    deteccao: float

    def __post_init__(self) -> None:
        """Valida pesos nao negativos que somem 1.

        Raises:
            ValueError: Se algum peso for negativo ou a soma nao for 1.
        """
        valores = (self.similaridade, self.margem, self.geometria, self.deteccao)
        if any(peso < 0 for peso in valores):
            raise ValueError(f"pesos nao podem ser negativos: {valores}")
        if abs(sum(valores) - 1.0) > 1e-6:
            raise ValueError(f"pesos devem somar 1.0, somam {sum(valores)}")


@dataclass(frozen=True)
class Calibragem:
    """Como converter cada sinal bruto para a escala [0, 1].

    Estes valores dependem do codificador e do detector em uso. **Nao ha default
    universal** — a distribuicao de similaridade de um codificador nao vale para
    outro, e um limiar importado falha em silencio.

    Attributes:
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

    similaridade_minima: float
    similaridade_maxima: float
    margem_confiante: float
    inliers_confiantes: float
    aceite: float
    rejeicao: float
    orfao_inliers_minimos: int
    orfao_similaridade_maxima: float

    def __post_init__(self) -> None:
        """Valida a coerencia entre os limiares.

        Raises:
            ValueError: Se algum intervalo estiver invertido ou degenerado.
        """
        if self.similaridade_maxima <= self.similaridade_minima:
            raise ValueError(
                f"similaridade_maxima deve superar a minima: "
                f"{self.similaridade_maxima} <= {self.similaridade_minima}"
            )
        if self.aceite <= self.rejeicao:
            raise ValueError(f"aceite deve superar rejeicao: {self.aceite} <= {self.rejeicao}")
        if self.margem_confiante <= 0 or self.inliers_confiantes <= 0:
            raise ValueError("margem_confiante e inliers_confiantes devem ser positivos")


def _entre_zero_e_um(valor: float) -> float:
    """Prende um valor ao intervalo [0, 1].

    Args:
        valor: Numero a limitar.

    Returns:
        O proprio valor, ou a borda mais proxima se estiver fora.
    """
    return max(0.0, min(1.0, valor))


class RoteadorDeFila:
    """Decide a fila de destino de uma regiao a partir das evidencias reunidas.

    Servico de dominio puro: sem I/O, sem estado entre chamadas, sem dependencia
    de framework. Duas chamadas com a mesma regiao produzem a mesma decisao.

    Attributes:
        pesos: Quanto cada sinal vale na pontuacao.
        calibragem: Como converter sinal bruto para escala [0, 1].
    """

    def __init__(
        self,
        pesos: PesosDeEvidencia,
        calibragem: Calibragem,
        grupos: GruposDeConfusao,
    ) -> None:
        """Inicializa o roteador com pesos, calibragem e politica de marcas.

        Args:
            pesos: Contribuicao de cada sinal para a pontuacao.
            calibragem: Limiares de conversao e de decisao.
            grupos: Politica de grupos de confusao e marcas negativas.
        """
        self.pesos = pesos
        self.calibragem = calibragem
        self._grupos = grupos

    def rotear(self, regiao: RegiaoAnalisada) -> Decisao:
        """Decide o destino da regiao aplicando as regras em ordem.

        Args:
            regiao: Regiao com deteccao, candidatos do banco e vereditos
                geometricos ja reunidos.

        Returns:
            A decisao, com a fila escolhida, a marca afirmada quando houver, e
            os motivos e contribuicoes que a produziram.
        """
        sem_candidatos = self._rotear_sem_candidatos(regiao)
        if sem_candidatos is not None:
            return sem_candidatos

        pontuacao, contribuicoes = self._pontuar(regiao)
        marca = regiao.marca_topo

        negativa = self._rotear_negativa(regiao, pontuacao, contribuicoes)
        if negativa is not None:
            return negativa

        confusao = self._rotear_confusao(regiao, pontuacao, contribuicoes)
        if confusao is not None:
            return confusao

        orfao = self._rotear_orfao_geometrico(regiao, pontuacao, contribuicoes)
        if orfao is not None:
            return orfao

        return self._rotear_por_limiar(regiao, marca, pontuacao, contribuicoes)

    # -- regras, na ordem em que sao aplicadas -----------------------------

    def _rotear_sem_candidatos(self, regiao: RegiaoAnalisada) -> Decisao | None:
        """Decide quando o banco nao devolveu candidato algum.

        Args:
            regiao: Regiao analisada.

        Returns:
            A decisao, ou None se houver candidatos e a regra nao se aplicar.
        """
        if regiao.candidatos:
            return None
        return Decisao(
            fila=Fila.AUTO_REJEICAO,
            marca=None,
            pontuacao=0.0,
            motivos=("o banco de referencia nao devolveu candidato algum",),
        )

    def _rotear_negativa(
        self,
        regiao: RegiaoAnalisada,
        pontuacao: float,
        contribuicoes: dict[str, float],
    ) -> Decisao | None:
        """Decide quando a melhor resposta e marca fora do portfolio.

        Args:
            regiao: Regiao analisada.
            pontuacao: Pontuacao ja calculada.
            contribuicoes: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        marca = regiao.marca_topo
        if not self._grupos.e_negativa(marca) or pontuacao < self.calibragem.rejeicao:
            return None
        return Decisao(
            fila=Fila.NEGATIVA,
            marca=marca,
            pontuacao=pontuacao,
            motivos=(
                f"{marca!r} esta no banco como concorrente, fora do portfolio. "
                "Contabilizar separado de 'sem logo' — aqui o sistema acertou.",
            ),
            contribuicoes=contribuicoes,
        )

    def _rotear_confusao(
        self,
        regiao: RegiaoAnalisada,
        pontuacao: float,
        contribuicoes: dict[str, float],
    ) -> Decisao | None:
        """Decide quando ha empate entre marcas do mesmo grupo de confusao.

        O guarda de evidencia (`pontuacao >= rejeicao`) nao e opcional: sem ele,
        ruido puro com dois palpites empatados do mesmo setor inunda a fila
        humana e destroi a estimativa de capacidade.

        Args:
            regiao: Regiao analisada.
            pontuacao: Pontuacao ja calculada.
            contribuicoes: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        marca = regiao.marca_topo
        rival = regiao.marca_rival
        houve_empate = regiao.margem < self._grupos.margem_de_empate
        if (
            pontuacao < self.calibragem.rejeicao
            or not self._grupos.mesmo_grupo(marca, rival)
            or not houve_empate
        ):
            return None
        return Decisao(
            fila=Fila.CONFUSAO,
            marca=marca,
            pontuacao=pontuacao,
            motivos=(
                f"{marca!r} e {rival!r} pertencem ao grupo "
                f"{self._grupos.grupo_de(marca)} e a margem e {regiao.margem:.3f}. "
                "Desempate obrigatorio com as referencias lado a lado.",
            ),
            contribuicoes=contribuicoes,
        )

    def _rotear_orfao_geometrico(
        self,
        regiao: RegiaoAnalisada,
        pontuacao: float,
        contribuicoes: dict[str, float],
    ) -> Decisao | None:
        """Decide quando a geometria confirma e o banco nao reconhece.

        Traducao do sinal: o banco tem a marca e **nao tem esta variacao dela**.
        E a referencia que falta — o insumo que alimenta o banco de volta.

        Args:
            regiao: Regiao analisada.
            pontuacao: Pontuacao ja calculada.
            contribuicoes: Contribuicao de cada termo.

        Returns:
            A decisao, ou None se a regra nao se aplicar.
        """
        veredito = regiao.veredito_confirmado
        if veredito is None:
            return None
        if veredito.inliers < self.calibragem.orfao_inliers_minimos:
            return None
        if regiao.similaridade_topo >= self.calibragem.orfao_similaridade_maxima:
            return None
        return Decisao(
            fila=Fila.ORFAO,
            marca=veredito.marca,
            pontuacao=pontuacao,
            motivos=(
                f"a geometria confirma {veredito.marca!r} com {veredito.inliers} "
                f"inliers, mas a melhor similaridade do banco e "
                f"{regiao.similaridade_topo:.3f}. O banco tem a marca e nao tem "
                "esta variacao — promova esta regiao para o banco.",
            ),
            contribuicoes=contribuicoes,
        )

    def _rotear_por_limiar(
        self,
        regiao: RegiaoAnalisada,
        marca: str | None,
        pontuacao: float,
        contribuicoes: dict[str, float],
    ) -> Decisao:
        """Decide pelos limiares de aceite e rejeicao.

        Args:
            regiao: Regiao analisada.
            marca: Marca do melhor candidato.
            pontuacao: Pontuacao ja calculada.
            contribuicoes: Contribuicao de cada termo.

        Returns:
            A decisao final. Regiao rejeitada nunca afirma marca.
        """
        motivos: list[str] = []
        veredito = regiao.melhor_veredito
        if veredito is not None:
            motivos.append(
                f"geometria: {veredito.inliers} inliers de "
                f"{veredito.correspondencias} correspondencias vs {veredito.marca!r}"
            )
        else:
            motivos.append(
                "geometria nao opinou — sem pontos suficientes para casar. "
                "Os demais pesos foram renormalizados."
            )
        motivos.append(
            f"pontuacao={pontuacao:.3f} "
            f"(similaridade={regiao.similaridade_topo:.3f} "
            f"margem={regiao.margem:.3f})"
        )

        if pontuacao >= self.calibragem.aceite:
            motivos.append(f"pontuacao >= aceite ({self.calibragem.aceite})")
            return Decisao(
                fila=Fila.AUTO_ACEITE,
                marca=marca,
                pontuacao=pontuacao,
                motivos=tuple(motivos),
                contribuicoes=contribuicoes,
            )

        if pontuacao < self.calibragem.rejeicao:
            motivos.append(f"pontuacao < rejeicao ({self.calibragem.rejeicao})")
            return Decisao(
                fila=Fila.AUTO_REJEICAO,
                marca=None,
                pontuacao=pontuacao,
                motivos=tuple(motivos),
                contribuicoes=contribuicoes,
            )

        motivos.append(
            f"faixa ambigua [{self.calibragem.rejeicao}, {self.calibragem.aceite}) "
            "— vai para conferencia humana"
        )
        return Decisao(
            fila=Fila.REVISAO,
            marca=marca,
            pontuacao=pontuacao,
            motivos=tuple(motivos),
            contribuicoes=contribuicoes,
        )

    # -- pontuacao ---------------------------------------------------------

    def _pontuar(self, regiao: RegiaoAnalisada) -> tuple[float, dict[str, float]]:
        """Converte os sinais em uma pontuacao unica entre 0 e 1.

        Quando a verificacao geometrica nao opinou, o peso dela e redistribuido
        entre os demais em vez de contar como zero. Contar como zero puniria a
        regiao por uma evidencia que nunca teve chance de existir, e derrubaria
        o teto da pontuacao abaixo do limiar de aceite.

        Args:
            regiao: Regiao com candidatos e vereditos.

        Returns:
            Tupla `(pontuacao, contribuicoes)`, onde as contribuicoes somam a
            pontuacao e permitem auditar a decisao sem reexecutar nada.
        """
        calibragem = self.calibragem
        amplitude = calibragem.similaridade_maxima - calibragem.similaridade_minima

        normalizados = {
            _TERMO_SIMILARIDADE: _entre_zero_e_um(
                (regiao.similaridade_topo - calibragem.similaridade_minima) / amplitude
            ),
            _TERMO_MARGEM: _entre_zero_e_um(regiao.margem / calibragem.margem_confiante),
            _TERMO_DETECCAO: _entre_zero_e_um(regiao.deteccao.confianca),
        }
        pesos = {
            _TERMO_SIMILARIDADE: self.pesos.similaridade,
            _TERMO_MARGEM: self.pesos.margem,
            _TERMO_DETECCAO: self.pesos.deteccao,
        }

        veredito = regiao.melhor_veredito
        if veredito is not None:
            normalizados[_TERMO_GEOMETRIA] = _entre_zero_e_um(
                veredito.inliers / calibragem.inliers_confiantes
            )
            pesos[_TERMO_GEOMETRIA] = self.pesos.geometria
        else:
            pesos = self._renormalizar(pesos)

        contribuicoes = {termo: round(pesos[termo] * normalizados[termo], 4) for termo in pesos}
        return _entre_zero_e_um(sum(contribuicoes.values())), contribuicoes

    @staticmethod
    def _renormalizar(pesos: dict[str, float]) -> dict[str, float]:
        """Reescala os pesos restantes para somarem 1.

        Args:
            pesos: Pesos dos termos que produziram evidencia.

        Returns:
            Os mesmos termos com pesos proporcionais somando 1. Se a soma for
            zero, devolve os pesos como estao — nao ha o que reescalar.
        """
        total = sum(pesos.values())
        if total <= 0:
            return pesos
        return {termo: peso / total for termo, peso in pesos.items()}
