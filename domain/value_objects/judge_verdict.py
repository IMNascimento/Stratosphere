"""Value object do parecer de um juiz visual sobre uma regiao em duvida.

O juiz responde **uma pergunta de comparacao**, nao de memoria: "estas duas
imagens sao a mesma marca?", com o recorte de um lado e a referencia do banco do
outro. A diferenca nao e estilistica.

Perguntar "que marca e essa?" so funciona para marca que o modelo conhece de
treino — nike, amazon, volkswagen. Metade do portfolio deste projeto e
`divino_fogao`, `menzoil`, `souza_lima`, `dryon`, `getv`: marca regional que
nenhum modelo de linguagem viu, e sobre a qual ele vai inventar um nome
plausivel. Comparacao funciona igual para marca famosa e desconhecida, porque a
resposta esta nas duas imagens mostradas.

E preserva a propriedade central do sistema: **o banco continua dizendo qual
marca e**. O juiz so confirma, nega, ou levanta a mao.

Typical usage:
    verdict = JudgeVerdict(agrees=True, brand="nike", confidence=0.9, reason="...")
"""

from dataclasses import dataclass

from domain.exceptions.domain_exceptions import DomainError


@dataclass(frozen=True)
class JudgeVerdict:
    """O que o juiz visual concluiu sobre um par (regiao, referencia).

    Attributes:
        agrees: Se o juiz confirma a marca que o banco propos.
        brand: Marca que o juiz afirma. Igual a do banco quando concorda; outra
            quando discorda e sabe nomear; None quando discorda e nao sabe.
        confidence: Quanta certeza o juiz declara, entre 0.0 e 1.0. **E a
            certeza declarada pelo modelo, nao uma probabilidade calibrada** —
            serve para ordenar casos, nao para virar limiar sozinha.
        reason: Justificativa curta, em uma frase. Vai para os motivos da
            decisao e e o que uma pessoa le na fila de revisao.
    """

    agrees: bool
    brand: str | None
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        """Valida a confianca e a coerencia entre concordar e nomear.

        Raises:
            DomainError: Se a confianca estiver fora de [0, 1], ou se o juiz
                concordar sem dizer com qual marca.
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise DomainError(f"confianca do juiz fora de [0, 1]: {self.confidence}")
        if self.agrees and not self.brand:
            raise DomainError("juiz que concorda precisa dizer com qual marca")

    @property
    def proposes_other_brand(self) -> bool:
        """Indica se o juiz discordou e ainda assim soube nomear outra marca."""
        return not self.agrees and bool(self.brand)
