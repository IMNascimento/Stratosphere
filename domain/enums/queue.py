"""Enum das filas de destino de uma regiao analisada.

A fila e a saida operacional do sistema: ela diz o que acontece com a regiao
depois que a pipeline termina. Nao e um grau de confianca, e um encaminhamento.

Os valores permanecem em portugues de proposito: eles sao o contrato de saida
do sistema - aparecem no json de relatorio e na tela da CLI, e traduzi-los
quebraria quem ja consome esse formato.

Typical usage:
    if decision.queue is Queue.ORPHAN:
        promote_to_database(decision)
"""

from enum import Enum


class Queue(Enum):
    """Destino operacional de uma regiao analisada.

    Attributes:
        AUTO_ACCEPT: Evidencia suficiente. Entra no relatorio sem humano.
        REVIEW: Ha um palpite e nao ha confianca. Vai para pessoa decidir.
        CONFUSION: Empate entre marcas do mesmo grupo declarado. Desempate
            obrigatorio, com as referencias lado a lado.
        ORPHAN: Ha logo e o banco nao reconheceu. **E a referencia que falta** -
            o sinal que alimenta o banco de volta.
        NEGATIVE: E logo de marca fora do portfolio. Acerto, nao rejeicao:
            contabilizar como rejeicao mascara a qualidade real do sistema.
        AUTO_REJECT: Nao ha evidencia de marca.
    """

    AUTO_ACCEPT = "auto_aceite"
    REVIEW = "revisao"
    CONFUSION = "confusao"
    ORPHAN = "orfao"
    NEGATIVE = "negativa"
    AUTO_REJECT = "auto_rejeicao"

    @property
    def requires_human(self) -> bool:
        """Indica se a fila consome capacidade de revisao humana.

        Returns:
            True para as filas que geram trabalho manual. E este conjunto que
            define o custo operacional do sistema, nao o total de regioes.
        """
        return self in (Queue.REVIEW, Queue.CONFUSION, Queue.ORPHAN)
