"""Enum das filas de destino de uma regiao analisada.

A fila e a saida operacional do sistema: ela diz o que acontece com a regiao
depois que a pipeline termina. Nao e um grau de confianca, e um encaminhamento.

Typical usage:
    if decisao.fila is Fila.ORFAO:
        promover_para_banco(decisao)
"""

from enum import Enum


class Fila(Enum):
    """Destino operacional de uma regiao analisada.

    Attributes:
        AUTO_ACEITE: Evidencia suficiente. Entra no relatorio sem humano.
        REVISAO: Ha um palpite e nao ha confianca. Vai para pessoa decidir.
        CONFUSAO: Empate entre marcas do mesmo grupo declarado. Desempate
            obrigatorio, com as referencias lado a lado.
        ORFAO: Ha logo e o banco nao reconheceu. **E a referencia que falta** —
            o sinal que alimenta o banco de volta.
        NEGATIVA: E logo de marca fora do portfolio. Acerto, nao rejeicao:
            contabilizar como rejeicao mascara a qualidade real do sistema.
        AUTO_REJEICAO: Nao ha evidencia de marca.
    """

    AUTO_ACEITE = "auto_aceite"
    REVISAO = "revisao"
    CONFUSAO = "confusao"
    ORFAO = "orfao"
    NEGATIVA = "negativa"
    AUTO_REJEICAO = "auto_rejeicao"

    @property
    def exige_humano(self) -> bool:
        """Indica se a fila consome capacidade de revisao humana.

        Returns:
            True para as filas que geram trabalho manual. E este conjunto que
            define o custo operacional do sistema, nao o total de regioes.
        """
        return self in (Fila.REVISAO, Fila.CONFUSAO, Fila.ORFAO)
