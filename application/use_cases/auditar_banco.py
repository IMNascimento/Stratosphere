"""Caso de uso que roda o banco contra ele mesmo procurando marcas confundiveis.

Cada par de marcas **diferentes** com referencias parecidas demais e um falso
positivo agendado, ou um grupo de confusao ainda nao declarado. Barato o
bastante para rodar a cada reconstrucao do banco, e e a unica forma de descobrir
que uma referencia ruim virou ima permanente de falso positivo antes que ela
apareca no relatorio de um cliente.

Typical usage:
    saida = caso.execute()
    for marca_a, marca_b, similaridade in saida.pares_confundiveis:
        ...
"""

from application.dtos.analise import BancoOutput
from config.settings import AppConfig
from domain.repositories.i_banco_referencia import IBancoReferencia


class AuditarBancoUseCase:
    """Lista marcas do banco que se parecem demais entre si."""

    def __init__(self, banco: IBancoReferencia, config: AppConfig) -> None:
        """Recebe o banco e a configuracao de limiares.

        Args:
            banco: Banco de referencia ja carregado.
            config: Parametros de busca, de onde vem o limiar de alerta.
        """
        self._banco = banco
        self._config = config

    def execute(self) -> BancoOutput:
        """Audita o banco e resume sua composicao.

        Returns:
            Resumo com a contagem por marca e os pares confundiveis acima do
            limiar de alerta, em ordem decrescente de similaridade.
        """
        por_marca = self._banco.referencias_por_marca()
        pares = self._banco.pares_confundiveis(self._config.busca.similaridade_de_alerta)
        return BancoOutput(
            total_de_referencias=self._banco.tamanho(),
            referencias_por_marca=por_marca,
            descartadas_por_redundancia=0,
            pares_confundiveis=pares,
        )

    def marcas_com_poucas_referencias(self, minimo: int) -> dict[str, int]:
        """Lista marcas com cobertura fina demais para funcionar bem.

        Marca com poucas referencias tende a produzir orfaos em vez de acertos.
        Saber disso de antemao muda a leitura de qualquer metrica por marca —
        sem isso, o baixo desempenho parece defeito do modelo quando e falta de
        material.

        Args:
            minimo: Abaixo desta contagem a marca e reportada.

        Returns:
            Mapa marca -> contagem, apenas das marcas abaixo do minimo.
        """
        return {
            marca: quantidade
            for marca, quantidade in self._banco.referencias_por_marca().items()
            if quantidade < minimo
        }
