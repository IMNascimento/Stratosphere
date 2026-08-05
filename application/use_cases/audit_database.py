"""Caso de uso que roda o banco contra ele mesmo procurando marcas confundiveis.

Cada par de marcas **diferentes** com referencias parecidas demais e um falso
positivo agendado, ou um grupo de confusao ainda nao declarado. Barato o
bastante para rodar a cada reconstrucao do banco, e e a unica forma de descobrir
que uma referencia ruim virou ima permanente de falso positivo antes que ela
apareca no relatorio de um cliente.

Typical usage:
    output = use_case.execute()
    for brand_a, brand_b, similarity in output.confusable_pairs:
        ...
"""

from application.dtos.analysis import DatabaseOutput
from config.settings import AppConfig
from domain.repositories.i_reference_database import IReferenceDatabase


class AuditDatabaseUseCase:
    """Lista marcas do banco que se parecem demais entre si."""

    def __init__(self, database: IReferenceDatabase, config: AppConfig) -> None:
        """Recebe o banco e a configuracao de limiares.

        Args:
            database: Banco de referencia ja carregado.
            config: Parametros de busca, de onde vem o limiar de alerta.
        """
        self._database = database
        self._config = config

    def execute(self) -> DatabaseOutput:
        """Audita o banco e resume sua composicao.

        Returns:
            Resumo com a contagem por marca e os pares confundiveis acima do
            limiar de alerta, em ordem decrescente de similaridade.
        """
        by_brand = self._database.references_by_brand()
        pairs = self._database.confusable_pairs(self._config.search.alert_similarity)
        return DatabaseOutput(
            total_references=self._database.size(),
            references_by_brand=by_brand,
            discarded_by_redundancy=0,
            confusable_pairs=pairs,
        )

    def brands_with_few_references(self, minimum: int) -> dict[str, int]:
        """Lista marcas com cobertura fina demais para funcionar bem.

        Marca com poucas referencias tende a produzir orfaos em vez de acertos.
        Saber disso de antemao muda a leitura de qualquer metrica por marca —
        sem isso, o baixo desempenho parece defeito do modelo quando e falta de
        material.

        Args:
            minimum: Abaixo desta contagem a marca e reportada.

        Returns:
            Mapa marca -> contagem, apenas das marcas abaixo do minimo.
        """
        return {
            brand: quantity
            for brand, quantity in self._database.references_by_brand().items()
            if quantity < minimum
        }
