"""Contrato da verificacao geometrica - "e literalmente o mesmo desenho?".

Pergunta diferente da que a busca vetorial responde. A busca diz "parece com";
esta camada diz "e o mesmo desenho, sob alguma transformacao coerente".

As duas convivem porque **falham de formas diferentes**: a busca vetorial e
fraca justamente em mudanca de ponto de vista, e o casamento de pontos sob
transformacao foi projetado para isso. Em compensacao, esta camada fica muda em
logo chapado, pequeno ou vetorial demais para ter cantos - e **ausencia de
veredito e diferente de veredito negativo**, uma distincao que o roteador trata
explicitamente.

Typical usage:
    verdicts = verifier.verify(crop, candidates)
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from application.ports.i_image_source import RgbImage
from domain.value_objects.candidate import Candidate
from domain.value_objects.geometric_verdict import GeometricVerdict


class IGeometricVerifier(ABC):
    """Confirma se uma regiao e o mesmo desenho de alguma referencia."""

    @abstractmethod
    def verify(
        self, crop: RgbImage, candidates: Sequence[Candidate]
    ) -> tuple[GeometricVerdict, ...]:
        """Compara a regiao com as referencias dos candidatos mais promissores.

        Args:
            crop: Regiao recortada, no mesmo preparo usado na codificacao.
            candidates: Candidatos do banco, ordenados por similaridade. A
                implementacao seleciona quais verificar - no maximo um por
                marca, e apenas acima do portao de similaridade configurado. O
                objetivo e decidir ENTRE marcas, nao gastar comparacao em varias
                fotos da mesma.

        Returns:
            Um veredito por referencia efetivamente comparada. **Tupla vazia e
            resultado valido** e significa "nao deu para opinar", nao "nao e
            nenhuma delas".
        """
        ...
