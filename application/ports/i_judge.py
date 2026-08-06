"""Contrato do juiz visual — a segunda opiniao sobre o que ficou em duvida.

Roda **depois** da primeira decisao e **apenas nas regioes que cairam em
revisao**. E a diferenca entre julgar 7 regioes por imagem e julgar 50: numa
base de 5.000 imagens/dia, isso separa dezenas de milhares de chamadas de
centenas de milhares.

A pergunta e sempre de comparacao — recorte contra referencia do banco — e nunca
"que marca e essa?". Ver `JudgeVerdict` para o porque.

**Esta camada e a mais lenta por regiao.** As outras rodam em lote; esta faz uma
passada de VLM por regiao. Por isso o teto em `JudgeConfig.max_regions` e por
isso ela so existe quando ligada explicitamente.

Typical usage:
    verdict = judge.judge(crop, reference, brand="nike")
"""

from abc import ABC, abstractmethod

from application.ports.i_image_source import RgbImage
from domain.value_objects.judge_verdict import JudgeVerdict


class IJudge(ABC):
    """Segunda opiniao visual sobre um par (regiao, referencia de uma marca)."""

    @abstractmethod
    def judge(self, crop: RgbImage, reference: RgbImage, brand: str) -> JudgeVerdict | None:
        """Compara a regiao com a referencia e diz se sao a mesma marca.

        Args:
            crop: Regiao recortada, no mesmo preparo usado na codificacao.
            reference: Imagem de referencia do banco que o candidato apontou.
            brand: Marca que o banco propos, para o juiz confirmar ou negar.

        Returns:
            O parecer, ou **None quando o juiz nao opinou** — por falha (modelo
            que nao carrega, servico fora) ou por duvida.

            **Quando se abster por duvida e decisao do adaptador**, e nao de
            quem chama: so ele sabe o que os proprios numeros significam. Um
            juiz que devolve probabilidade nao calibrada precisa dos cortes
            dele; um que devolve confianca auto-declarada precisa de outro
            criterio. Centralizar isso aqui obrigaria o caso de uso a conhecer
            a escala de cada implementacao.

            A distincao entre `None` e um parecer discordante e a mesma que a
            verificacao geometrica ja faz, e pelo mesmo motivo. Um parecer que
            nao concorda significa "olhei as duas imagens e nao sao a mesma
            marca", e manda a regiao para o descarte. Devolver isso quando
            ninguem olhou faria uma queda de rede **descartar deteccao real em
            silencio** — verificado em teste: sem credencial, sete regioes
            legitimas em revisao viraram rejeicao automatica.

            Implementacoes **nao levantam excecao** por falha de servico: quem
            nao conseguiu opinar devolve None, a regiao fica onde estava, e uma
            pessoa decide.
        """
        ...
