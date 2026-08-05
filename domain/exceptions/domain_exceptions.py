"""Hierarquia de excecoes de dominio do Stratosphere.

Toda violacao de invariante de negocio levanta uma excecao desta hierarquia,
nunca `Exception` generica. Isso permite que a borda do sistema distinga erro de
dominio (dado invalido, contrato violado) de erro de infraestrutura (GPU
indisponivel, arquivo corrompido) sem inspecionar mensagem.

Typical usage:
    from domain.exceptions.domain_exceptions import CaixaInvalidaError

    if x2 <= x1:
        raise CaixaInvalidaError(f"x2 deve ser maior que x1: {x2} <= {x1}")
"""


class DomainError(Exception):
    """Classe base de todas as excecoes de dominio."""


class CaixaInvalidaError(DomainError):
    """Coordenadas de caixa violam a invariante de retangulo valido."""


class CandidatoInvalidoError(DomainError):
    """Candidato do banco de referencia com marca vazia ou similaridade fora de [-1, 1]."""


class DeteccaoInvalidaError(DomainError):
    """Deteccao com confianca fora do intervalo [0, 1]."""


class DimensaoIncompativelError(DomainError):
    """Vetor de consulta tem dimensao diferente da do banco de referencia.

    Levantada em vez de degradar silenciosamente: uma consulta com dimensao
    errada nao produz erro numerico, produz numeros sem significado.
    """

    def __init__(self, dimensao_consulta: int, dimensao_banco: int) -> None:
        """Inicializa a excecao com as duas dimensoes envolvidas.

        Args:
            dimensao_consulta: Dimensao do vetor recebido na busca.
            dimensao_banco: Dimensao dos vetores armazenados no banco.
        """
        super().__init__(
            f"dimensao incompativel: consulta D={dimensao_consulta} "
            f"vs banco D={dimensao_banco}. O banco foi construido com outro "
            f"codificador — reconstrua o indice."
        )


class CodificadorIncompativelError(DomainError):
    """Banco construido com um codificador e consultado com outro.

    Sem esta checagem o sistema roda, devolve similaridades e monta relatorio —
    e tudo esta errado, porque os dois espacos vetoriais nao tem relacao nenhuma.
    E o tipo de defeito que consome semanas ate ser percebido.
    """

    def __init__(self, assinatura_banco: str, assinatura_atual: str) -> None:
        """Inicializa a excecao com as duas assinaturas de codificador.

        Args:
            assinatura_banco: Assinatura registrada no manifesto do banco.
            assinatura_atual: Assinatura do codificador em uso na execucao.
        """
        super().__init__(
            f"banco construido com codificador {assinatura_banco!r} mas a "
            f"execucao usa {assinatura_atual!r}. Espacos vetoriais diferentes — "
            f"reconstrua o indice."
        )


class BancoVazioError(DomainError):
    """Operacao exige um banco de referencia com ao menos uma entrada."""


class ReferenciasNaoEncontradasError(DomainError):
    """Pasta de referencias nao contem nenhuma imagem utilizavel."""
