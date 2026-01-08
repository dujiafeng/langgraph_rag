from langchain_community.embeddings import DashScopeEmbeddings

from config.settings import settings

_embeddings = None


def get_embeddings() -> DashScopeEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(
            model=settings.embedding_model,
            dashscope_api_key=settings.dashscope_api_key,
        )
    return _embeddings


def embed_text(text: str) -> list[float]:
    return get_embeddings().embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)
