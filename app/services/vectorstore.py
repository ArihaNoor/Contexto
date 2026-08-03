"""Per-session ChromaDB collections backed by local fastembed embeddings."""

from functools import lru_cache

import chromadb
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import settings

COLLECTION_PREFIX = "session_"


class FastEmbedEmbeddings(Embeddings):
    """LangChain adapter for fastembed's ONNX runtime (no torch required)."""

    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(settings.embedding_model)


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=settings.chroma_dir)


def collection_name(session_id: str) -> str:
    return f"{COLLECTION_PREFIX}{session_id}"


def get_vectorstore(session_id: str) -> Chroma:
    return Chroma(
        client=get_client(),
        collection_name=collection_name(session_id),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def session_exists(session_id: str) -> bool:
    try:
        get_client().get_collection(collection_name(session_id))
        return True
    except Exception:
        return False


def delete_session(session_id: str) -> None:
    get_client().delete_collection(collection_name(session_id))
