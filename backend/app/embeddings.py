from __future__ import annotations
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import SparseVector
from app.config import settings

_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=settings.embed_model)
    return _model


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def embed_query(text: str) -> list[float]:
    return list(next(get_model().embed([text]))).copy()


def embed_sparse(text: str) -> SparseVector:
    result = next(get_sparse_model().embed([text]))
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )
