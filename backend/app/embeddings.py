from __future__ import annotations
from fastembed import TextEmbedding
from app.config import settings

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=settings.embed_model)
    return _model


def embed_query(text: str) -> list[float]:
    model = get_model()
    vectors = list(model.embed([text]))
    return vectors[0].tolist()
