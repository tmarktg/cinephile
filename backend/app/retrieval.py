from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    Range,
    MatchAny,
    MatchExcept,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
)
from app.config import settings
from app.models import Filters

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def _build_filter(filters: Filters | None) -> Filter | None:
    if not filters:
        return None

    conditions = []

    if filters.year_min is not None or filters.year_max is not None:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(
                    gte=filters.year_min,
                    lte=filters.year_max,
                ),
            )
        )

    if filters.runtime_max is not None:
        conditions.append(
            FieldCondition(
                key="runtime",
                range=Range(lte=filters.runtime_max),
            )
        )

    if filters.genres_include:
        conditions.append(
            FieldCondition(
                key="genres",
                match=MatchAny(any=filters.genres_include),
            )
        )

    if filters.genres_exclude:
        conditions.append(
            FieldCondition(
                key="genres",
                match=MatchExcept(**{"except": filters.genres_exclude}),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def _is_named_vector_collection() -> bool:
    """Return True if the collection uses named vectors (hybrid-capable)."""
    client = get_client()
    try:
        info = client.get_collection(settings.qdrant_collection)
        return isinstance(info.config.params.vectors, dict)
    except Exception:
        return False


def search(
    query_vector: list[float],
    k: int = 15,
    filters: Filters | None = None,
    sparse_vector: SparseVector | None = None,
) -> list[dict]:
    client = get_client()
    qdrant_filter = _build_filter(filters)
    use_hybrid = (
        settings.hybrid_search
        and sparse_vector is not None
        and _is_named_vector_collection()
    )

    if use_hybrid:
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=k * 3,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=k * 3,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            with_payload=True,
        )
    else:
        # Dense-only path: works with both named ("dense") and legacy unnamed collections
        named = _is_named_vector_collection()
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            using="dense" if named else None,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

    candidates = []
    for r in results.points:
        payload = r.payload or {}
        payload["_score"] = r.score
        candidates.append(payload)

    return candidates


def get_movie_vector(tmdb_id: int) -> list[float] | None:
    client = get_client()
    results = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="tmdb_id", match=MatchAny(any=[tmdb_id]))]
        ),
        with_vectors=True,
        limit=1,
    )
    points, _ = results
    if not points:
        return None
    vec = points[0].vector
    # Named vector collection returns a dict; legacy returns a plain list
    if isinstance(vec, dict):
        return vec.get("dense")
    return vec


def check_health() -> dict:
    client = get_client()
    try:
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        collection = settings.qdrant_collection if settings.qdrant_collection in names else None
        return {"status": "ok", "collection": collection}
    except Exception as e:
        return {"status": "error", "error": str(e)}
