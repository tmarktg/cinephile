from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchAny, MatchExcept
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


def search(
    query_vector: list[float],
    k: int = 15,
    filters: Filters | None = None,
) -> list[dict]:
    client = get_client()
    qdrant_filter = _build_filter(filters)

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
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
    return points[0].vector


def check_health() -> dict:
    client = get_client()
    try:
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        collection = settings.qdrant_collection if settings.qdrant_collection in names else None
        return {"status": "ok", "collection": collection}
    except Exception as e:
        return {"status": "error", "error": str(e)}
