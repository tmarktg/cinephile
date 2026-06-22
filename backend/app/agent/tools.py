"""Fixed toolset for the agentic path."""
from __future__ import annotations
from app.embeddings import embed_query
from app.retrieval import search
from app.models import Filters


def search_by_theme(text: str, k: int = 20) -> list[dict]:
    vector = embed_query(text)
    return search(vector, k=k)


def search_by_metadata(filters: dict, k: int = 20) -> list[dict]:
    f = Filters(
        year_min=filters.get("year_min"),
        year_max=filters.get("year_max"),
        runtime_max=filters.get("runtime_max"),
        genres_include=filters.get("genres_include", []),
        genres_exclude=filters.get("genres_exclude", []),
    )
    # Use a broad semantic query with filters; empty string falls back to Qdrant scroll
    # Use a neutral embedding with the filter applied
    theme = filters.get("theme", "movie")
    vector = embed_query(theme)
    return search(vector, k=k, filters=f)


def apply_constraints(candidates: list[dict], constraints: dict) -> list[dict]:
    """Filter and deduplicate a candidate list against structured constraints."""
    seen: set[int] = set()
    result = []
    for c in candidates:
        tid = c.get("tmdb_id")
        if tid in seen:
            continue
        seen.add(tid)

        if constraints.get("year_min") and (c.get("year") or 0) < constraints["year_min"]:
            continue
        if constraints.get("year_max") and (c.get("year") or 9999) > constraints["year_max"]:
            continue
        if constraints.get("runtime_max") and (c.get("runtime") or 0) > constraints["runtime_max"]:
            continue

        genres = set(g.lower() for g in c.get("genres", []))
        if constraints.get("genres_include"):
            required = set(g.lower() for g in constraints["genres_include"])
            if not required & genres:
                continue
        if constraints.get("genres_exclude"):
            excluded = set(g.lower() for g in constraints["genres_exclude"])
            if excluded & genres:
                continue

        result.append(c)
    return result
