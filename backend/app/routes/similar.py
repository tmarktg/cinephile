from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.models import RecommendResponse, MovieResult
from app.retrieval import get_movie_vector, search

router = APIRouter()


@router.get("/similar/{movie_id}", response_model=RecommendResponse)
async def similar(movie_id: int, k: int = 10) -> RecommendResponse:
    vector = get_movie_vector(movie_id)
    if vector is None:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found in collection")

    candidates = search(vector, k=k + 1)
    results = [
        MovieResult(
            tmdb_id=c["tmdb_id"],
            title=c.get("title", ""),
            year=c.get("year"),
            genres=c.get("genres", []),
            overview=c.get("overview", ""),
            runtime=c.get("runtime"),
            poster_path=c.get("poster_path"),
            score=c.get("_score"),
        )
        for c in candidates
        if c["tmdb_id"] != movie_id
    ][:k]

    return RecommendResponse(results=results, degraded=False, query=f"similar to {movie_id}")
