from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class Filters(BaseModel):
    year_min: int | None = None
    year_max: int | None = None
    runtime_max: int | None = None
    genres_include: list[str] = Field(default_factory=list)
    genres_exclude: list[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    query: str
    k: int = Field(default=10, ge=1, le=50)
    filters: Filters | None = None
    session_id: str | None = None


class SimilarRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=50)


class MovieResult(BaseModel):
    tmdb_id: int
    title: str
    year: int | None = None
    genres: list[str] = Field(default_factory=list)
    overview: str = ""
    runtime: int | None = None
    poster_path: str | None = None
    reason: str | None = None
    score: float | None = None


class RecommendResponse(BaseModel):
    results: list[MovieResult]
    degraded: bool = False
    query: str
    session_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    collection: str | None = None
