from __future__ import annotations
import logging
from fastapi import APIRouter
from app.models import RecommendRequest, RecommendResponse, MovieResult
from app.embeddings import embed_query
from app.retrieval import search
from app.generation import rank_and_explain, LLMError
from app.agent.router import classify_query
from app.agent.graph import run_agent
from app.sessions import get_or_create, format_history

logger = logging.getLogger(__name__)
router = APIRouter()


def _build_movie_result(payload: dict, reason: str | None = None) -> MovieResult:
    return MovieResult(
        tmdb_id=payload["tmdb_id"],
        title=payload.get("title", ""),
        year=payload.get("year"),
        genres=payload.get("genres", []),
        overview=payload.get("overview", ""),
        runtime=payload.get("runtime"),
        poster_path=payload.get("poster_path"),
        reason=reason,
        score=payload.get("_score"),
    )


def _linear_path(req: RecommendRequest, history: str = "") -> RecommendResponse:
    vector = embed_query(req.query)
    candidates = search(vector, k=max(req.k * 2, 15), filters=req.filters)

    if not candidates:
        return RecommendResponse(results=[], degraded=False, query=req.query)

    try:
        ranked = rank_and_explain(req.query, candidates, history=history)
        payload_by_id = {c["tmdb_id"]: c for c in candidates}
        results = []
        for item in ranked[: req.k]:
            payload = payload_by_id.get(item["tmdb_id"])
            if payload:
                results.append(_build_movie_result(payload, reason=item["reason"]))

        ranked_ids = {item["tmdb_id"] for item in ranked}
        for c in candidates:
            if len(results) >= req.k:
                break
            if c["tmdb_id"] not in ranked_ids:
                results.append(_build_movie_result(c))

        return RecommendResponse(results=results, degraded=False, query=req.query)

    except LLMError as e:
        logger.warning("LLM step failed, returning degraded results: %s", e)
        results = [_build_movie_result(c) for c in candidates[: req.k]]
        return RecommendResponse(results=results, degraded=True, query=req.query)


def _agent_path(req: RecommendRequest, history: str = "") -> RecommendResponse:
    try:
        payloads, degraded = run_agent(req.query, history=history)
        results = [
            _build_movie_result(p, reason=p.get("reason"))
            for p in payloads[: req.k]
        ]
        return RecommendResponse(results=results, degraded=degraded, query=req.query)
    except Exception as e:
        logger.warning("Agent path failed, falling back to linear: %s", e)
        return _linear_path(req, history=history)


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    session_id, session = get_or_create(req.session_id)
    history = format_history(session)

    query_type = classify_query(req.query)
    logger.info("Query classified as '%s': %s", query_type, req.query)

    if query_type == "complex":
        response = _agent_path(req, history=history)
    else:
        response = _linear_path(req, history=history)

    session.add_turn(req.query, [r.title for r in response.results])
    response.session_id = session_id
    return response
