from __future__ import annotations
import asyncio
import json
import logging
import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models import RecommendRequest, RecommendResponse, MovieResult
from app.embeddings import embed_query, embed_sparse
from app.retrieval import search
from app.generation import rank_and_explain, astream_rank_and_explain, LLMError
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
    t0 = time.perf_counter()
    vector = embed_query(req.query)
    sparse = embed_sparse(req.query)
    candidates = search(vector, k=max(req.k * 2, 15), filters=req.filters, sparse_vector=sparse)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    if not candidates:
        return RecommendResponse(results=[], degraded=False, query=req.query)

    t1 = time.perf_counter()
    try:
        ranked = rank_and_explain(req.query, candidates, history=history)
        generation_ms = (time.perf_counter() - t1) * 1000

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

        logger.info(
            "recommend path=linear retrieval_ms=%.0f generation_ms=%.0f candidates=%d results=%d",
            retrieval_ms, generation_ms, len(candidates), len(results),
        )
        return RecommendResponse(results=results, degraded=False, query=req.query)

    except LLMError as e:
        generation_ms = (time.perf_counter() - t1) * 1000
        logger.warning(
            "recommend path=linear degraded=true retrieval_ms=%.0f generation_ms=%.0f error=%s",
            retrieval_ms, generation_ms, e,
        )
        results = [_build_movie_result(c) for c in candidates[: req.k]]
        return RecommendResponse(results=results, degraded=True, query=req.query)


def _agent_path(req: RecommendRequest, history: str = "") -> RecommendResponse:
    t0 = time.perf_counter()
    try:
        payloads, degraded = run_agent(req.query, history=history)
        results = [
            _build_movie_result(p, reason=p.get("reason"))
            for p in payloads[: req.k]
        ]
        logger.info(
            "recommend path=agent duration_ms=%.0f results=%d degraded=%s",
            (time.perf_counter() - t0) * 1000, len(results), degraded,
        )
        return RecommendResponse(results=results, degraded=degraded, query=req.query)
    except Exception as e:
        logger.warning("Agent path failed, falling back to linear: %s", e)
        return _linear_path(req, history=history)


@router.post("/recommend/stream")
async def recommend_stream(req: RecommendRequest) -> StreamingResponse:
    """SSE endpoint. Yields:
      data: {"type": "chunk", "text": "..."}   — one token at a time (linear path only)
      data: {"type": "thinking"}                — complex query, agent running
      data: {"type": "done", "results": [...], "degraded": bool, "session_id": "...", "query": "..."}
    """
    async def generate():
        session_id, session = get_or_create(req.session_id)
        history = format_history(session)

        vector = embed_query(req.query)
        sparse = embed_sparse(req.query)
        candidates = search(vector, k=max(req.k * 2, 15), filters=req.filters, sparse_vector=sparse)

        if not candidates:
            payload = {"type": "done", "results": [], "degraded": False,
                       "session_id": session_id, "query": req.query}
            yield f"data: {json.dumps(payload)}\n\n"
            return

        payload_by_id = {c["tmdb_id"]: c for c in candidates}
        results: list[MovieResult] = []
        degraded = False

        query_type = await asyncio.to_thread(classify_query, req.query)

        if query_type == "complex":
            yield f"data: {json.dumps({'type': 'thinking'})}\n\n"
            try:
                payloads, degraded = await asyncio.to_thread(run_agent, req.query, history=history)
                results = [_build_movie_result(p, reason=p.get("reason")) for p in payloads[: req.k]]
            except Exception as e:
                logger.warning("Agent failed in stream endpoint: %s", e)
                results = [_build_movie_result(c) for c in candidates[: req.k]]
                degraded = True
        else:
            try:
                ranked = None
                async for event in astream_rank_and_explain(req.query, candidates, history=history):
                    if event["type"] == "chunk":
                        yield f"data: {json.dumps(event)}\n\n"
                    elif event["type"] == "result":
                        ranked = event["ranked"]
                    elif event["type"] == "error":
                        logger.warning("Stream parse error: %s", event["error"])
                        degraded = True

                if ranked and not degraded:
                    for item in ranked[: req.k]:
                        p = payload_by_id.get(item["tmdb_id"])
                        if p:
                            results.append(_build_movie_result(p, reason=item["reason"]))
                    ranked_ids = {item["tmdb_id"] for item in ranked}
                    for c in candidates:
                        if len(results) >= req.k:
                            break
                        if c["tmdb_id"] not in ranked_ids:
                            results.append(_build_movie_result(c))
                if degraded or not results:
                    results = [_build_movie_result(c) for c in candidates[: req.k]]
                    degraded = True
            except LLMError as e:
                logger.warning("LLM stream failed, degraded: %s", e)
                results = [_build_movie_result(c) for c in candidates[: req.k]]
                degraded = True

        session.add_turn(req.query, [r.title for r in results])
        done = {
            "type": "done",
            "results": [r.model_dump() for r in results],
            "degraded": degraded,
            "session_id": session_id,
            "query": req.query,
        }
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    session_id, session = get_or_create(req.session_id)
    history = format_history(session)

    query_type = classify_query(req.query)
    logger.info("query_type=%s query=%r", query_type, req.query)

    if query_type == "complex":
        response = _agent_path(req, history=history)
    else:
        response = _linear_path(req, history=history)

    session.add_turn(req.query, [r.title for r in response.results])
    response.session_id = session_id
    return response
