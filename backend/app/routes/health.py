from __future__ import annotations
from fastapi import APIRouter
from app.models import HealthResponse
from app.retrieval import check_health

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    result = check_health()
    return HealthResponse(
        status=result.get("status", "error"),
        qdrant=result.get("status", "error"),
        collection=result.get("collection"),
    )
