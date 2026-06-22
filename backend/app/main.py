import time
import logging
import logging.config
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(levelname)s %(name)s %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "default"}},
    "root": {"level": "INFO", "handlers": ["console"]},
})
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.embeddings import get_model, get_sparse_model
from app.routes import health, recommend, similar

logger = logging.getLogger("cinephile.requests")

_rate_window: dict[str, list[float]] = defaultdict(list)
_RATE_PATHS = {"/recommend", "/recommend/stream"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    get_sparse_model()
    yield


app = FastAPI(title="Cinephile", lifespan=lifespan)

_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in _RATE_PATHS:
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        _rate_window[ip] = [t for t in _rate_window[ip] if now - t < 60]
        if len(_rate_window[ip]) >= settings.rate_limit_per_minute:
            return JSONResponse(
                {"error": "Rate limit exceeded. Try again in a minute."},
                status_code=429,
            )
        _rate_window[ip].append(now)
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("%s %s %d %.0fms", request.method, request.url.path, response.status_code, ms)
    return response


app.include_router(health.router)
app.include_router(recommend.router)
app.include_router(similar.router)

# Serve built React frontend when running in production container
_static = Path(__file__).parent.parent / "static"
if _static.exists():
    _assets = _static / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_fallback(path: str):
        return FileResponse(str(_static / "index.html"))

    @app.get("/", include_in_schema=False)
    async def spa_root():
        return FileResponse(str(_static / "index.html"))
