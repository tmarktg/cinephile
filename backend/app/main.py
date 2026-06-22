import time
import logging
import logging.config
from contextlib import asynccontextmanager

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"default": {"format": "%(levelname)s %(name)s %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "default"}},
    "root": {"level": "INFO", "handlers": ["console"]},
})
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.embeddings import get_model, get_sparse_model
from app.routes import health, recommend, similar

logger = logging.getLogger("cinephile.requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    get_sparse_model()
    yield


app = FastAPI(title="Cinephile", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
