# syntax=docker/dockerfile:1

# ── Stage 1: build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + built frontend ───────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

COPY backend/ ./
RUN pip install --no-cache-dir .

# Pre-download embedding models so the container starts without network I/O
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
    TextEmbedding('BAAI/bge-small-en-v1.5'); \
    SparseTextEmbedding('Qdrant/bm25')"

COPY --from=frontend /frontend/dist ./static

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
