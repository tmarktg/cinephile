"""Embedding model ablation: compare recall@10 across three FastEmbed models.

Reads movie payloads from the existing Qdrant collection (no TMDB key needed),
re-embeds them with each candidate model, creates a temporary Qdrant collection,
runs recall@10 over all 18 test queries, then cleans up.

Models compared:
  bge-small-en-v1.5  (384-dim) — project baseline
  all-MiniLM-L6-v2  (384-dim) — common sentence-transformer baseline
  bge-base-en-v1.5  (768-dim) — larger BGE variant

Run:
  docker compose up -d qdrant
  cd backend
  python scripts/embedding_ablation.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
)

from app.config import settings
from app.eval import TEST_QUERIES, recall_at_k

MODELS = [
    ("BAAI/bge-small-en-v1.5", 384),
    ("sentence-transformers/all-MiniLM-L6-v2", 384),
    ("BAAI/bge-base-en-v1.5", 768),
]

EVAL_K = 10
BATCH_SIZE = 64


def _scroll_all_payloads(client: QdrantClient) -> list[dict]:
    """Fetch every point's payload from the existing collection."""
    payloads = []
    offset = None
    while True:
        result, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            offset=offset,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        for point in result:
            if point.payload:
                payloads.append(point.payload)
        if next_offset is None:
            break
        offset = next_offset
    return payloads


def _build_doc(payload: dict) -> str:
    genres = ", ".join(payload.get("genres") or [])
    overview = payload.get("overview") or ""
    return f"{payload.get('title', '')}. {overview} Genres: {genres}."


def _embed_all(model_name: str, docs: list[str]) -> list[list[float]]:
    model = TextEmbedding(model_name)
    vectors: list[list[float]] = []
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        vectors.extend([list(v) for v in model.embed(batch)])
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"    embedded {min(i + BATCH_SIZE, len(docs))}/{len(docs)}", end="\r")
    print()
    return vectors


def _run_recall(
    client: QdrantClient,
    collection: str,
    model_name: str,
    dim: int,
) -> list[float]:
    query_model = TextEmbedding(model_name)
    scores = []
    for item in TEST_QUERIES:
        query_vec = list(next(query_model.embed([item["query"]])))
        results = client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=EVAL_K,
            with_payload=True,
        )
        retrieved_ids = [p.payload.get("tmdb_id") for p in results.points if p.payload]
        score = recall_at_k(set(item["relevant_ids"]), retrieved_ids, EVAL_K)
        scores.append(score)
    return scores


def main() -> None:
    client = QdrantClient(url=settings.qdrant_url)

    print("Reading payloads from existing collection…")
    payloads = _scroll_all_payloads(client)
    print(f"  {len(payloads)} movies loaded")
    docs = [_build_doc(p) for p in payloads]
    tmdb_ids = [p["tmdb_id"] for p in payloads]

    col_w = 36
    print(f"\nEmbedding Model Ablation — recall@{EVAL_K}\n{'=' * 62}")
    print(f"{'Model':<{col_w}} {'Avg recall@' + str(EVAL_K):>12}   {'vs baseline':>11}")
    print("-" * 62)

    baseline_avg: float | None = None
    temp_collections: list[str] = []

    for model_name, dim in MODELS:
        short_name = model_name.split("/")[-1]
        temp_col = f"_ablation_{short_name.replace('-', '_')}"
        temp_collections.append(temp_col)

        print(f"\n[{short_name}]  dim={dim}")
        print("  Embedding…")
        t0 = time.time()
        vectors = _embed_all(model_name, docs)
        embed_time = time.time() - t0
        print(f"  Done in {embed_time:.1f}s. Upserting to Qdrant…")

        # create / recreate temp collection
        existing = [c.name for c in client.get_collections().collections]
        if temp_col in existing:
            client.delete_collection(temp_col)
        client.create_collection(
            collection_name=temp_col,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=tmdb_id, vector=vec, payload={"tmdb_id": tmdb_id})
            for tmdb_id, vec in zip(tmdb_ids, vectors)
        ]
        for i in range(0, len(points), BATCH_SIZE):
            client.upsert(collection_name=temp_col, points=points[i : i + BATCH_SIZE])

        print("  Running recall eval…")
        scores = _run_recall(client, temp_col, model_name, dim)
        avg = sum(scores) / len(scores) if scores else 0.0

        if baseline_avg is None:
            baseline_avg = avg
            delta_str = "baseline"
        else:
            delta = avg - baseline_avg
            delta_str = f"{'+' if delta >= 0 else ''}{delta:.3f}"

        print(f"  {short_name:<{col_w}} {avg:>12.3f}   {delta_str:>11}")

    print("\n" + "-" * 62)
    print("Cleaning up temp collections…")
    for col in temp_collections:
        try:
            client.delete_collection(col)
        except Exception:
            pass
    print("Done.\n")


if __name__ == "__main__":
    main()
