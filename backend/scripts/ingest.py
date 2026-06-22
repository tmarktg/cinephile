#!/usr/bin/env python3
"""Pull movies from TMDB, embed, and upsert to Qdrant."""
from __future__ import annotations
import argparse
import os
import sys
import time
import httpx
from pathlib import Path

# allow running from repo root: python backend/scripts/ingest.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
)

from app.config import settings
from app.embeddings import embed_query, embed_sparse

TMDB_BASE = "https://api.themoviedb.org/3"


def fetch_popular_movies(api_key: str, count: int) -> list[dict]:
    movies = []
    page = 1
    print(f"Fetching up to {count} movies from TMDB...")

    with httpx.Client(timeout=30) as client:
        while len(movies) < count:
            resp = client.get(
                f"{TMDB_BASE}/movie/popular",
                params={"api_key": api_key, "language": "en-US", "page": page},
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("results", [])
            if not batch:
                break
            movies.extend(batch)
            total_pages = data.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1
            if page % 10 == 0:
                print(f"  Fetched {len(movies)} movies (page {page}/{total_pages})...")
                time.sleep(0.25)

    return movies[:count]


def fetch_movie_details(api_key: str, tmdb_id: int) -> dict | None:
    with httpx.Client(timeout=30) as client:
        try:
            detail_resp = client.get(
                f"{TMDB_BASE}/movie/{tmdb_id}",
                params={"api_key": api_key, "language": "en-US"},
            )
            detail_resp.raise_for_status()
            detail = detail_resp.json()

            kw_resp = client.get(
                f"{TMDB_BASE}/movie/{tmdb_id}/keywords",
                params={"api_key": api_key},
            )
            kw_resp.raise_for_status()
            keywords = [k["name"] for k in kw_resp.json().get("keywords", [])]

            return {
                "tmdb_id": tmdb_id,
                "title": detail.get("title", ""),
                "overview": detail.get("overview", ""),
                "genres": [g["name"] for g in detail.get("genres", [])],
                "keywords": keywords,
                "runtime": detail.get("runtime"),
                "year": int(detail.get("release_date", "0")[:4]) if detail.get("release_date") else None,
                "poster_path": detail.get("poster_path"),
            }
        except Exception as e:
            print(f"  Warning: failed to fetch details for {tmdb_id}: {e}")
            return None


def build_doc(movie: dict) -> str:
    genres = ", ".join(movie.get("genres", []))
    keywords = ", ".join(movie.get("keywords", [])[:20])
    return f"{movie['title']}. {movie.get('overview', '')} Genres: {genres}. Keywords: {keywords}."


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection in existing:
        info = client.get_collection(settings.qdrant_collection)
        # Migrate legacy unnamed-vector collections to named vectors for hybrid search
        if not isinstance(info.config.params.vectors, dict):
            print(f"Migrating '{settings.qdrant_collection}' to named-vector format (required for hybrid search)…")
            client.delete_collection(settings.qdrant_collection)
        else:
            print(f"Collection '{settings.qdrant_collection}' already exists")
            return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={"dense": VectorParams(size=settings.embed_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(f"Created collection '{settings.qdrant_collection}' with dense + sparse vectors")


def main():
    parser = argparse.ArgumentParser(description="Ingest TMDB movies into Qdrant")
    parser.add_argument("--count", type=int, default=3000, help="Number of movies to ingest")
    parser.add_argument("--batch-size", type=int, default=50, help="Upsert batch size")
    args = parser.parse_args()

    api_key = settings.tmdb_api_key
    if not api_key:
        print("Error: TMDB_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    qdrant = QdrantClient(url=settings.qdrant_url)
    ensure_collection(qdrant)

    popular = fetch_popular_movies(api_key, args.count)
    print(f"Got {len(popular)} movies from popular endpoint, fetching details...")

    points = []
    skipped = 0

    for i, movie in enumerate(popular):
        tmdb_id = movie["id"]
        details = fetch_movie_details(api_key, tmdb_id)
        if not details:
            skipped += 1
            continue

        doc = build_doc(details)
        dense = embed_query(doc)
        sparse = embed_sparse(doc)

        points.append(
            PointStruct(
                id=tmdb_id,
                vector={"dense": dense, "sparse": sparse},
                payload={
                    "tmdb_id": tmdb_id,
                    "title": details["title"],
                    "year": details["year"],
                    "genres": details["genres"],
                    "overview": details["overview"],
                    "runtime": details["runtime"],
                    "poster_path": details["poster_path"],
                },
            )
        )

        if len(points) >= args.batch_size:
            qdrant.upsert(collection_name=settings.qdrant_collection, points=points)
            print(f"  Upserted {i + 1 - skipped}/{len(popular)} movies...")
            points = []
            time.sleep(0.1)

    if points:
        qdrant.upsert(collection_name=settings.qdrant_collection, points=points)

    total = len(popular) - skipped
    print(f"Done. Ingested {total} movies ({skipped} skipped).")


if __name__ == "__main__":
    main()
