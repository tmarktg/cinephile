"""Expand TEST_QUERIES relevant_ids using LLM-as-judge.

For each test query, retrieves top-15 candidates and asks Claude to score
each movie 1–5. Movies scoring ≥ 4 are added to the relevant_ids set.
Prints updated TEST_QUERIES ready to paste into app/eval.py.

Run (once, with Qdrant populated and ANTHROPIC_API_KEY set):
  cd backend
  python scripts/expand_eval_ids.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from app.config import settings
from app.embeddings import embed_query
from app.retrieval import search
from app.eval import TEST_QUERIES

_JUDGE_SYSTEM = (
    "You are evaluating movie recommendations. "
    "For each movie listed, rate how well it matches the given query on a scale of 1–5:\n"
    "  1 = completely irrelevant\n"
    "  2 = loosely related\n"
    "  3 = somewhat relevant\n"
    "  4 = good match\n"
    "  5 = excellent match\n"
    "Return ONLY a JSON array of integers, one per movie, in the same order. "
    "Example for 3 movies: [4, 2, 5]. No other text."
)

RELEVANCE_THRESHOLD = 4
RETRIEVE_K = 15


def score_candidates(client: anthropic.Anthropic, query: str, candidates: list[dict]) -> list[int]:
    lines = []
    for c in candidates:
        genres = ", ".join(c.get("genres") or [])
        overview = (c.get("overview") or "")[:150]
        lines.append(f"- {c.get('title')} ({c.get('year')}) [{genres}] — {overview}")
    prompt = f"Query: {query}\n\nMovies:\n" + "\n".join(lines)
    try:
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=128,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        scores = json.loads(msg.content[0].text.strip())
        return [max(1, min(5, int(s))) for s in scores[: len(candidates)]]
    except Exception as e:
        print(f"  Warning: judge call failed: {e}", file=sys.stderr)
        return [0] * len(candidates)


def main() -> None:
    if not settings.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY required.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    updated: list[dict] = []

    for item in TEST_QUERIES:
        query = item["query"]
        seed_ids = set(item["relevant_ids"])
        print(f"  {query[:60]}")

        vector = embed_query(query)
        candidates = search(vector, k=RETRIEVE_K)

        scores = score_candidates(client, query, candidates)
        judge_ids = {
            c["tmdb_id"]
            for c, s in zip(candidates, scores)
            if s >= RELEVANCE_THRESHOLD
        }

        merged = sorted(seed_ids | judge_ids)
        added = judge_ids - seed_ids
        if added:
            added_titles = [
                f"{c['title']} ({c['tmdb_id']})"
                for c in candidates
                if c["tmdb_id"] in added
            ]
            print(f"    + {', '.join(added_titles)}")

        updated.append({"query": query, "relevant_ids": merged})

    print("\n\n# ── paste into backend/app/eval.py ──────────────────────────")
    print("TEST_QUERIES = [")
    for item in updated:
        ids_str = ", ".join(str(i) for i in item["relevant_ids"])
        print(f'    {{"query": "{item["query"]}", "relevant_ids": [{ids_str}]}},')
    print("]")


if __name__ == "__main__":
    main()
