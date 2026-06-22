"""End-to-end eval: LLM-as-judge scoring of the full recommendation pipeline.

Compares two paths on the same 18 test queries:
  - LLM path:      retrieval → rank_and_explain → top-k
  - Degraded path: retrieval → raw vector rank  → top-k

Each path's top-k movies are scored 1–5 by Claude (based on metadata only,
so the score reflects relevance, not the quality of the generated reason).
Average scores and the LLM-ranking lift are reported.

Run:
  docker compose up -d qdrant
  cd backend
  python -m app.eval_e2e        # k=5
  python -m app.eval_e2e 3      # k=3
"""
from __future__ import annotations
import json
import sys
import anthropic
from app.config import settings
from app.embeddings import embed_query
from app.retrieval import search
from app.generation import rank_and_explain, LLMError
from app.eval import TEST_QUERIES

_JUDGE_SYSTEM = (
    "You are evaluating movie recommendations. "
    "For each movie listed, rate how well it matches the given query on a scale of 1–5:\n"
    "  1 = completely irrelevant\n"
    "  2 = loosely related\n"
    "  3 = somewhat relevant\n"
    "  4 = good match\n"
    "  5 = excellent match\n"
    "Base your rating solely on the movie's title, year, genres, and description — "
    "not on any explanation provided by the recommendation system.\n"
    "Return ONLY a JSON array of integers, one per movie, in the same order. "
    "Example for 4 movies: [3, 5, 2, 4]. No other text."
)


def _movie_line(c: dict) -> str:
    genres = ", ".join(c.get("genres", []))
    overview = (c.get("overview") or "")[:150]
    return f"- {c.get('title')} ({c.get('year')}) [{genres}] — {overview}"


def _score_batch(client: anthropic.Anthropic, query: str, candidates: list[dict]) -> list[int]:
    """Score a list of candidates against a query in a single API call. Returns 1–5 per item."""
    if not candidates:
        return []
    lines = "\n".join(_movie_line(c) for c in candidates)
    prompt = f"Query: {query}\n\nMovies:\n{lines}"
    try:
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=128,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        scores = json.loads(raw)
        return [max(1, min(5, int(s))) for s in scores[: len(candidates)]]
    except Exception:
        return [0] * len(candidates)


def run_e2e_eval(k: int = 5) -> None:
    if not settings.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY is required for end-to-end eval.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    col_w = 44
    print(f"\nEnd-to-End Eval (LLM-as-Judge) — top-{k} results\n{'=' * 62}")
    print(f"{'Query':<{col_w}} {'LLM':>5} {'Degraded':>9} {'Lift':>6}")
    print("-" * 62)

    llm_avgs: list[float] = []
    deg_avgs: list[float] = []

    for item in TEST_QUERIES:
        query = item["query"]
        vector = embed_query(query)
        candidates = search(vector, k=max(k * 3, 20))

        if not candidates:
            continue

        # Degraded path: raw vector rank
        degraded_top = candidates[:k]

        # LLM path: rank_and_explain then resolve back to payloads
        try:
            ranked = rank_and_explain(query, candidates)
            payload_by_id = {c["tmdb_id"]: c for c in candidates}
            llm_top = [
                payload_by_id[r["tmdb_id"]]
                for r in ranked[:k]
                if r["tmdb_id"] in payload_by_id
            ]
        except LLMError:
            llm_top = degraded_top  # fall back; scores will be same

        # Score both paths (metadata only — no generated reasons)
        llm_scores = _score_batch(client, query, llm_top)
        deg_scores = _score_batch(client, query, degraded_top)

        avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
        avg_deg = sum(deg_scores) / len(deg_scores) if deg_scores else 0.0
        llm_avgs.append(avg_llm)
        deg_avgs.append(avg_deg)

        label = query[: col_w - 2] + ".." if len(query) > col_w else query
        lift = avg_llm - avg_deg
        lift_str = f"+{lift:.2f}" if lift >= 0 else f"{lift:.2f}"
        print(f"{label:<{col_w}} {avg_llm:>5.2f} {avg_deg:>9.2f} {lift_str:>6}")

    print("-" * 62)
    total_llm = sum(llm_avgs) / len(llm_avgs) if llm_avgs else 0.0
    total_deg = sum(deg_avgs) / len(deg_avgs) if deg_avgs else 0.0
    total_lift = total_llm - total_deg
    lift_str = f"+{total_lift:.2f}" if total_lift >= 0 else f"{total_lift:.2f}"
    print(f"{'Average':<{col_w}} {total_llm:>5.2f} {total_deg:>9.2f} {lift_str:>6}")
    print(f"\nQueries evaluated: {len(llm_avgs)}, k={k}")
    print(f"LLM ranking lift over degraded: {lift_str} points (out of 5)\n")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_e2e_eval(k)
