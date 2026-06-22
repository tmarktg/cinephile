from __future__ import annotations
import json
import anthropic
from app.config import settings


class LLMError(Exception):
    pass


class LLMParseError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not settings.anthropic_api_key:
            raise LLMAPIError("ANTHROPIC_API_KEY is not set")
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def rank_and_explain(query: str, candidates: list[dict], history: str = "") -> list[dict]:
    if not candidates:
        return []

    if not settings.anthropic_api_key:
        raise LLMAPIError("ANTHROPIC_API_KEY is not set")

    candidate_list = "\n".join(
        f"- tmdb_id={c['tmdb_id']}, title={c.get('title', '')}, year={c.get('year', '')}, "
        f"genres={c.get('genres', [])}, overview={c.get('overview', '')[:300]}"
        for c in candidates
    )

    system_prompt = (
        "You are a movie recommendation assistant. You will be given a user query and a list of candidate movies. "
        "Your task is to rank the candidates by how well they match the query and provide a brief explanation for each. "
        "You MUST only recommend movies from the provided candidate list — never invent or suggest movies not in the list. "
        "If prior conversation history is provided, use it to understand follow-up queries "
        "(e.g. 'something similar' refers to what was previously recommended). "
        "Respond with ONLY a valid JSON array. No prose, no markdown, no code fences. "
        "Each element must have exactly two fields: tmdb_id (integer) and reason (string, 1-2 sentences). "
        "Order the array from best to worst match. Include only movies that are actually relevant to the query."
    )

    if history:
        user_message = f"{history}\n\nCurrent query: {query}\n\nCandidates:\n{candidate_list}"
    else:
        user_message = f"Query: {query}\n\nCandidates:\n{candidate_list}"

    try:
        client = get_client()
        message = client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = message.content[0].text.strip()
        # strip markdown code fences if the model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()
    except anthropic.APIError as e:
        raise LLMAPIError(f"Anthropic API error: {e}") from e

    try:
        ranked = json.loads(raw)
        if not isinstance(ranked, list):
            raise LLMParseError("Expected a JSON array")
        for item in ranked:
            if "tmdb_id" not in item or "reason" not in item:
                raise LLMParseError("Missing tmdb_id or reason field")
        return ranked
    except json.JSONDecodeError as e:
        raise LLMParseError(f"Failed to parse LLM JSON response: {e}") from e
