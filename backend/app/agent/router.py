"""Logical query router: classify single-constraint vs multi-constraint queries."""
from __future__ import annotations
import json
import anthropic
from app.config import settings


def classify_query(query: str) -> str:
    """Return 'simple' or 'complex'. Falls back to 'simple' on any error."""
    if not settings.anthropic_api_key:
        return "simple"

    system_prompt = (
        "You classify movie recommendation queries. "
        "A query is 'complex' if it has MULTIPLE distinct constraints that must all be satisfied simultaneously "
        "(e.g., specific director + genre + era, or mood + runtime limit + genre exclusion). "
        "A query is 'simple' if it expresses a single theme, mood, or style. "
        "Respond with ONLY a JSON object: {\"type\": \"simple\"} or {\"type\": \"complex\"}. "
        "No other text."
    )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=settings.llm_model,
            max_tokens=32,
            system=system_prompt,
            messages=[{"role": "user", "content": query}],
        )
        raw = message.content[0].text.strip()
        result = json.loads(raw)
        return result.get("type", "simple")
    except Exception:
        return "simple"
