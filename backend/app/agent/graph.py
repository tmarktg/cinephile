"""LangGraph agent for multi-constraint queries. Hard cap: 2 retrieval rounds."""
from __future__ import annotations
import json
import logging
from typing import TypedDict, Annotated
import operator

import anthropic
from langgraph.graph import StateGraph, END

from app.config import settings
from app.agent.tools import search_by_theme, search_by_metadata, apply_constraints
from app.generation import rank_and_explain, LLMError

logger = logging.getLogger(__name__)

MAX_ROUNDS = 2


class AgentState(TypedDict):
    query: str
    history: str
    semantic_part: str
    constraints: dict
    # operator.add means returned candidates are appended to existing list
    candidates: Annotated[list[dict], operator.add]
    rounds: int
    final_results: list[dict]
    degraded: bool


def _decompose(query: str, history: str = "") -> dict:
    system_prompt = (
        "Decompose this movie recommendation query into two parts. "
        "If prior conversation history is provided, use it to resolve references like 'something similar' or 'more recent'. "
        "Return ONLY valid JSON with these fields: "
        "{\"semantic\": \"<the mood/theme/style part for vector search>\", "
        "\"constraints\": {\"year_min\": <int|null>, \"year_max\": <int|null>, "
        "\"runtime_max\": <int|null>, \"genres_include\": [<str>], \"genres_exclude\": [<str>]}}. "
        "No prose, no markdown."
    )
    content = f"{history}\n\nQuery to decompose: {query}" if history else query
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        return json.loads(msg.content[0].text.strip())
    except Exception as e:
        logger.warning("Decompose failed: %s", e)
        return {"semantic": query, "constraints": {}}


def node_decompose(state: AgentState) -> dict:
    decomposed = _decompose(state["query"], history=state.get("history", ""))
    return {
        "semantic_part": decomposed.get("semantic", state["query"]),
        "constraints": decomposed.get("constraints", {}),
        "rounds": 0,
        "final_results": [],
        "degraded": False,
    }


def node_retrieve(state: AgentState) -> dict:
    theme_results = search_by_theme(state["semantic_part"], k=20)
    constraints = state["constraints"]

    has_structured = any([
        constraints.get("year_min"),
        constraints.get("year_max"),
        constraints.get("runtime_max"),
        constraints.get("genres_include"),
        constraints.get("genres_exclude"),
    ])

    if has_structured:
        meta_results = search_by_metadata({**constraints, "theme": state["semantic_part"]}, k=20)
        combined = theme_results + meta_results
    else:
        combined = theme_results

    filtered = apply_constraints(combined, constraints)
    # Return delta: candidates to append, incremented rounds
    return {
        "candidates": filtered,
        "rounds": state["rounds"] + 1,
    }


def node_check(state: AgentState) -> str:
    if state["rounds"] >= MAX_ROUNDS:
        return "rank"
    if len(state["candidates"]) >= 10:
        return "rank"
    return "retrieve"


def node_rank(state: AgentState) -> dict:
    candidates = state["candidates"]

    seen: set[int] = set()
    unique = []
    for c in candidates:
        tid = c.get("tmdb_id")
        if tid not in seen:
            seen.add(tid)
            unique.append(c)

    try:
        ranked = rank_and_explain(state["query"], unique[:30], history=state.get("history", ""))
        payload_by_id = {c["tmdb_id"]: c for c in unique}
        results = []
        for item in ranked[:10]:
            payload = payload_by_id.get(item["tmdb_id"])
            if payload:
                entry = dict(payload)
                entry["reason"] = item["reason"]
                results.append(entry)
        return {"final_results": results, "degraded": False}
    except LLMError as e:
        logger.warning("Agent rank failed: %s", e)
        return {"final_results": unique[:10], "degraded": True}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("decompose", node_decompose)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("rank", node_rank)

    builder.set_entry_point("decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_conditional_edges("retrieve", node_check, {"retrieve": "retrieve", "rank": "rank"})
    builder.add_edge("rank", END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(query: str, history: str = "") -> tuple[list[dict], bool]:
    """Returns (results, degraded). Each result is a payload dict with optional 'reason'."""
    graph = get_graph()
    initial: AgentState = {
        "query": query,
        "history": history,
        "semantic_part": "",
        "constraints": {},
        "candidates": [],
        "rounds": 0,
        "final_results": [],
        "degraded": False,
    }
    final_state = graph.invoke(initial)
    return final_state["final_results"], final_state["degraded"]
