# Session Notes

## 2026-06-22 — Portfolio review

### What we did

Reviewed the project end-to-end (all backend modules, eval harness, agent, README, BUILD_SPEC) and assessed what would make it portfolio-worthy for an entry-level ML/AI role.

---

### What's already strong

- `Key decisions` and `Deliberately out of scope` sections in README show ML reasoning, not just coding
- Retrieval eval harness (`app/eval.py`) with recall@k gives a real number to cite
- Graceful degradation (`degraded: true`) as a first-class feature shows production awareness
- Bounded LangGraph agent (hard cap: 2 rounds) is the right instinct vs. open-ended autonomy
- Docker Compose one-command bring-up, pydantic-settings config, typed throughout

---

### Critical issue

**The project is called "conversational" but is stateless.** The commit message, README, and repo name all say "conversational," but `POST /recommend` takes a single query string with no session/history. Either implement multi-turn context or rename the project and drop the "conversational" framing. This is the first thing to fix — it's a credibility problem.

---

### Prioritized action items

**P0 — Fix the credibility gap**

- [x] Implement multi-turn conversation: add `session_id` + conversation history to `POST /recommend`, prepend prior turns to the LLM prompt

**P1 — Eval depth (highest ML signal)**

- [x] Add LLM-as-judge end-to-end eval (`app/eval_e2e.py`): scores top-5 per query 1–5; LLM path 3.95 vs degraded 3.18, **+0.78 lift**; largest gains on musicals (+2.75) and war films (+1.60)
- [x] Fix retrieval eval: original `relevant_ids` were too narrow (hardcoded IDs not in collection → recall@10 ≈ 0); added `scripts/expand_eval_ids.py` to expand IDs via LLM-as-judge; **recall@10 now 0.40** on meaningful ground truth
- [x] Add embedding model ablation script (`scripts/embedding_ablation.py`): reads Qdrant payloads, re-embeds with 3 models into temp collections, compares recall@10; no TMDB key needed; table in README (run to fill in numbers)

**P2 — Technical depth**

- [x] Hybrid retrieval: BM25 sparse (`Qdrant/bm25`) + dense RRF fusion via Qdrant `Prefetch`/`FusionQuery`; `ingest.py` auto-migrates legacy unnamed-vector collections on next re-ingest; falls back cleanly to dense-only until then; `HYBRID_SEARCH=false` to disable
- [x] Per-request observability: HTTP middleware logs method/path/status/`duration_ms`; route logs `retrieval_ms`, `generation_ms`, `path=linear|agent`; `generation.py` logs `in=N out=N tokens` per LLM call

**P3 — Polish**

- [x] Streaming LLM responses: `POST /recommend/stream` SSE endpoint; async Anthropic client streams tokens as `{type:"chunk"}`; frontend renders live JSON preview below search bar with gradient fade; snaps to cards on `{type:"done"}`; complex queries emit `{type:"thinking"}` and run agent blocking then send single done event

**P4 — Deployment**

- [ ] Deploy publicly (with guardrails)
  - Platform: containerized FastAPI + vector DB + static React frontend — pick one, justify in README
  - Qdrant: managed Qdrant Cloud free tier vs container on same host — decide and document
  - Rate limiting on `/recommend` (per-IP)
  - Anthropic billing alert + hard cap documented in README
  - API key server-side only — confirm not in frontend bundle
  - CORS locked to deployed origin (not `*`)
  - "Live demo" link + deployment section in README
  - Verify: fresh browser, multi-turn conversation, degraded path, devtools confirms no key in client assets

### Effort estimates

| Item                        | Estimated effort |
| --------------------------- | ---------------- |
| Multi-turn conversation     | 2–3 hrs          |
| LLM-as-judge eval           | 2–3 hrs          |
| Embedding ablation notebook | 2–3 hrs          |
| Hybrid retrieval            | 4–6 hrs          |
| Observability middleware    | 1 hr             |
| Streaming responses         | 1–2 hrs          |
| Deploy                      | 1–2 hrs          |
