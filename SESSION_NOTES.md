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
- [ ] Add LLM-as-judge end-to-end eval: feed query + top results back to Claude and score relevance 1–5; run across the existing 18 test queries; report average score in degraded vs. non-degraded mode
- [ ] Add embedding model ablation: extend `eval.py` (or add a notebook) to compare recall@10 across `bge-small-en-v1.5`, `all-MiniLM-L6-v2`, `bge-base-en-v1.5`; put the table in the README

**P2 — Technical depth**
- [ ] Hybrid retrieval: add BM25 sparse vectors alongside dense in Qdrant; fuse scores; measure whether it lifts recall on keyword-heavy queries vs. semantic queries
- [ ] Per-request observability: FastAPI middleware logging `retrieval_ms`, `generation_ms`, `tokens_used`

**P3 — Polish**
- [ ] Streaming LLM responses via `StreamingResponse`
- [ ] Deploy to Fly.io or Railway for a live public URL

---

### Effort estimates
| Item | Estimated effort |
|---|---|
| Multi-turn conversation | 2–3 hrs |
| LLM-as-judge eval | 2–3 hrs |
| Embedding ablation notebook | 2–3 hrs |
| Hybrid retrieval | 4–6 hrs |
| Observability middleware | 1 hr |
| Streaming responses | 1–2 hrs |
| Deploy | 1–2 hrs |
