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

- [ ] Hybrid retrieval: add BM25 sparse vectors alongside dense in Qdrant; fuse scores; measure whether it lifts recall on keyword-heavy queries vs. semantic queries
- [ ] Per-request observability: FastAPI middleware logging `retrieval_ms`, `generation_ms`, `tokens_used`

**P3 — Polish**

- [ ] Streaming LLM responses via `StreamingResponse`
- [ ] Deploy publicly (with guardrails)

Problem: For entry-level roles, a live "try it now" link is worth more than a local-only repo, because interviewers don't always run local setups.

Build:

Recommend and use a platform that fits this stack (containerized FastAPI + a vector DB + a static React frontend). Pick based on the actual stack and explain the choice briefly in the README (e.g. container support, free tier, managed vs self-hosted Qdrant). Note the Qdrant decision explicitly: managed Qdrant Cloud free tier vs running the container on the same host — pick one and justify it.
Non-negotiable guardrails (a public endpoint that calls a paid API is a liability without these):

Rate limiting on /recommend (per-IP, sensible low limit) so the endpoint can't be hammered.
A hard spending cap / billing alert on the Anthropic key — document the cap value in the README ops section. Assume the worst case of a bot looping requests.
API key server-side only — never shipped in the frontend bundle. Confirm the built frontend contains no secret.
CORS locked to the deployed frontend origin, not \*.
A lightweight fallback so a cold/slow vector DB or a tripped rate limit returns a graceful message, not a stack trace.

Add a "Live demo" link and a short "Running locally" + "Deployment" section to the README.

## Verify: Hit the public URL from a fresh browser, run a multi-turn conversation, confirm the degraded path still degrades gracefully, and confirm (via devtools) no API key is present in client assets.

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
