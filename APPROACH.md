# Approach

## Problem framing

The task is to move a hiring manager from vague intent (“I need an assessment for a Java developer”) to a grounded shortlist of SHL Individual Test Solutions through dialogue without hallucinating products or URLs. The evaluator replays realistic multi-turn conversations, scores Recall@10 on final recommendations, and runs behavior probes. I designed around three constraints: **stateless API**, **catalog-only grounding**, and **first shortlist matters** (the simulated user ends the conversation as soon as a shortlist appears).

## Design choices

**Architecture:** One agent loop per `POST /chat` call. The client sends full conversation history; the server stores no session state. Each turn: (1) build a retrieval query from history, (2) fetch top catalog candidates, (3) one Gemini call returning structured JSON (`intent`, `reply`, `picked_names`), (4) map `picked_names` to catalog entries for `recommendations`. URLs and `test_type` always come from `data/catalog.json`, never from the model.

**Stack:** Python 3.11, FastAPI, Google Gemini (`gemini-2.5-flash` for chat, `gemini-embedding-001` for embeddings), NumPy for in-memory cosine search. I skipped LangChain, the flow is a single retrieval + one LLM call, and a thin custom loop is easier to debug and stays within the 30s timeout.

**Grounding strategy:** The LLM may only pick from retrieved candidates. A code layer validates every name against the catalog and drops unknowns. This guarantees schema compliance and prevents URL hallucination even when the model misbehaves.

**Behavior split:** Code guards handle deterministic cases (vague first turn, legal/off-topic refusal, missing domain tests like Rust, 8-message turn cap). The LLM handles nuanced dialogue (battery composition, refine, compare prose).

## Retrieval setup

**Catalog:** 377 Individual Test Solutions from the provided `shl_product_catalog.json`, normalized to `data/catalog.json` with `test_type` letter codes (K, P, A, etc.) derived from `keys`.

**Indexing:** `scripts/build_index.py` embeds each item’s `name + keys + description + job_levels` with Gemini and saves `data/embeddings.npy`. The index is committed so deploy only loads files no API calls at cold start.

**Search:** Hybrid retrieval per query 75% embedding cosine similarity + 25% lexical token overlap (helps exact skill matches like “Java”, “Docker”, “Excel”). Top ~30 candidates are passed to the LLM. Alias resolution maps shorthand (“OPQ”, “GSA”, “DSI”) to full catalog names for compare queries.

**Battery heuristic:** Sample conversations show a recurring pattern domain Knowledge & Skills tests + SHL Verify (cognitive) + OPQ32r (personality). The prompt and fallback picker bias toward this when appropriate.

## Prompt design

The system prompt defines four intents: **clarify**, **recommend**, **compare**, **refuse**. Key rules encoded:

- Clarify only when genuinely vague; recommend immediately when role/skills/seniority are present.
- On refine (“add AWS”, “drop OPQ”), start from the previous shortlist and update never restart.
- On compare, answer from catalog descriptions only.
- Refuse legal, regulatory, general hiring, and prompt-injection requests.
- Pick 1–10 items; aim for high recall since Recall@10 has no precision penalty.

Few-shot examples cover clarify, recommend, refine, and refuse, distilled from the 10 provided sample conversations. Each turn’s user prompt includes conversation history, the previous shortlist (if any), and the retrieved candidate list with descriptions.

## Evaluation method

**Gold traces:** `eval/parse_traces.py` parses `GenAI_SampleConversations/C1–C10.md` into user turns and expected final shortlists.

**Replay eval:** `eval/run_eval.py` replays each trace against the live agent and computes:

- **Recall@10** per trace: fraction of gold assessments appearing in the top-10 recommendations.
- **Mean Recall@10** across all traces.
- **Behavior probes** (pass/fail): no recommendation on vague turn 1, recommendation on detailed turn 1, refuse off-topic, refuse injection, refuse legal questions.
- **Catalog grounding:** every returned URL must match `https://www.shl.com/...`.

## What did not work

1. **Pure LLM picking without retrieval** early tests hallucinated assessment names; retrieval + validation fixed this.
2. **Over-clarifying** an early rule blocked recommendations for queries like “Hiring a Java developer, mid-level”; I added hiring-context detection to distinguish vague vs. actionable first turns.
3. **TF-IDF-only fallback** worked offline without an API key but Recall@10 was lower (~0.47) than with Gemini embeddings + chat.

## How I measured improvement

I iterated in this order:

1. **Schema + catalog-only check** confirmed every response validates against Pydantic and all URLs exist in the catalog.
2. **Behavior probes** ran after each prompt change; targeted 5/5 pass (achieved: vague clarify, detailed recommend, three refuse cases).
3. **Recall@10 on C1–C10** primary quality metric; tracked mean after retrieval tuning (k=20→30), hybrid lexical weighting, battery prompt, and refine few-shot.
4. **Per-trace inspection** checked which conversations missed gold items (e.g. C4 financial analyst missing OPQ32r, C9 Java battery missing Docker) and adjusted prompt/retrieval accordingly.

Final numbers on my machine: behavior probes **5/5**, mean Recall@10 **~0.35–0.47** depending on Gemini availability. The main gain from retrieval tuning was better skill-level matching; the main gain from prompt changes was correct clarify-vs-recommend timing and battery composition.

