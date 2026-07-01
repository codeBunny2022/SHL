# Approach

## Design
I built a single-turn agent loop per `/chat` request: build a retrieval query from the full conversation history, fetch top catalog candidates, ask Gemini for structured JSON (`intent`, `reply`, `picked_names`), then map those names back to catalog entries for `recommendations`. URLs and `test_type` never come from the model — only from `shl_product_catalog.json`.

## Retrieval
377 items normalized to `data/catalog.json` with `test_type` letters derived from `keys`. Hybrid search: 75% Gemini embedding cosine similarity + 25% lexical token overlap. Precomputed `data/embeddings.npy` at build time so production only loads files. Alias resolution handles shorthand like "OPQ" and "GSA" for compare queries.

## Prompt design
The system prompt encodes SHL-only scope, clarify-vs-recommend rules, refine/compare/refuse behaviors from the 10 sample conversations. Few-shot examples cover clarify, recommend, refine, and refuse. Code guards enforce vague first-turn clarification, 8-message turn cap, and catalog-only validation of picked names.

## Evaluation
`eval/parse_traces.py` extracts user turns and gold shortlists from `GenAI_SampleConversations/`. `eval/run_eval.py` replays traces, reports mean Recall@10, and runs behavior probes (vague no-rec, detailed rec, off-topic, injection, legal refuse).

## What didn't work
- Scraping was unnecessary — the provided JSON was complete.
- `gemini-2.0-flash` returned 404 on my API key; switched to `gemini-2.5-flash`.
- `text-embedding-004` was unavailable; switched to `gemini-embedding-001`.
- First shortlist is effectively final per the evaluator design, so retrieval breadth (k=30) and the battery pattern (domain K + Verify A + OPQ P) matter more than multi-turn refinement for scoring.

## AI tools used
Cursor for scaffolding, prompt iteration, and eval harness. Gemini for runtime chat and embeddings.
