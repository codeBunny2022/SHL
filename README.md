# SHL Conversational Assessment Recommender

I built a stateless FastAPI agent that helps hiring managers go from a vague hiring intent to a grounded shortlist of SHL Individual Test Solutions. It clarifies when needed, recommends 1–10 catalog assessments with real URLs, supports refinement and comparison, and refuses off-topic or legal questions.

## What it does

- **GET /health** — returns `{"status":"ok"}` for readiness checks
- **POST /chat** — accepts full conversation history, returns `reply`, `recommendations`, and `end_of_conversation`
- **GET /** — lightweight chat UI for manual testing

The agent is grounded entirely in a normalized catalog of 377 Individual Test Solutions (`shl_product_catalog.json`). Every recommendation name and URL comes from that catalog — never from the model.

## How I built it

- **Retrieval:** Gemini embeddings (`gemini-embedding-001`) over a pre-built index, blended with lexical overlap for exact skill matches (Java, Docker, Excel, etc.)
- **Agent:** One Gemini call per turn (`gemini-2.5-flash`) with structured JSON output; `picked_names` are validated against the catalog before building the response
- **Behaviors:** Clarify vague queries, recommend batteries (domain K tests + Verify + OPQ32r), refine on add/drop, compare from catalog descriptions, refuse legal/off-topic/injection

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini API key to `.env`, then:

```bash
python scripts/normalize_catalog.py   # only needed if catalog source changes
python scripts/build_index.py         # only needed if re-embedding
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 for the chat UI.

## API example

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer, mid-level"}]}'
```

## Evaluation

I used the 10 provided sample conversations as gold traces:

```bash
python eval/parse_traces.py
python eval/run_eval.py
```

This reports mean Recall@10 and behavior probes (vague no-rec, detailed rec, refuse off-topic/legal/injection).

## Deploy on Render

1. Push this repo to GitHub
2. Create a Web Service from `render.yaml`
3. Set `GEMINI_API_KEY` in Render environment variables
4. `data/catalog.json`, `data/embeddings.npy`, and `data/index_meta.json` are committed so cold start only loads files — no re-embedding at deploy time

## Project layout

```
app/           FastAPI service, agent, retrieval, prompts
data/          Normalized catalog + pre-built embedding index
scripts/       Catalog normalization and index build
eval/          Trace parser and eval harness
GenAI_SampleConversations/   Gold evaluation traces
shl_product_catalog.json     Source catalog
APPROACH.md    Design document for submission
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | — | Required at runtime for chat |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Chat model |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Embedding model |
