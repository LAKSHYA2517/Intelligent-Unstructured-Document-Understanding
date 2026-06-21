# Changes — Hybrid GraphRAG (`injestion.py`)

This document summarizes the 5 new capabilities added to the single-file GraphRAG
app and what must be installed before running. All existing behavior (single-doc
batch build via `build_hybrid_index`, the CLI, async concurrency, and
`post_with_backoff` retry handling) is preserved.

## Setup / Installation

One new dependency was added to `requirements.txt`:

- `streamlit-agraph` — interactive contributing-subgraph visualization in the
  "Why this answer" panel. If it is not installed, the app still works and falls
  back to a text/edge-list view.

Install everything (the `.venv` already exists):

```bash
uv pip install -r requirements.txt
```

Run the app:

```bash
# Streamlit UI
streamlit run injestion.py

# CLI fallback (unchanged)
python injestion.py --pdf test-5.pdf --build --query "..."
```

`NVIDIA_API_KEY` is still read from `.env` (or the sidebar) via `get_api_key()`.
Model names / URLs remain the constants at the top of the file (REASONING_MODEL =
llama-4-maverick, VISION_MODEL = llama-3.2-90b-vision, EMBED_MODEL = nv-embed-v1,
RERANK_MODEL = rerank-qa-mistral-4b).

---

## 1) Cross-Document QA

Ingestion is now **additive** — multiple PDFs accumulate into one corpus.

- `HybridIndex` gained corpus state: `sources`, `markdown_by_source`,
  `vision_by_source`, plus a `source_names()` helper.
- `build_document_chunks(..., source_name=...)` — every chunk's `source` is now
  the document (PDF) name, enabling per-document filtering. `chunk_id` seeding
  includes the source so identical content across docs does not collide.
- `vectorize_chunks` uses `collection.upsert` (idempotent) and accepts an
  existing collection; `reset_collection=False` keeps prior documents.
- `build_knowledge_graph` accumulates into one persistent `nx.DiGraph`:
  - `_register_entity_source()` maintains a `source_documents` set per entity.
  - `link_cross_document_entities()` adds `document::<name>` nodes and
    `RELATED_TO` edges between any two sources that share ≥1 entity.
  - Cross-document entity linking is automatic: `deterministic_entity_uuid` is
    source-independent, so the same canonical name from doc A and doc B maps to
    one node.
- `build_hybrid_index(..., existing_index=..., source_name=...)` extends an
  existing corpus instead of rebuilding.
- `retrieve_and_rerank(..., source_filter=...)` filters Chroma by `source`.
- `answer_query` synthesizes across sources, cites the document name per claim,
  and explicitly compares when evidence spans multiple documents.
- **UI:** multi-file uploader (`accept_multiple_files=True`), a "Clear corpus"
  button (the only place the collection is reset), and a scope selector
  (All documents / a specific document).

## 2) Explainability Mode (extended)

- `synthesize_cited_answer` / `answer_query`: the model attaches inline
  `[S1], [S2]…` citation markers to every factual claim (no raw chunk IDs in
  prose).
- Returns a `sources` list (`build_sources_list`): per `S#` — document, chunk_id,
  title/section, sequence, rerank score, and a short snippet.
- Returns a `contributing_subgraph` (`build_contributing_subgraph`): only the
  nodes/edges connecting the cited chunks to their entities — not the full 1-hop
  dump.
- **UI:** answer rendered with `[S#]` markers; a "Sources" expander; a
  "Why this answer" expander visualizing the contributing subgraph
  (streamlit-agraph, with a text fallback).

## 3) Agentic Multi-Hop Reasoning

- `plan_subquestions(nim, query)` → `{"multi_hop": bool, "sub_questions": [...]}`
  via one JSON-forced `chat_completion` (temperature 0). Parses defensively
  (`_coerce_sub_questions` handles string or object items); returns `[]` for
  simple questions.
- `answer_query_agentic(...)`: if not multi-hop → falls back to `answer_query`.
  Otherwise retrieves + builds a localized subgraph + a grounded sub-answer per
  sub-question, then one final `chat_completion` synthesizes them into a single
  cited answer (preserving `[S#]`/document citations, noting conflicts). Returns
  `{answer, sub_questions, sub_answers, ranked_chunks (union), graph_context (union)}`.
- **UI:** an "Agentic multi-hop" toggle; decomposed sub-questions + sub-answers
  shown in an expander.

## 4) Real-Time / Streaming Ingestion (fast-path / slow-path)

- `stream_ingest(pdf_path, nim, on_event, ...)` implements progressive ingestion:
  - **Fast path:** as Docling yields each page range, text chunks are built,
    embedded, and `upsert`ed into Chroma immediately — searchable within ~1
    round-trip; emits `on_event("indexed", ...)`.
  - **Slow path:** chunks are pushed onto a bounded `asyncio.Queue`; a pool of
    `N` workers (N = concurrency limit) runs `extract_graph_facts` and
    incrementally updates the **shared** `DiGraph` (eventual consistency); emits
    `on_event("graph_updated", ...)`.
  - The document is queryable as soon as the first chunks are embedded; the graph
    keeps enriching. Respects rate limits via `post_with_backoff` + the
    `asyncio.Semaphore`; Docling conversion runs off the event loop via
    `asyncio.to_thread`.
- Helpers `add_chunk_node` and `apply_graph_facts` were factored out of
  `build_knowledge_graph` and are shared by both the batch and streaming paths;
  `chunk_markdown_text` chunks in-memory markdown.
- **UI:** an "Ingestion mode" radio (Batch / Streaming). Streaming shows live
  per-chunk progress via `st.status`/`st.write`. Batch remains the default and is
  the only path that runs the full VLM chart/table extraction.

## 5) Adversarial Robustness (extended)

- `classify_query(nim, query)` →
  `{"category": "answerable|ambiguous|unanswerable|out_of_scope|trick", "reason"}`
  via one JSON-forced `chat_completion`.
- `answer_query_robust(...)` is the guarded query path:
  - **Hard evidence gate (programmatic):** after `retrieve_and_rerank`, if no
    chunks survive or the top rerank score is below the configurable threshold,
    the answer LLM is **not** called — returns a fixed graceful message
    (`GRACEFUL_NO_EVIDENCE`). Prevents hallucinated answers.
  - `ambiguous` → returns a clarifying question seeded by top retrieved entities.
  - `out_of_scope` / `trick` → honestly states it is not supported by the docs.
  - Strengthened answer prompt: "Use ONLY the provided context. If the context
    lacks the answer, say so. If sources conflict, present both and cite them.
    Do not speculate."
  - A single retrieval pass feeds both the gate and synthesis (no double
    embed/rerank).
- **UI:** detected category + a confidence indicator (top rerank score), and a
  sidebar "Evidence gate (min rerank score)" slider.

---

## Files touched

- `injestion.py` — all feature logic + Streamlit UI wiring.
- `requirements.txt` — added `streamlit-agraph`.

## Verification performed

- `python -c "import injestion"` — imports cleanly after each feature.
- `streamlit run injestion.py` (headless) — serves HTTP 200 with no startup
  errors.
