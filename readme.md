# Multi-Modal Semantic Integration for Intelligent Unstructured Document Understanding

An end-to-end **GraphRAG** system that batch-ingests PDF/DOCX documents into a
FalkorDB knowledge graph (POLE+O with rich cross-modal, structural, entity and
cross-document edges) and answers natural-language questions with grounded,
cited answers. The system is **domain-agnostic** (finance, legal, medical,
technical, general) and **model-agnostic** — every model name and threshold is
read from `.env` at runtime; swap models by editing `.env` only.

## Pipeline

**Ingestion (per batch):** parse (Docling) → tiered image processing (vision
model) → element-aware chunking → domain detection → NER + co-reference
(spaCy/`en_core_web_trf` + fastcoref) → GLiNER entities (runtime domain labels) →
LLM relation triples → embedding → graph publishing. After all documents are in,
**global entity resolution** (Splink) merges entities across documents and writes
cross-document edges (`SAME_AS`, `RELATED_TO`, `CORROBORATES`, `CONTRADICTS`).

**Query (real-time):** query understanding (type + scope) → hybrid retrieval
(HNSW vector + BM25 full-text + 2-hop graph traversal, fused with Reciprocal Rank
Fusion) → cross-encoder re-ranking with an adversarial gate → element-aware,
cited answer generation (with multi-hop decomposition and visual `IMAGE_PATH`
support).

## Requirements

- Python 3.11, [`uv`](https://docs.astral.sh/uv/) (package manager — **never** pip)
- Docker (FalkorDB), [Ollama](https://ollama.com/)

## Setup

```bash
# 1. FalkorDB (in-memory; restart wipes the graph)
docker run -d -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest

# 2. Ollama models (defaults — lightweight, runs anywhere)
ollama pull phi3.5:3.8b        # LLM
ollama pull moondream          # vision
ollama pull nomic-embed-text   # embeddings

# 3. Python environment (downloads spaCy/GLiNER/cross-encoder models on first use)
uv sync

# 4. Validate everything is reachable
uv run python -m src.config
```

To use stronger models on a capable machine, edit `.env` only (e.g.
`OLLAMA_LLM_MODEL=llama3.1:8b`, `OLLAMA_VISION_MODEL=llava:7b`) — zero code changes.

## Run the app

```bash
uv run streamlit run app/streamlit_app.py
```

Tab **Ingest**: upload PDFs/DOCX, click *Start Ingestion*, watch live per-stage
progress (every number comes from pipeline return values). Tab **Query**: ask
questions and get cited, grounded answers with Sources and Graph-context panels.

## Run any stage directly

```bash
uv run python -m src.pipeline.ingestion_pipeline data/uploads/doc1.pdf data/uploads/doc2.pdf
uv run python -m src.pipeline.query_pipeline "Who acquired Beta Labs and who leads that company?"
```

## Tests

```bash
uv run pytest                 # full suite (includes slow end-to-end tests)
uv run pytest -m "not slow"   # fast unit tests only
```

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `FALKORDB_HOST` / `FALKORDB_PORT` | FalkorDB connection |
| `OLLAMA_BASE_URL` | Ollama endpoint |
| `OLLAMA_LLM_MODEL` / `OLLAMA_VISION_MODEL` / `OLLAMA_EMBED_MODEL` | Ollama models |
| `CROSS_ENCODER_MODEL` | Re-ranking model |
| `GLINER_MODEL` | Entity extraction model |
| `SPLINK_MERGE_THRESHOLD` | Entity resolution merge threshold |
| `ADVERSARIAL_SCORE_THRESHOLD` | Minimum relevance to answer |
| `GLINER_CONFIDENCE_THRESHOLD` | GLiNER entity confidence cut-off |
| `CHUNK_MAX_TOKENS` | Chunk size cap |

## Architecture

```
src/
  config.py                    runtime config + startup validation
  ingestion/  parser, image_processor, chunker
  extraction/ domain_detector, ner, gliner_extractor, relation_extractor
  resolution/ entity_resolution (Splink)
  graph/      falkordb_client, graph_publisher
  retrieval/  embedder, hybrid_retriever, reranker, answer_generator
  pipeline/   ingestion_pipeline, query_pipeline
app/streamlit_app.py           demo UI
```
