"""Streamlit demo UI for the GraphRAG pipeline.

Two tabs — batch ingestion and querying — over the FalkorDB knowledge graph.
Every number, filename and stat shown comes from pipeline return values
(``ParseResult``, ``ImageProcessResult``, ``ChunkResult``, ``DomainResult``,
``ExtractionResult``, ``ResolutionResult``) or a live FalkorDB query; nothing is
hardcoded. Heavy model-backed pipelines are built once via ``st.cache_resource``.

Run with: ``uv run streamlit run app/streamlit_app.py``
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import requests
import streamlit as st

# Make the project importable when launched via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure progress formatting — every value comes from the stage's return object
# --------------------------------------------------------------------------- #
def format_stage(stage: str, payload: object) -> str:
    """Render a progress line for a pipeline stage from its return value only.

    Args:
        stage: Stage identifier emitted by the ingestion pipeline.
        payload: The stage's typed result object (or an int for cross-document).

    Returns:
        A human-readable progress line built entirely from ``payload`` fields.
    """
    if stage == "parse":
        return (f"✅ Parsing complete ({payload.page_count} pages, "
                f"{payload.table_count} tables, {payload.figure_count} figures)")
    if stage == "image":
        return (f"✅ Image processing ({payload.vision_calls} vision model calls, "
                f"{payload.skipped} skipped)")
    if stage == "chunk":
        return f"✅ Chunking ({payload.chunk_count} chunks)"
    if stage == "domain":
        return f"✅ Domain detected: {payload.domain} (confidence {payload.confidence:.2f})"
    if stage == "extract":
        return (f"✅ NER + relations ({payload.entity_count} entities, "
                f"{payload.triple_count} triples)")
    if stage == "embed":
        return f"✅ Embedded {payload.embedded_count} chunks (dim {payload.dim})"
    if stage == "publish":
        return (f"✅ Graph published ({payload.sections} sections, {payload.chunks} chunks, "
                f"{payload.figures} figures, {payload.entities} entities, "
                f"{payload.relationships} relations)")
    if stage == "resolution":
        return f"🔗 Entity resolution complete ({payload.merged_count} entities merged)"
    if stage == "cross_document":
        return f"✅ Cross-document edges created ({payload} edges)"
    return f"… {stage}"


# --------------------------------------------------------------------------- #
# Cached services (heavy models loaded once)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_services():
    """Build and cache the shared client, ingestion and query pipelines."""
    from src.graph.falkordb_client import FalkorDBClient
    from src.graph.graph_publisher import GraphPublisher
    from src.pipeline.ingestion_pipeline import IngestionPipeline
    from src.pipeline.query_pipeline import QueryPipeline
    from src.retrieval.answer_generator import AnswerGenerator
    from src.retrieval.embedder import Embedder
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker

    client = FalkorDBClient()
    client.ensure_indexes()
    embedder = Embedder()  # one model, shared by ingestion and query
    ingestion = IngestionPipeline(publisher=GraphPublisher(client=client), embedder=embedder)
    query = QueryPipeline(
        retriever=HybridRetriever(embedder=embedder, client=client),
        reranker=Reranker(), generator=AnswerGenerator(), client=client,
    )
    return SimpleNamespace(client=client, ingestion=ingestion, query=query)


# --------------------------------------------------------------------------- #
# Live status / graph queries
# --------------------------------------------------------------------------- #
def _normalise_model(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def live_status(client) -> dict:
    """Check FalkorDB, Ollama and the configured models live."""
    status = {"falkordb": False, "ollama": False, "models": {}}
    try:
        status["falkordb"] = client.ping()
    except Exception:  # noqa: BLE001
        status["falkordb"] = False

    available: set[str] = set()
    try:
        resp = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        available = {_normalise_model(m["name"]) for m in resp.json().get("models", [])}
        status["ollama"] = True
    except Exception:  # noqa: BLE001
        status["ollama"] = False

    for model in (config.ollama_llm_model, config.ollama_vision_model, config.ollama_embed_model):
        status["models"][model] = _normalise_model(model) in available
    return status


def document_registry(client) -> list[dict]:
    """Return the Document nodes currently in the graph."""
    rows = client.query(
        """
        MATCH (d:Document)
        RETURN d.name, d.domain, d.page_count, d.chunk_count
        ORDER BY d.name
        """,
        read_only=True,
    )
    return [
        {"name": r[0], "domain": r[1], "pages": r[2], "chunks": r[3], "status": "ready"}
        for r in rows
    ]


def graph_stats(client) -> dict:
    """Return live totals across the whole graph."""
    def _count(cypher: str) -> int:
        rows = client.query(cypher, read_only=True)
        return int(rows[0][0]) if rows else 0

    return {
        "documents": _count("MATCH (d:Document) RETURN count(d)"),
        "chunks": _count("MATCH (c:Chunk) RETURN count(c)"),
        "entities": _count("MATCH (e:Entity) RETURN count(e)"),
        "relationships": _count("MATCH (:Entity)-[r]->(:Entity) RETURN count(r)"),
    }


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def render_sidebar(services) -> None:
    """Render the sidebar: title, status, document registry, clear button."""
    st.sidebar.title("📚 GraphRAG Document Understanding")
    st.sidebar.caption(
        "Batch-ingest PDF/DOCX into a FalkorDB knowledge graph, then ask "
        "grounded, cited questions across documents."
    )

    st.sidebar.subheader("System status")
    status = live_status(services.client)
    st.sidebar.write(f"FalkorDB: {'✅ connected' if status['falkordb'] else '❌ unreachable'}")
    st.sidebar.write(f"Ollama: {'✅ running' if status['ollama'] else '❌ unreachable'}")
    for model, ok in status["models"].items():
        st.sidebar.write(f"Model `{model}`: {'✅' if ok else '❌'}")

    st.sidebar.subheader("Document registry")
    registry = document_registry(services.client)
    if registry:
        st.sidebar.dataframe(registry, hide_index=True, use_container_width=True)
    else:
        st.sidebar.info("No documents ingested yet.")

    if st.sidebar.button("🗑️ Clear all documents", use_container_width=True):
        services.client.clear_graph()
        st.session_state["ingested"] = False
        st.sidebar.success("Graph cleared.")
        st.rerun()


def render_ingest_tab(services) -> None:
    """Tab 1 — batch document ingestion with live progress."""
    st.header("Ingest Documents")
    uploaded = st.file_uploader(
        "Upload PDF or DOCX files (multiple allowed)",
        type=["pdf", "docx"], accept_multiple_files=True,
    )
    if not uploaded:
        return

    if st.button("🚀 Start Ingestion", type="primary"):
        saved: list[tuple[str, str]] = []
        for file in uploaded:
            dest = config.uploads_dir / file.name
            dest.write_bytes(file.getbuffer())
            saved.append((str(dest), file.name))

        progress = st.container()

        def on_stage(index, count, name, stage, payload):
            with progress:
                if name and stage == "parse":
                    st.write(f"**[Doc {index + 1}/{count}] {name}**")
                st.write(format_stage(stage, payload))

        with st.status("Running batch ingestion…", expanded=True):
            batch = services.ingestion.ingest_batch(saved, on_stage=on_stage)
        st.success("✅ System ready for queries")
        st.session_state["ingested"] = True

        stats = graph_stats(services.client)
        cols = st.columns(4)
        cols[0].metric("Documents", stats["documents"])
        cols[1].metric("Chunks", stats["chunks"])
        cols[2].metric("Entities", stats["entities"])
        cols[3].metric("Relationships", stats["relationships"])
        st.caption(
            f"Batch totals — chunks: {batch.total_chunks}, entities: {batch.total_entities}, "
            f"triples: {batch.total_triples}, merged: {batch.resolution_result.merged_count}, "
            f"cross-doc edges: {batch.cross_doc_edges_written}"
        )


def render_query_tab(services) -> None:
    """Tab 2 — ask grounded questions (enabled after ingestion)."""
    st.header("Query Documents")
    if not graph_stats(services.client)["documents"]:
        st.info("Ingest at least one document first.")
        return

    query = st.text_input("Ask anything about your documents…")
    if not (query and st.button("🔍 Submit", type="primary")):
        return

    with st.spinner("Retrieving and reasoning…"):
        response = services.query.answer(query)
    analysis = response.analysis
    answer = response.answer_result

    st.caption(
        f"Type: **{analysis.query_type}** · Scope: **{analysis.scope}**"
        + (f" ({analysis.scope_doc_name})" if analysis.scope_doc_name else "")
    )
    st.markdown(answer.answer)

    for path in answer.image_paths:
        if Path(path).exists():
            st.image(path, caption=path)

    if answer.sub_questions:
        with st.expander("Sub-questions (multi-hop)"):
            for q, a in zip(answer.sub_questions, answer.sub_answers or []):
                st.markdown(f"**{q}**\n\n{a}")

    with st.expander(f"Sources ({len(answer.citations)})"):
        for cite in answer.citations:
            st.markdown(
                f"`{cite['element_type']}` · **{cite['document_name']}** "
                f"· page {cite['page_number']} · `{cite['chunk_id']}` "
                f"· confidence {cite['confidence']:.2f}"
            )

    if response.retrieval_result is not None:
        rr = response.retrieval_result
        with st.expander("Graph context"):
            st.write(f"Entities matched: {rr.query_entities or '—'}")
            st.write(f"Edge types used: {rr.edge_types_used or '—'}")
            st.write(f"Max hops: {rr.max_hops}")
            st.write(f"Method counts: {rr.method_counts}")


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(page_title="GraphRAG Document Understanding", layout="wide")
    st.session_state.setdefault("ingested", False)
    services = get_services()
    render_sidebar(services)
    tab_ingest, tab_query = st.tabs(["📥 Ingest Documents", "💬 Query Documents"])
    with tab_ingest:
        render_ingest_tab(services)
    with tab_query:
        render_query_tab(services)


if __name__ == "__main__":
    main()
