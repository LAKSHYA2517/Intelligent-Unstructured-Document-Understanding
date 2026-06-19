"""Tests for Stage 10 (hybrid retrieval) — src/retrieval/hybrid_retriever.py.

Pure tests for Reciprocal Rank Fusion and the full-text query builder, plus a
slow end-to-end test that ingests the corpus into a dedicated graph and checks
the three searches fuse into relevant, scope-filterable results.
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


# --------------------------------------------------------------------------- #
# Pure: RRF + query builder
# --------------------------------------------------------------------------- #
def test_reciprocal_rank_fusion_orders_and_tracks_methods() -> None:
    rankings = {
        "vector": ["a", "b", "c"],
        "bm25": ["b", "a", "d"],
        "graph": ["a"],
    }
    fused = HybridRetriever._reciprocal_rank_fusion(rankings)
    ids = [cid for cid, _s, _m in fused]
    # 'a' appears in all three (and near the top) → highest fused score.
    assert ids[0] == "a"
    by_id = {cid: (score, methods) for cid, score, methods in fused}
    assert set(by_id["a"][1]) == {"vector", "bm25", "graph"}
    assert by_id["a"][0] > by_id["b"][0] > by_id["c"][0]
    # Score is the sum of 1/(rank+60) across methods.
    assert by_id["a"][0] == pytest.approx(1 / 61 + 1 / 62 + 1 / 61)


def test_fulltext_query_builder() -> None:
    q = HybridRetriever._to_fulltext_query("What is Acme's quarterly revenue?")
    terms = q.split(" | ")
    assert "Acme" in terms and "revenue" in terms and "quarterly" in terms
    assert "is" not in terms  # short stopword-ish tokens dropped
    assert HybridRetriever._to_fulltext_query("a an of") == ""


# --------------------------------------------------------------------------- #
# Real end-to-end retrieval
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def populated_retriever(corpus_pdfs):
    """Ingest the corpus into a dedicated graph and return a retriever + doc ids."""
    from src.graph.falkordb_client import FalkorDBClient
    from src.graph.graph_publisher import GraphPublisher
    from src.pipeline.ingestion_pipeline import IngestionPipeline
    from src.retrieval.embedder import Embedder

    db = FalkorDBClient(graph_name="test_retrieval_graph")
    db.clear_graph()
    db.ensure_indexes()
    embedder = Embedder()
    pipeline = IngestionPipeline(publisher=GraphPublisher(client=db), embedder=embedder)
    batch = pipeline.ingest_batch(corpus_pdfs)
    retriever = HybridRetriever(embedder=embedder, client=db)
    return retriever, batch


@pytest.mark.slow
def test_retrieves_relevant_fused_chunks(populated_retriever) -> None:
    retriever, _ = populated_retriever
    result = retriever.retrieve("What is Acme Corporation's revenue and who is its CEO?")
    assert result.chunks, "expected retrieved chunks"
    assert "Acme Corporation" in result.query_entities
    # The top result should be fused from more than one method.
    assert len(result.chunks[0].methods) >= 2
    # Some chunk should mention Jane Smith (the CEO) — answers the query.
    assert any("Jane Smith" in c.text for c in result.chunks)
    # Graph traversal reached chunks at >=1 hop.
    assert result.max_hops >= 1
    assert result.method_counts["vector"] > 0


@pytest.mark.slow
def test_scope_filter_restricts_to_one_document(populated_retriever) -> None:
    retriever, batch = populated_retriever
    doc_id = batch.documents[0].doc_id
    result = retriever.retrieve("revenue and leadership", scope_doc_id=doc_id)
    assert result.chunks
    assert all(c.document_id == doc_id for c in result.chunks)


@pytest.mark.slow
def test_vector_search_contributes_to_results(populated_retriever) -> None:
    """Vector search runs and contributes chunks to the fused results.

    HNSW recall is approximate and query-dependent on a tiny (14-node) graph, so
    the only reliable invariant is that vector retrieval participates in the
    fusion; the efRuntime + over-fetch tuning improves recall on real corpora.
    """
    retriever, _ = populated_retriever
    result = retriever.retrieve("revenue growth across regions and quarters")
    assert result.method_counts["vector"] >= 1
    assert any("vector" in c.methods for c in result.chunks)
