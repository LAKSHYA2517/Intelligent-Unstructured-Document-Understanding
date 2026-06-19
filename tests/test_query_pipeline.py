"""Tests for Stage 9 + orchestration — src/pipeline/query_pipeline.py.

Pure tests for query-type detection and document-scope matching, a stubbed
orchestration test (including the multi-hop retrieve_fn), and a slow end-to-end
test over a freshly ingested corpus.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.pipeline.query_pipeline import QueryPipeline


# --------------------------------------------------------------------------- #
# Query-type detection (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query, qtype, visual, comparative, multi_hop",
    [
        ("Show me the revenue chart", "visual", True, False, False),
        ("Compare revenue across documents", "comparative", False, True, False),
        ("Who acquired Beta Labs?", "multi_hop", False, False, True),
        ("What is the total revenue?", "standard", False, False, False),
        ("Display a diagram comparing both documents", "multi_hop"[:0] or "comparative", True, True, False),
    ],
)
def test_query_type_detection(query, qtype, visual, comparative, multi_hop) -> None:
    pipe = QueryPipeline(client=_StubDB([]))
    a = pipe.understand(query)
    assert a.visual is visual
    assert a.comparative is comparative
    assert a.multi_hop is multi_hop
    assert a.query_type == qtype


# --------------------------------------------------------------------------- #
# Scope matching (pure)
# --------------------------------------------------------------------------- #
def test_document_scope_matching() -> None:
    docs = [("id_acme", "acme_annual_report.pdf"), ("id_globex", "globex_q3_filing.pdf")]
    match = QueryPipeline._match_document
    # Exact filename mention.
    assert match("summarise acme_annual_report.pdf", docs) == ("id_acme", "acme_annual_report.pdf")
    # Distinctive stem tokens.
    assert match("what does the acme annual report say?", docs) == ("id_acme", "acme_annual_report.pdf")
    # A single shared token must not force single-doc scope.
    assert match("what was the annual revenue?", docs) is None
    # No document reference → cross-doc.
    assert match("who is the CEO?", docs) is None


def test_scope_resolution_uses_graph_documents() -> None:
    db = _StubDB([["id_globex", "globex_q3_filing.pdf"]])
    a = QueryPipeline(client=db).understand("summarise the globex q3 filing")
    assert a.scope == "single_doc" and a.scope_doc_id == "id_globex"


# --------------------------------------------------------------------------- #
# Orchestration (stubbed)
# --------------------------------------------------------------------------- #
class _StubDB:
    def __init__(self, doc_rows):
        self._rows = doc_rows

    def query(self, cypher, params=None, read_only=False):
        return self._rows


def test_orchestration_runs_retrieve_rerank_generate() -> None:
    calls = {"retrieve": [], "rerank": [], "generate": 0}

    class Retriever:
        def retrieve(self, q, scope_doc_id=None):
            calls["retrieve"].append((q, scope_doc_id))
            return SimpleNamespace(chunks=[f"chunk_for_{q}"])

    class Reranker:
        def rerank(self, q, chunks):
            calls["rerank"].append(q)
            return SimpleNamespace(chunks=chunks, adversarial=False, max_score=0.9)

    class Generator:
        def generate(self, q, rr, *, query_type, visual, retrieve_fn):
            calls["generate"] += 1
            # Multi-hop path should be able to call retrieve_fn for sub-questions.
            if query_type == "multi_hop":
                retrieve_fn("sub q")
            return SimpleNamespace(answer="A", adversarial=False, sub_questions=None)

    pipe = QueryPipeline(retriever=Retriever(), reranker=Reranker(),
                         generator=Generator(), client=_StubDB([]))
    resp = pipe.answer("Who acquired Beta Labs?")  # multi_hop
    assert resp.analysis.query_type == "multi_hop"
    assert calls["generate"] == 1
    # Main query + one sub-question retrieval.
    assert ("Who acquired Beta Labs?", None) in calls["retrieve"]
    assert ("sub q", None) in calls["retrieve"]


# --------------------------------------------------------------------------- #
# Real end-to-end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def populated_pipeline(corpus_pdfs):
    from src.graph.falkordb_client import FalkorDBClient
    from src.graph.graph_publisher import GraphPublisher
    from src.pipeline.ingestion_pipeline import IngestionPipeline
    from src.retrieval.embedder import Embedder
    from src.retrieval.hybrid_retriever import HybridRetriever

    db = FalkorDBClient(graph_name="test_query_graph")
    db.clear_graph()
    db.ensure_indexes()
    embedder = Embedder()
    IngestionPipeline(publisher=GraphPublisher(client=db), embedder=embedder).ingest_batch(corpus_pdfs)
    pipe = QueryPipeline(retriever=HybridRetriever(embedder=embedder, client=db), client=db)
    return pipe


@pytest.mark.slow
def test_real_standard_query(populated_pipeline) -> None:
    resp = populated_pipeline.answer("What was Acme Corporation's revenue in 2024?")
    assert not resp.answer_result.adversarial
    assert "540" in resp.answer_result.answer
    assert resp.answer_result.citations


@pytest.mark.slow
def test_real_adversarial_query(populated_pipeline) -> None:
    resp = populated_pipeline.answer("What is the recipe for sourdough bread?")
    assert resp.answer_result.adversarial is True


@pytest.mark.slow
def test_real_multi_hop_query(populated_pipeline) -> None:
    resp = populated_pipeline.answer("Who acquired Beta Labs and who leads that company?")
    assert resp.analysis.multi_hop is True
    assert resp.answer_result.sub_questions
    assert "Acme" in resp.answer_result.answer
