"""Tests for Stage 1-8 orchestration — src/pipeline/ingestion_pipeline.py.

A fully-stubbed test verifies orchestration order, progress callbacks, batch
accumulation feeding the resolver, and result totals; a slow test runs the real
pipeline over the corpus and confirms the graph is populated.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.pipeline.ingestion_pipeline import IngestionPipeline


# --------------------------------------------------------------------------- #
# Stub components
# --------------------------------------------------------------------------- #
class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.resolver_entities = None


def _make_stubs(rec: _Recorder):
    def chunk_obj(doc, i):
        return SimpleNamespace(chunk_id=f"{doc}_c{i}", text=f"text {i} of {doc}")

    class Parser:
        def parse(self, path, doc_name=None):
            rec.calls.append("parse")
            return SimpleNamespace(doc_id=f"id_{doc_name}", doc_name=doc_name,
                                   elements=[], page_count=2, table_count=1, figure_count=1)

    class Image:
        def process(self, pr):
            rec.calls.append("image")
            return SimpleNamespace(processed_figures=[], vision_calls=1, skipped=0,
                                   skipped_element_ids=[])

    class Chunker:
        def chunk(self, pr, ir):
            rec.calls.append("chunk")
            chunks = [chunk_obj(pr.doc_id, i) for i in range(3)]
            return SimpleNamespace(chunks=chunks, chunk_count=len(chunks))

    class Domain:
        def detect(self, cr):
            rec.calls.append("domain")
            return SimpleNamespace(domain="finance", confidence=0.9)

    class NER:
        def extract(self, cr):
            rec.calls.append("ner")
            return SimpleNamespace(entities=[f"e_{cr.chunks[0].chunk_id}"], entity_count=1)

    class GLiNER:
        def extract(self, cr, dom, prior_entities=None):
            rec.calls.append("gliner")
            ents = list(prior_entities or []) + [f"g_{cr.chunks[0].chunk_id}"]
            return SimpleNamespace(entities=ents, triples=[], entity_count=len(ents),
                                   triple_count=0)

    class Relation:
        def extract(self, cr, er):
            rec.calls.append("relation")
            er.triples = ["t1", "t2"]
            er.triple_count = 2
            return er

    class Embedder:
        def embed(self, cr):
            rec.calls.append("embed")
            return SimpleNamespace(embedded_count=cr.chunk_count, dim=768, model="m")

    class Publisher:
        def publish_document(self, pr, cr, er, domain):
            rec.calls.append("publish")
            assert domain == "finance"
            return SimpleNamespace(document_id=pr.doc_id, chunks=cr.chunk_count)

        def publish_cross_document_edges(self, res):
            rec.calls.append("cross_doc")
            return len(res.cross_doc_edges)

        def apply_canonical_metadata(self, canonicals):
            rec.calls.append("canonical")
            return len(canonicals)

    class Resolver:
        def resolve_all(self, entities, chunk_text=None, classify_pairs=True):
            rec.calls.append("resolve")
            rec.resolver_entities = list(entities)
            rec.resolver_chunk_text = dict(chunk_text or {})
            return SimpleNamespace(merged_count=2,
                                   canonical_entities=["c1", "c2", "c3"],
                                   cross_doc_edges=[{"type": "SAME_AS"}, {"type": "RELATED_TO"}])

    return dict(parser=Parser(), image_processor=Image(), chunker=Chunker(),
                domain_detector=Domain(), ner=NER(), gliner=GLiNER(),
                relation_extractor=Relation(), embedder=Embedder(),
                publisher=Publisher(), resolver=Resolver())


# --------------------------------------------------------------------------- #
# Orchestration (stubbed)
# --------------------------------------------------------------------------- #
def test_orchestration_order_and_totals() -> None:
    rec = _Recorder()
    pipeline = IngestionPipeline(**_make_stubs(rec))
    events: list[tuple[str, str]] = []
    batch = pipeline.ingest_batch(
        ["/tmp/a.pdf", "/tmp/b.pdf"],
        on_stage=lambda i, n, name, stage, payload: events.append((name, stage)),
    )

    per_doc = ["parse", "image", "chunk", "domain", "ner", "gliner", "relation",
               "embed", "publish"]
    # Each document runs the full per-doc sequence, in order, before the next.
    assert rec.calls[:9] == per_doc
    assert rec.calls[9:18] == per_doc
    # Global resolution + cross-doc publishing happen once, after all documents.
    assert rec.calls[18:] == ["resolve", "cross_doc", "canonical"]

    assert batch.document_count == 2
    assert batch.total_chunks == 6  # 3 per doc
    assert batch.total_entities == 4  # 2 per doc (1 ner + 1 gliner)
    assert batch.total_triples == 4  # 2 per doc
    assert batch.cross_doc_edges_written == 2
    assert batch.canonical_nodes_updated == 3


def test_progress_events_emitted_per_stage() -> None:
    rec = _Recorder()
    pipeline = IngestionPipeline(**_make_stubs(rec))
    events: list[tuple[str, str]] = []
    pipeline.ingest_batch(
        [("/tmp/a.pdf", "a.pdf")],
        on_stage=lambda i, n, name, stage, payload: events.append((name, stage)),
    )
    stages = [stage for _, stage in events]
    assert stages == ["parse", "image", "chunk", "domain", "extract", "embed",
                      "publish", "resolution", "cross_document"]


def test_batch_accumulates_entities_for_resolver() -> None:
    rec = _Recorder()
    pipeline = IngestionPipeline(**_make_stubs(rec))
    pipeline.ingest_batch(["/tmp/a.pdf", "/tmp/b.pdf"])
    # Resolver receives every entity from both documents.
    assert len(rec.resolver_entities) == 4
    # And the chunk-text map spans both documents' chunks.
    assert len(rec.resolver_chunk_text) == 6


def test_callback_errors_do_not_break_ingestion() -> None:
    rec = _Recorder()
    pipeline = IngestionPipeline(**_make_stubs(rec))

    def _boom(*a):
        raise RuntimeError("UI exploded")

    batch = pipeline.ingest_batch(["/tmp/a.pdf"], on_stage=_boom)
    assert batch.document_count == 1  # ingestion completed despite callback errors


# --------------------------------------------------------------------------- #
# Real end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_batch_ingestion_populates_graph(corpus_pdfs) -> None:
    from src.graph.falkordb_client import FalkorDBClient

    db = FalkorDBClient(graph_name="test_ingest_graph")
    db.clear_graph()
    db.ensure_indexes()
    from src.graph.graph_publisher import GraphPublisher

    pipeline = IngestionPipeline(publisher=GraphPublisher(client=db))
    batch = pipeline.ingest_batch(corpus_pdfs)

    assert batch.document_count == 2
    assert batch.total_chunks > 0 and batch.total_entities > 0
    assert batch.resolution_result.merged_count >= 3
    assert batch.cross_doc_edges_written >= 1
    # The graph really has two documents and a SAME_AS link across them.
    assert int(db.query("MATCH (d:Document) RETURN count(d)", read_only=True)[0][0]) == 2
    assert int(db.query("MATCH (:Entity)-[:SAME_AS]->(:Entity) RETURN count(*)", read_only=True)[0][0]) >= 1
    assert int(db.query("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c)", read_only=True)[0][0]) > 0
