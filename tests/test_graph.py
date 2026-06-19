"""Tests for Stage 8 (graph publishing) — src/graph/graph_publisher.py.

Uses hand-built synthetic ParseResult/ChunkResult/ExtractionResult inputs (no
docling/LLM) published into a dedicated FalkorDB test graph, then asserts the
node/edge inventory, vector + table_json storage, typed entity relations,
idempotency, and cross-document edge application.
"""

from __future__ import annotations

import json

import pytest

from src.extraction.gliner_extractor import ExtractionResult, Triple
from src.extraction.ner import Entity
from src.graph.falkordb_client import FalkorDBClient
from src.graph.graph_publisher import GraphPublisher
from src.ingestion.chunker import Chunk, ChunkResult
from src.ingestion.parser import ParsedElement, ParseResult

_DOC_ID = "testdoc_graph01"
_EMB = [0.01] * 768


def _el(eid, etype, text, *, self_ref, level=None, parent_ref=None, table_json=None,
        image_path=None) -> ParsedElement:
    return ParsedElement(
        element_id=eid, element_type=etype, text=text, page_number=1,
        reading_order=0, level=level, self_ref=self_ref, parent_ref=parent_ref,
        table_json=table_json, image_path=image_path,
    )


def _ch(idx, etype, text, src, *, table_json=None, image_path=None) -> Chunk:
    return Chunk(
        chunk_id=f"{_DOC_ID}_chunk_{idx:04d}", text=text, table_json=table_json,
        element_type=etype, page_number=1, document_id=_DOC_ID,
        document_name="synthetic.pdf", section_title="Overview",
        position_in_doc=idx, source_element_id=src, image_path=image_path,
        embedding=list(_EMB),
    )


def _entity(name, etype, *chunk_ids, aliases=None) -> Entity:
    e = Entity(name=name, entity_type=etype, document_id=_DOC_ID,
               document_name="synthetic.pdf", aliases=aliases or [])
    for cid in chunk_ids:
        e.add_mention(cid, 1)
    if chunk_ids:
        e.first_chunk_id, e.first_page = chunk_ids[0], 1
    return e


def _build_synthetic():
    table_json = {"num_rows": 2, "num_cols": 2, "headers": ["A", "B"],
                  "rows": [["1", "2"]], "grid": [["A", "B"], ["1", "2"]]}
    elements = [
        _el("e0", "heading", "Overview", self_ref="#/h0", level=1),
        _el("e1", "paragraph", "Acme Corporation is based in New York.", self_ref="#/p1"),
        _el("e2", "table", "A | B", self_ref="#/t2", table_json=table_json),
        _el("e3", "figure", "Revenue chart", self_ref="#/f3", image_path="/tmp/fig.png"),
        _el("e4", "caption", "Figure 1: chart", self_ref="#/c4", parent_ref="#/f3"),
    ]
    pr = ParseResult(doc_id=_DOC_ID, doc_name="synthetic.pdf", elements=elements,
                     page_count=1, table_count=1, figure_count=1,
                     reading_order=[e.element_id for e in elements])
    chunks = [
        _ch(0, "heading", "Overview", "e0"),
        _ch(1, "paragraph", "Acme Corporation is based in New York.", "e1"),
        _ch(2, "table", "A | B", "e2", table_json=table_json),
        _ch(3, "figure", "Revenue chart", "e3", image_path="/tmp/fig.png"),
        _ch(4, "caption", "Figure 1: chart", "e4"),
    ]
    cr = ChunkResult(chunks=chunks)
    acme = _entity("Acme Corporation", "Organisation", chunks[1].chunk_id, aliases=["Acme"])
    ny = _entity("New York", "Location", chunks[1].chunk_id)
    triples = [
        Triple("Acme Corporation", "is_based_in", "New York", chunks[1].chunk_id, _DOC_ID),
        Triple("Jane Smith", "works_at", "Acme Corporation", chunks[1].chunk_id, _DOC_ID),
        Triple("Acme Corporation", "partnered_with", "New York", chunks[1].chunk_id, _DOC_ID),
    ]
    er = ExtractionResult(entities=[acme, ny], triples=triples)
    return pr, cr, er


@pytest.fixture(scope="module")
def db() -> FalkorDBClient:
    client = FalkorDBClient(graph_name="test_publish_graph")
    client.clear_graph()
    return client


@pytest.fixture(scope="module")
def published(db: FalkorDBClient):
    pr, cr, er = _build_synthetic()
    publisher = GraphPublisher(client=db)
    result = publisher.publish_document(pr, cr, er, domain="finance")
    return publisher, result


def _count(db, cypher: str) -> int:
    return int(db.query(cypher, read_only=True)[0][0])


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def test_node_inventory(db, published) -> None:
    _, result = published
    assert _count(db, "MATCH (d:Document) RETURN count(d)") == 1
    assert _count(db, "MATCH (c:Chunk) RETURN count(c)") == 5
    assert _count(db, "MATCH (f:Figure) RETURN count(f)") == 1
    assert _count(db, "MATCH (s:Section) RETURN count(s)") == 1
    assert _count(db, "MATCH (o:Organisation) RETURN count(o)") == 1
    assert _count(db, "MATCH (l:Location) RETURN count(l)") == 1
    # Every typed entity also carries the generic :Entity label.
    assert _count(db, "MATCH (o:Organisation) WHERE o:Entity RETURN count(o)") == 1
    assert result.chunks == 5 and result.figures == 1


def test_document_node_has_real_stats(db, published) -> None:
    row = db.query(
        "MATCH (d:Document {id:$id}) RETURN d.domain, d.page_count, d.chunk_count",
        {"id": _DOC_ID}, read_only=True,
    )[0]
    assert row == ["finance", 1, 5]


# --------------------------------------------------------------------------- #
# Chunk storage: vector + table_json
# --------------------------------------------------------------------------- #
def test_chunk_vector_and_table_json(db) -> None:
    has_vec = _count(
        db, "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c)"
    )
    assert has_vec == 5
    raw = db.query(
        "MATCH (c:Chunk {element_type:'table'}) RETURN c.table_json", read_only=True
    )[0][0]
    parsed = json.loads(raw)
    assert parsed["headers"] == ["A", "B"]


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
def test_structural_edges(db) -> None:
    assert _count(db, "MATCH (:Document)-[:HAS_CHUNK]->(:Chunk) RETURN count(*)") == 5
    assert _count(db, "MATCH (:Chunk)-[:PART_OF]->(:Document) RETURN count(*)") == 5
    assert _count(db, "MATCH (:Chunk)-[:PRECEDES]->(:Chunk) RETURN count(*)") == 4
    assert _count(db, "MATCH (:Section)-[:CONTAINS]->(:Chunk) RETURN count(*)") >= 1
    assert _count(db, "MATCH (:Chunk)-[:BELONGS_TO_SECTION]->(:Section) RETURN count(*)") >= 1


def test_cross_modal_edges(db) -> None:
    assert _count(db, "MATCH (:Chunk)-[:INTRODUCES]->(:Chunk {element_type:'table'}) RETURN count(*)") == 1
    assert _count(db, "MATCH (:Chunk)-[:REFERENCES]->(:Chunk {element_type:'figure'}) RETURN count(*)") == 1
    assert _count(db, "MATCH (:Chunk {element_type:'table'})-[:VISUALISED_BY]->(:Chunk {element_type:'figure'}) RETURN count(*)") == 1
    assert _count(db, "MATCH (:Chunk {element_type:'figure'})-[:CAPTIONED_BY]->(:Chunk {element_type:'caption'}) RETURN count(*)") == 1
    assert _count(db, "MATCH (:Chunk {element_type:'caption'})-[:DESCRIBES]->(:Chunk {element_type:'figure'}) RETURN count(*)") == 1


def test_entity_chunk_edges(db) -> None:
    assert _count(db, "MATCH (:Entity)-[:MENTIONED_IN]->(:Chunk) RETURN count(*)") >= 2
    assert _count(db, "MATCH (:Entity)-[:FIRST_MENTIONED_IN]->(:Chunk) RETURN count(*)") == 2
    assert _count(db, "MATCH (:Entity)-[:DEFINED_IN]->(:Chunk) RETURN count(*)") == 2


def test_entity_relations_are_typed(db) -> None:
    # "is_based_in" → LOCATED_IN between the two known entities.
    assert _count(
        db, "MATCH (:Organisation {name:'Acme Corporation'})-[:LOCATED_IN]->(:Location {name:'New York'}) RETURN count(*)"
    ) == 1
    # "works_at" with an unknown subject mints a Concept and a WORKS_AT edge.
    assert _count(db, "MATCH (:Entity)-[:WORKS_AT]->(:Entity) RETURN count(*)") == 1
    assert _count(db, "MATCH (c:Concept {name:'Jane Smith'}) RETURN count(c)") == 1
    # Unknown predicate falls back to RELATED_TO carrying the predicate.
    pred = db.query(
        "MATCH (:Entity)-[r:RELATED_TO]->(:Entity) WHERE r.predicate='partnered_with' RETURN r.predicate",
        read_only=True,
    )
    assert pred and pred[0][0] == "partnered_with"


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #
def test_publish_is_idempotent(db, published) -> None:
    publisher, _ = published
    before = _count(db, "MATCH (n) RETURN count(n)")
    edges_before = _count(db, "MATCH ()-[r]->() RETURN count(r)")
    pr, cr, er = _build_synthetic()
    publisher.publish_document(pr, cr, er, domain="finance")
    assert _count(db, "MATCH (n) RETURN count(n)") == before
    assert _count(db, "MATCH ()-[r]->() RETURN count(r)") == edges_before


# --------------------------------------------------------------------------- #
# Cross-document edges
# --------------------------------------------------------------------------- #
def test_publish_cross_document_edges(db, published) -> None:
    publisher, _ = published

    class _Res:
        cross_doc_edges = [
            {"type": "SAME_AS",
             "from_id": f"{_DOC_ID}::Organisation::acme corporation",
             "to_id": f"{_DOC_ID}::Location::new york",
             "props": {"confidence": 0.9}},
            {"type": "CORROBORATES",
             "from_id": f"{_DOC_ID}_chunk_0001", "to_id": f"{_DOC_ID}_chunk_0002",
             "props": {}},
        ]

    written = publisher.publish_cross_document_edges(_Res())
    assert written == 2
    assert _count(db, "MATCH (:Entity)-[r:SAME_AS]->(:Entity) RETURN count(r)") == 1
    assert _count(db, "MATCH (:Chunk)-[:CORROBORATES]->(:Chunk) RETURN count(*)") == 1
