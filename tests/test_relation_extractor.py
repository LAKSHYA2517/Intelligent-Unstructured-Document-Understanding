"""Tests for Stage 5 Step 3 (LLM triples) — src/extraction/relation_extractor.py.

Deterministic tests for candidate selection, JSON parsing (including malformed
and null-field responses) and predicate normalisation via a monkeypatched LLM,
plus one real call extracting triples from a relationship-rich chunk.
"""

from __future__ import annotations

import pytest

from src.extraction.gliner_extractor import ExtractionResult
from src.extraction.ner import Entity
from src.extraction.relation_extractor import RelationExtractor
from src.ingestion.chunker import Chunk, ChunkResult


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc_chunk_{idx:04d}", text=text, table_json=None,
        element_type="paragraph", page_number=1, document_id="doc",
        document_name="doc.pdf", section_title="", position_in_doc=idx,
    )


def _entity(name: str, etype: str, *chunk_ids: str) -> Entity:
    e = Entity(name=name, entity_type=etype, document_id="doc", document_name="doc.pdf")
    for cid in chunk_ids:
        e.add_mention(cid, 1)
    return e


@pytest.fixture(scope="module")
def extractor() -> RelationExtractor:
    return RelationExtractor()


# --------------------------------------------------------------------------- #
# Candidate selection (pure)
# --------------------------------------------------------------------------- #
def test_only_multi_entity_chunks_selected(extractor: RelationExtractor) -> None:
    chunks = {c.chunk_id: c for c in (_chunk(0, "A and B."), _chunk(1, "Just C."))}
    entities = [
        _entity("A", "Organisation", "doc_chunk_0000"),
        _entity("B", "Person", "doc_chunk_0000"),
        _entity("C", "Person", "doc_chunk_0001"),
    ]
    cands = extractor._select_candidates(entities, chunks)
    assert len(cands) == 1
    assert cands[0].chunk.chunk_id == "doc_chunk_0000"
    assert set(cands[0].entity_names) == {"A", "B"}


def test_predicate_normalisation(extractor: RelationExtractor) -> None:
    assert extractor._normalise_predicate("Works At") == "works_at"
    assert extractor._normalise_predicate("  is the CEO of  ") == "is_the_ceo_of"
    assert extractor._normalise_predicate("acquired") == "acquired"


# --------------------------------------------------------------------------- #
# JSON parsing (pure)
# --------------------------------------------------------------------------- #
def test_parse_clean_json(extractor: RelationExtractor) -> None:
    raw = '{"triples": [{"subject": "Acme", "predicate": "located in", "object": "NY"}]}'
    triples = extractor._parse_triples(raw, _chunk(0, "x"))
    assert len(triples) == 1
    t = triples[0]
    assert (t.subject, t.predicate, t.object) == ("Acme", "located_in", "NY")
    assert t.chunk_id == "doc_chunk_0000" and t.extraction_method == "llm"


def test_parse_skips_null_and_empty_fields(extractor: RelationExtractor) -> None:
    raw = (
        '{"triples": ['
        '{"subject": "Acme", "predicate": "reported", "object": null},'
        '{"subject": "", "predicate": "x", "object": "y"},'
        '{"subject": "Jane", "predicate": "works at", "object": "Acme"}]}'
    )
    triples = extractor._parse_triples(raw, _chunk(0, "x"))
    assert len(triples) == 1
    assert (triples[0].subject, triples[0].object) == ("Jane", "Acme")


def test_parse_tolerates_prose_wrapped_json(extractor: RelationExtractor) -> None:
    raw = 'Sure! Here is the JSON:\n{"triples": [{"subject":"A","predicate":"x","object":"B"}]}'
    assert len(extractor._parse_triples(raw, _chunk(0, "x"))) == 1


def test_parse_returns_empty_on_garbage(extractor: RelationExtractor) -> None:
    assert extractor._parse_triples("not json at all", _chunk(0, "x")) == []


# --------------------------------------------------------------------------- #
# extract() wiring with a monkeypatched LLM (deterministic)
# --------------------------------------------------------------------------- #
def test_extract_populates_and_dedups(extractor: RelationExtractor, monkeypatch) -> None:
    cr = ChunkResult(chunks=[_chunk(0, "Jane Smith works at Acme Corporation.")])
    er = ExtractionResult(entities=[
        _entity("Jane Smith", "Person", "doc_chunk_0000"),
        _entity("Acme Corporation", "Organisation", "doc_chunk_0000"),
    ])
    # Return the same triple twice → dedup to one.
    monkeypatch.setattr(
        extractor, "_call_llm",
        lambda prompt: '{"triples": ['
        '{"subject":"Jane Smith","predicate":"works at","object":"Acme Corporation"},'
        '{"subject":"Jane Smith","predicate":"works at","object":"Acme Corporation"}]}',
    )
    out = extractor.extract(cr, er)
    assert out.triple_count == 1
    assert out.triples[0].predicate == "works_at"


def test_extract_no_candidates_yields_no_triples(extractor: RelationExtractor, monkeypatch) -> None:
    cr = ChunkResult(chunks=[_chunk(0, "Only one entity here.")])
    er = ExtractionResult(entities=[_entity("Acme", "Organisation", "doc_chunk_0000")])
    monkeypatch.setattr(extractor, "_call_llm", lambda p: pytest.fail("LLM must not be called"))
    out = extractor.extract(cr, er)
    assert out.triple_count == 0


# --------------------------------------------------------------------------- #
# Real LLM call
# --------------------------------------------------------------------------- #
def test_real_triple_extraction(extractor: RelationExtractor) -> None:
    cr = ChunkResult(chunks=[
        _chunk(0, "Jane Smith is the chief executive of Acme Corporation, "
                  "which is headquartered in New York."),
    ])
    er = ExtractionResult(entities=[
        _entity("Jane Smith", "Person", "doc_chunk_0000"),
        _entity("Acme Corporation", "Organisation", "doc_chunk_0000"),
        _entity("New York", "Location", "doc_chunk_0000"),
    ])
    out = extractor.extract(cr, er)
    assert out.triple_count >= 1
    for t in out.triples:
        assert t.subject and t.predicate and t.object
        assert t.subject.lower() not in {"none", "null"}
        assert t.chunk_id == "doc_chunk_0000" and t.extraction_method == "llm"
