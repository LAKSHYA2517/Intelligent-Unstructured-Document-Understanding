"""Tests for Stage 5 Step 1 (NER + co-reference) — src/extraction/ner.py.

Real spaCy-transformer + fastcoref extraction (the model loads once per module),
plus pure unit tests for the sliding-window construction, offset→chunk mapping,
and the Entity merge/dedup helpers.
"""

from __future__ import annotations

import pytest

from src.extraction.ner import Entity, NERExtractor
from src.ingestion.chunker import Chunk, ChunkResult


def _chunk(idx: int, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"doc_chunk_{idx:04d}", text=text, table_json=None,
        element_type="paragraph", page_number=page, document_id="doc",
        document_name="doc.pdf", section_title="", position_in_doc=idx,
    )


@pytest.fixture(scope="module")
def extractor() -> NERExtractor:
    """Load the heavy spaCy-trf + fastcoref pipeline once for the module."""
    return NERExtractor()


# --------------------------------------------------------------------------- #
# Pure helpers (no model)
# --------------------------------------------------------------------------- #
def test_sliding_windows_size_and_stride() -> None:
    chunks = [_chunk(i, f"text {i}") for i in range(6)]
    windows = NERExtractor._windows(chunks)
    assert len(windows) == 4  # 6 - 3 + 1
    assert all(len(w) == 3 for w in windows)
    assert [w[0].position_in_doc for w in windows] == [0, 1, 2, 3]  # slide by 1
    # Short documents collapse to a single window.
    assert len(NERExtractor._windows(chunks[:2])) == 1


def test_offset_to_chunk_mapping() -> None:
    window = [_chunk(0, "Acme grew."), _chunk(1, "Profits rose.")]
    text, offsets = NERExtractor._concat_with_offsets(window)
    assert "Acme grew." in text and "Profits rose." in text
    assert NERExtractor._chunk_for_offset(0, offsets).position_in_doc == 0
    second_start = text.index("Profits")
    assert NERExtractor._chunk_for_offset(second_start, offsets).position_in_doc == 1


def test_entity_alias_and_mention_dedup() -> None:
    e = Entity(name="Acme Corporation", entity_type="Organisation",
               document_id="d", document_name="d.pdf")
    e.add_alias("Acme")
    e.add_alias("acme")  # case-insensitive duplicate
    e.add_alias("Acme Corporation")  # equals canonical → ignored
    assert e.aliases == ["Acme"]
    e.add_mention("c1", 1)
    e.add_mention("c1", 1)  # dedup
    e.add_mention("c2", 2)
    assert [m.chunk_id for m in e.mentions] == ["c1", "c2"]
    assert e.key == ("acme corporation", "Organisation")


# --------------------------------------------------------------------------- #
# Real extraction
# --------------------------------------------------------------------------- #
def test_extracts_named_entities(extractor: NERExtractor) -> None:
    cr = ChunkResult(chunks=[
        _chunk(0, "Acme Corporation reported strong revenue growth this quarter."),
        _chunk(1, "Jane Smith, the chief executive, presented the results."),
    ])
    res = extractor.extract(cr)
    assert res.entity_count >= 2
    by_type = {(e.entity_type, e.name) for e in res.entities}
    assert any(t == "Organisation" and "Acme" in n for t, n in by_type)
    assert any(t == "Person" and "Jane" in n for t, n in by_type)
    for e in res.entities:
        assert e.first_chunk_id is not None and e.mentions


def test_coreference_links_pronouns_to_entity(extractor: NERExtractor) -> None:
    """Pronoun/alias mentions across chunks attach to the canonical entity."""
    cr = ChunkResult(chunks=[
        _chunk(0, "Acme Corporation announced record profits in 2024."),
        _chunk(1, "The company said it would expand into new markets."),
        _chunk(2, "Acme also opened three regional offices."),
    ])
    res = extractor.extract(cr)
    orgs = [e for e in res.entities if e.entity_type == "Organisation" and "Acme" in e.name]
    assert orgs, "expected an Acme organisation entity"
    acme = max(orgs, key=lambda e: len(e.mentions))
    # Co-reference should attribute mentions from more than one chunk.
    assert len({m.chunk_id for m in acme.mentions}) >= 2


def test_empty_document_yields_no_entities(extractor: NERExtractor) -> None:
    assert extractor.extract(ChunkResult(chunks=[])).entity_count == 0
