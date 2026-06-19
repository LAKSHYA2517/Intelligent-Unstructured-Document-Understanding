"""Tests for Stage 5 Step 2 (GLiNER extraction) — src/extraction/gliner_extractor.py.

Pure tests for runtime label assembly, the label→type map, and the entity merge,
plus real GLiNER inference and a monkeypatched check that the *runtime-assembled*
label set is exactly what gets passed to the model.
"""

from __future__ import annotations

import pytest

from src.extraction.domain_detector import DomainResult
from src.extraction.gliner_extractor import (
    BASE_LABELS,
    DOMAIN_EXTENSION_LABELS,
    _LABEL_TO_TYPE,
    ExtractionResult,
    GLiNERExtractor,
    Triple,
    assemble_labels,
)
from src.extraction.ner import Entity
from src.ingestion.chunker import Chunk, ChunkResult


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc_chunk_{idx:04d}", text=text, table_json=None,
        element_type="paragraph", page_number=1, document_id="doc",
        document_name="doc.pdf", section_title="", position_in_doc=idx,
    )


# --------------------------------------------------------------------------- #
# Pure: labels, type map, counts, merge
# --------------------------------------------------------------------------- #
def test_assemble_labels_per_domain() -> None:
    assert assemble_labels("general") == BASE_LABELS
    assert len(assemble_labels("finance")) == len(BASE_LABELS) + 5
    assert len(assemble_labels("legal")) == len(BASE_LABELS) + 5
    assert len(assemble_labels("technical")) == len(BASE_LABELS) + 4
    assert assemble_labels("unknown-domain") == BASE_LABELS  # safe fallback


def test_every_label_has_a_type() -> None:
    all_labels = set(BASE_LABELS)
    for labels in DOMAIN_EXTENSION_LABELS.values():
        all_labels.update(labels)
    missing = [lbl for lbl in all_labels if lbl not in _LABEL_TO_TYPE]
    assert not missing, f"labels without a type mapping: {missing}"


def test_extraction_result_counts_and_recount() -> None:
    res = ExtractionResult(entities=[
        Entity("Acme", "Organisation", "d", "d.pdf"),
    ])
    assert res.entity_count == 1 and res.triple_count == 0
    res.triples.append(Triple("Acme", "located_in", "NY", "c1", "d"))
    res.recount()
    assert res.triple_count == 1


def test_merge_span_creates_then_enriches() -> None:
    merged: dict = {}
    chunk = _chunk(0, "Acme Corporation grew.")
    # spaCy entity already present.
    spacy_entity = Entity("Acme Corporation", "Organisation", "doc", "doc.pdf",
                          extraction_method="spacy_ner", confidence=0.8)
    merged[spacy_entity.key] = spacy_entity
    # GLiNER agrees on the same entity → method records agreement, conf rises.
    GLiNERExtractor._merge_span(
        merged, {"text": "Acme Corporation", "label": BASE_LABELS[1], "score": 0.95}, chunk
    )
    assert len(merged) == 1
    enriched = merged[spacy_entity.key]
    assert enriched.extraction_method == "spacy_ner+gliner"
    assert enriched.confidence == pytest.approx(0.95)
    # A brand-new span creates a new entity tagged gliner.
    GLiNERExtractor._merge_span(
        merged, {"text": "Revenue", "label": BASE_LABELS[6], "score": 0.7}, chunk
    )
    assert len(merged) == 2
    new = merged[("revenue", "Quantity")]
    assert new.extraction_method == "gliner"


# --------------------------------------------------------------------------- #
# Real model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def extractor() -> GLiNERExtractor:
    return GLiNERExtractor()


def test_runtime_labels_passed_to_model(extractor: GLiNERExtractor, monkeypatch) -> None:
    """The label set handed to GLiNER is BASE + domain extension, built at runtime."""
    captured: list[list[str]] = []

    def _spy(text, labels, threshold):
        captured.append(labels)
        return []

    monkeypatch.setattr(extractor._model, "predict_entities", _spy)
    cr = ChunkResult(chunks=[_chunk(0, "Some text.")])
    extractor.extract(cr, DomainResult("finance", 0.9))
    assert captured and len(captured[0]) == len(BASE_LABELS) + 5
    assert "Stock ticker or trading symbol" in captured[0]

    captured.clear()
    extractor.extract(cr, DomainResult("general", 0.9))
    assert captured[0] == BASE_LABELS


def test_real_entity_extraction(extractor: GLiNERExtractor) -> None:
    cr = ChunkResult(chunks=[
        _chunk(0, "Acme Corporation reported revenue of 5 million dollars in Q3."),
    ])
    res = extractor.extract(cr, DomainResult("finance", 0.9))
    assert res.entity_count > 0 and res.triple_count == 0
    assert any("Acme" in e.name and e.entity_type == "Organisation" for e in res.entities)


def test_prior_entities_are_merged(extractor: GLiNERExtractor) -> None:
    prior = [Entity("Acme Corporation", "Organisation", "doc", "doc.pdf",
                    extraction_method="spacy_ner")]
    cr = ChunkResult(chunks=[_chunk(0, "Acme Corporation expanded operations.")])
    res = extractor.extract(cr, DomainResult("general", 0.9), prior_entities=prior)
    acme = [e for e in res.entities if "Acme" in e.name and e.entity_type == "Organisation"]
    assert len(acme) == 1  # not duplicated
