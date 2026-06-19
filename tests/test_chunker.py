"""Tests for Stage 3 (element-aware chunking) — src/ingestion/chunker.py.

Combines a real end-to-end run over the sample PDF with focused unit tests for
the token-cap splitting and the Stage 2 figure-override / skip integration,
using lightweight synthetic ParseResults so no Docling run is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.ingestion.chunker import Chunk, Chunker, _estimate_tokens
from src.ingestion.parser import DocumentParser, ParsedElement, ParseResult


# --------------------------------------------------------------------------- #
# End-to-end over the real sample PDF
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def chunk_result(sample_pdf):
    """Parse then chunk the sample PDF once for the module."""
    pr = DocumentParser().parse(sample_pdf)
    return Chunker().chunk(pr)


def test_chunk_ids_and_positions_are_sequential(chunk_result) -> None:
    """chunk_id encodes position and positions are a contiguous 0..n-1 range."""
    positions = [c.position_in_doc for c in chunk_result.chunks]
    assert positions == list(range(len(chunk_result.chunks)))
    assert chunk_result.chunk_count == len(chunk_result.chunks)
    for c in chunk_result.chunks:
        assert c.chunk_id.endswith(f"_chunk_{c.position_in_doc:04d}")


def test_table_chunk_keeps_structured_json(chunk_result) -> None:
    """A table chunk carries both rendered text and structured table_json."""
    tables = [c for c in chunk_result.chunks if c.element_type == "table"]
    assert tables, "expected at least one table chunk"
    t = tables[0]
    assert t.table_json is not None and t.table_json["headers"] == ["Region", "Q1", "Q2", "Q3"]
    assert "Region" in t.text  # rendered markdown present too


def test_heading_and_body_grouped(chunk_result) -> None:
    """The executive-summary heading and its body share one paragraph chunk."""
    grouped = [
        c
        for c in chunk_result.chunks
        if "Executive Summary" in c.text and "Acme Corporation" in c.text
    ]
    assert len(grouped) == 1
    assert grouped[0].element_type == "paragraph"
    assert grouped[0].section_title == "1. Executive Summary"


# --------------------------------------------------------------------------- #
# Token-cap splitting (unit)
# --------------------------------------------------------------------------- #
def test_token_cap_splits_only_on_sentence_boundaries() -> None:
    """A long multi-sentence paragraph splits into capped, whole-sentence chunks."""
    sentence = "This is sentence number {} about revenue and growth metrics."
    body = " ".join(sentence.format(i) for i in range(120))
    doc = _synthetic_result(
        [_el("e0", "heading", "Overview", page=1),
         _el("e1", "paragraph", body, page=1)]
    )
    chunker = Chunker(max_tokens=64)
    chunks = chunker.chunk(doc).chunks
    paras = [c for c in chunks if c.element_type == "paragraph"]
    assert len(paras) > 1  # actually split
    for c in paras:
        assert _estimate_tokens(c.text) <= 64 or c.text.count(".") <= 1
        assert not c.text.strip().endswith(("sentence", "number"))  # no mid-sentence cut


def test_long_single_sentence_not_broken() -> None:
    """One oversized sentence is emitted whole rather than split mid-sentence."""
    long_sentence = "alpha " * 200  # no sentence terminators
    doc = _synthetic_result([_el("e0", "paragraph", long_sentence.strip(), page=1)])
    chunks = Chunker(max_tokens=32).chunk(doc).chunks
    assert len(chunks) == 1


# --------------------------------------------------------------------------- #
# Stage 2 figure integration (unit)
# --------------------------------------------------------------------------- #
@dataclass
class _FakeFigure:
    element_id: str
    text: str
    image_path: str | None


@dataclass
class _FakeImageResult:
    processed_figures: list[_FakeFigure]
    skipped_element_ids: list[str]


def test_figure_override_and_skip() -> None:
    """Figure text comes from the Stage 2 description; skipped figures are dropped."""
    doc = _synthetic_result(
        [
            _el("fig_keep", "figure", "vague caption", page=2, image_path="/x/a.png"),
            _el("fig_drop", "figure", "", page=2, image_path="/x/b.png"),
        ]
    )
    image_result = _FakeImageResult(
        processed_figures=[_FakeFigure("fig_keep", "A bar chart of revenue by region.", "/x/a.png")],
        skipped_element_ids=["fig_drop"],
    )
    chunks = Chunker().chunk(doc, image_result).chunks
    figures = [c for c in chunks if c.element_type == "figure"]
    assert len(figures) == 1  # the skipped one is gone
    assert figures[0].text == "A bar chart of revenue by region."
    assert figures[0].image_path == "/x/a.png"


def test_empty_document_yields_no_chunks() -> None:
    """An empty ParseResult produces zero chunks without error."""
    result = Chunker().chunk(_synthetic_result([]))
    assert result.chunk_count == 0 and result.chunks == []


# --------------------------------------------------------------------------- #
# Synthetic builders
# --------------------------------------------------------------------------- #
def _el(eid: str, etype: str, text: str, *, page: int, image_path: str | None = None) -> ParsedElement:
    return ParsedElement(
        element_id=eid,
        element_type=etype,
        text=text,
        page_number=page,
        reading_order=0,
        image_path=image_path,
        self_ref=f"#/{eid}",
    )


def _synthetic_result(elements: list[ParsedElement]) -> ParseResult:
    for i, el in enumerate(elements):
        el.reading_order = i
    return ParseResult(
        doc_id="testdoc_0001",
        doc_name="synthetic.pdf",
        elements=elements,
        page_count=1,
        table_count=0,
        figure_count=sum(1 for e in elements if e.element_type == "figure"),
        reading_order=[e.element_id for e in elements],
    )
