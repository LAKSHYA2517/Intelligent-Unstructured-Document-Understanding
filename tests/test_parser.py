"""Tests for Stage 1 (Docling parsing) — src/ingestion/parser.py.

Exercises the real Docling pipeline against a generated multi-page PDF
(see tests/conftest.py) and asserts that every element type, real document
statistic, figure-crop persistence, caption linkage, and multi-page table
merge behave as specified.
"""

from __future__ import annotations

import os

import pytest

from src.ingestion.parser import DocumentParser, ParseResult, ParserError


@pytest.fixture(scope="module")
def parse_result(sample_pdf) -> ParseResult:
    """Parse the sample PDF once and reuse across the module's tests."""
    return DocumentParser().parse(sample_pdf)


def test_real_document_stats(parse_result: ParseResult) -> None:
    """Counts come from Docling, not hardcoded: 3 pages, 2 tables, 1 figure."""
    assert parse_result.page_count == 3
    assert parse_result.table_count == 2  # two detected tables (pre-merge)
    assert parse_result.figure_count == 1
    assert parse_result.doc_name == "quarterly_report.pdf"
    assert parse_result.doc_id.startswith("quarterly_report_")


def test_reading_order_is_contiguous(parse_result: ParseResult) -> None:
    """Reading order is contiguous and mirrors the elements list."""
    orders = [e.reading_order for e in parse_result.elements]
    assert orders == list(range(len(parse_result.elements)))
    assert parse_result.reading_order == [e.element_id for e in parse_result.elements]


def test_all_expected_element_types_present(parse_result: ParseResult) -> None:
    """Headings, paragraphs, a figure, a caption, and a (merged) table appear."""
    types = {e.element_type for e in parse_result.elements}
    assert {"heading", "paragraph", "figure", "caption", "table"} <= types


def test_figure_crop_saved_and_caption_linked(parse_result: ParseResult) -> None:
    """The figure image crop is persisted and its caption is linked back."""
    figures = [e for e in parse_result.elements if e.element_type == "figure"]
    assert len(figures) == 1
    fig = figures[0]
    assert fig.image_path and os.path.exists(fig.image_path)

    captions = [e for e in parse_result.elements if e.element_type == "caption"]
    assert any(c.parent_ref == fig.self_ref for c in captions)


def test_multipage_table_merged(parse_result: ParseResult) -> None:
    """Two same-header tables across pages collapse into one with all rows."""
    tables = [e for e in parse_result.elements if e.element_type == "table"]
    assert len(tables) == 1  # merged from the two detected tables
    table = tables[0]
    assert table.table_json is not None
    assert table.table_json["headers"] == ["Region", "Q1", "Q2", "Q3"]
    region_col = [row[0] for row in table.table_json["rows"]]
    assert {"North", "South", "East", "West"} <= set(region_col)


def test_table_structure_preserved(parse_result: ParseResult) -> None:
    """Tables keep row/column structure and are not flattened to plain text."""
    table = next(e for e in parse_result.elements if e.element_type == "table")
    assert table.table_json["num_cols"] == 4
    assert all(len(row) == 4 for row in table.table_json["rows"])


def test_rejects_missing_and_unsupported_files(tmp_path) -> None:
    """Missing files and unsupported extensions raise ParserError."""
    parser = DocumentParser()
    with pytest.raises(ParserError):
        parser.parse(tmp_path / "nope.pdf")
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(ParserError):
        parser.parse(bad)
