"""Tests for the Streamlit progress formatting — app/streamlit_app.py.

The UI must display only runtime pipeline values, never hardcoded ones. These
tests feed synthetic stage payloads to ``format_stage`` and assert each rendered
line reflects the payload's fields.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.streamlit_app import format_stage


def test_parse_line_uses_parse_result_fields() -> None:
    payload = SimpleNamespace(page_count=7, table_count=3, figure_count=2)
    line = format_stage("parse", payload)
    assert "7 pages" in line and "3 tables" in line and "2 figures" in line


def test_image_line_uses_vision_calls() -> None:
    line = format_stage("image", SimpleNamespace(vision_calls=4, skipped=1))
    assert "4 vision model calls" in line and "1 skipped" in line


def test_chunk_and_domain_and_extract_lines() -> None:
    assert "12 chunks" in format_stage("chunk", SimpleNamespace(chunk_count=12))
    dom = format_stage("domain", SimpleNamespace(domain="finance", confidence=0.95))
    assert "finance" in dom and "0.95" in dom
    ext = format_stage("extract", SimpleNamespace(entity_count=20, triple_count=9))
    assert "20 entities" in ext and "9 triples" in ext


def test_embed_publish_resolution_crossdoc_lines() -> None:
    assert "5 chunks" in format_stage("embed", SimpleNamespace(embedded_count=5, dim=768))
    pub = format_stage("publish", SimpleNamespace(sections=4, chunks=7, figures=1,
                                                  entities=26, relationships=19))
    assert "4 sections" in pub and "26 entities" in pub and "19 relations" in pub
    assert "9 entities merged" in format_stage("resolution", SimpleNamespace(merged_count=9))
    assert "13 edges" in format_stage("cross_document", 13)


def test_unknown_stage_is_safe() -> None:
    assert "mystery" in format_stage("mystery", SimpleNamespace())
