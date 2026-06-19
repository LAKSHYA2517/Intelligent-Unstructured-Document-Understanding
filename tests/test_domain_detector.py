"""Tests for Stage 4 (domain detection) — src/extraction/domain_detector.py.

A real classification of the sample financial PDF, plus network-free unit tests
for response parsing, the 500-word sampling window, and the empty-document
short-circuit (which must make no LLM call).
"""

from __future__ import annotations

import pytest

from src.extraction.domain_detector import VALID_DOMAINS, DomainDetector, DomainResult
from src.ingestion.chunker import Chunk, Chunker, ChunkResult
from src.ingestion.parser import DocumentParser


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"d_chunk_{idx:04d}", text=text, table_json=None,
        element_type="paragraph", page_number=1, document_id="d",
        document_name="d.pdf", section_title="", position_in_doc=idx,
    )


@pytest.fixture(scope="module")
def detector() -> DomainDetector:
    return DomainDetector()


# --------------------------------------------------------------------------- #
# Response parsing (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("finance", "finance"),
        ("MEDICAL", "medical"),
        ("  Technical  ", "technical"),
        ("This is clearly a legal document.", "legal"),
        ("general", "general"),
        ("fintech buzzwords only", "general"),  # no valid label present → default
        ("???", "general"),
    ],
)
def test_parse_domain(detector: DomainDetector, raw: str, expected: str) -> None:
    domain, confidence = detector._parse_domain(raw)
    assert domain == expected
    assert 0.0 <= confidence <= 1.0
    assert domain in VALID_DOMAINS


def test_first_words_truncates_to_window(detector: DomainDetector) -> None:
    big = ChunkResult(chunks=[_chunk(" ".join(f"w{i}" for i in range(400)), j) for j in range(3)])
    sampled = detector._first_words(big, 500)
    assert len(sampled.split()) == 500


# --------------------------------------------------------------------------- #
# Empty document short-circuit (no LLM call)
# --------------------------------------------------------------------------- #
def test_empty_document_makes_no_llm_call(detector: DomainDetector, monkeypatch) -> None:
    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("LLM should not be called for empty input")

    monkeypatch.setattr(detector._client, "generate", _boom)
    result = detector.detect(ChunkResult(chunks=[]))
    assert result.domain == "general" and result.confidence == 0.0


# --------------------------------------------------------------------------- #
# Real classification
# --------------------------------------------------------------------------- #
def test_real_detection_of_financial_doc(detector: DomainDetector, sample_pdf) -> None:
    pr = DocumentParser().parse(sample_pdf)
    cr = Chunker().chunk(pr)
    result = detector.detect(cr)
    assert isinstance(result, DomainResult)
    assert result.domain in VALID_DOMAINS
    assert result.domain == "finance"  # quarterly revenue report
    assert result.confidence >= 0.7
