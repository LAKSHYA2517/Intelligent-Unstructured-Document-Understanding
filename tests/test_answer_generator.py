"""Tests for Stage 12 (answer generation) — src/retrieval/answer_generator.py.

A stub LLM gives deterministic coverage of the element-aware context, the
adversarial gate (no LLM call), visual IMAGE_PATH handling, and multi-hop
decompose→synthesise; one real LLM call confirms grounded answering.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval.answer_generator import ADVERSARIAL_MESSAGE, AnswerGenerator
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.retrieval.reranker import RerankResult


def _chunk(cid, text, etype="paragraph", image_path=None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=text, element_type=etype, page_number=2,
        document_id="d", document_name="report.pdf", section_title="Summary",
        table_json=None, image_path=image_path, confidence=0.9,
    )


def _rr(chunks, *, adversarial=False, max_score=0.9) -> RerankResult:
    return RerankResult(chunks=chunks, max_score=max_score, adversarial=adversarial,
                        threshold=0.3)


class _StubLLM:
    """Branches on json mode / prompt content to mimic the three call types."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, model, prompt, **kwargs):
        self.calls += 1
        if kwargs.get("format") == "json":
            return SimpleNamespace(response='{"sub_questions": ["What is revenue?", "Who is CEO?"]}')
        if "Combine the sub-answers" in prompt:
            return SimpleNamespace(response="FINAL synthesised answer.")
        return SimpleNamespace(response="A grounded answer. IMAGE_PATH: /figs/chart.png")


class _ExplodingLLM:
    def generate(self, *a, **k):  # pragma: no cover
        raise AssertionError("LLM must not be called")


# --------------------------------------------------------------------------- #
# Context construction
# --------------------------------------------------------------------------- #
def test_build_context_uses_runtime_metadata() -> None:
    ctx = AnswerGenerator.build_context([_chunk("c1", "Revenue rose.", "table")])
    assert "[TABLE" in ctx
    assert "chunk_id: c1" in ctx
    assert "Summary" in ctx and "page 2" in ctx and "doc: report.pdf" in ctx
    assert "Revenue rose." in ctx


# --------------------------------------------------------------------------- #
# Adversarial gate
# --------------------------------------------------------------------------- #
def test_adversarial_gate_returns_canned_without_llm() -> None:
    gen = AnswerGenerator(llm_client=_ExplodingLLM())
    result = gen.generate("q", _rr([_chunk("c1", "x")], adversarial=True, max_score=0.1))
    assert result.adversarial is True
    assert result.answer == ADVERSARIAL_MESSAGE


def test_empty_chunks_trip_gate() -> None:
    gen = AnswerGenerator(llm_client=_ExplodingLLM())
    result = gen.generate("q", _rr([], adversarial=True, max_score=0.0))
    assert result.adversarial and result.answer == ADVERSARIAL_MESSAGE


# --------------------------------------------------------------------------- #
# Standard generation + citations + visual
# --------------------------------------------------------------------------- #
def test_standard_generation_sets_citations() -> None:
    gen = AnswerGenerator(llm_client=_StubLLM())
    chunks = [_chunk("c1", "Revenue is 540M."), _chunk("c2", "CEO is Jane.")]
    result = gen.generate("q", _rr(chunks))
    assert not result.adversarial
    assert {c["chunk_id"] for c in result.citations} == {"c1", "c2"}
    assert result.citations[0]["document_name"] == "report.pdf"


def test_visual_query_surfaces_image_paths() -> None:
    gen = AnswerGenerator(llm_client=_StubLLM())
    chunks = [_chunk("f1", "A bar chart.", "figure", image_path="/figs/chart.png")]
    result = gen.generate("show me the revenue chart", _rr(chunks), visual=True)
    assert "/figs/chart.png" in result.image_paths


# --------------------------------------------------------------------------- #
# Multi-hop
# --------------------------------------------------------------------------- #
def test_multi_hop_decomposes_and_synthesises() -> None:
    gen = AnswerGenerator(llm_client=_StubLLM())
    calls: list[str] = []

    def retrieve_fn(sub_q: str) -> RerankResult:
        calls.append(sub_q)
        return _rr([_chunk("c1", f"context for {sub_q}")])

    result = gen.generate("complex q", query_type="multi_hop", retrieve_fn=retrieve_fn)
    assert result.sub_questions == ["What is revenue?", "Who is CEO?"]
    assert calls == result.sub_questions  # retrieval ran per sub-question
    assert len(result.sub_answers) == 2
    assert result.answer == "FINAL synthesised answer."
    assert not result.adversarial


def test_multi_hop_requires_retrieve_fn_else_standard() -> None:
    gen = AnswerGenerator(llm_client=_StubLLM())
    # No retrieve_fn → falls back to standard generation on the provided result.
    result = gen.generate("q", _rr([_chunk("c1", "x")]), query_type="multi_hop")
    assert result.sub_questions is None


# --------------------------------------------------------------------------- #
# Real LLM
# --------------------------------------------------------------------------- #
def test_real_grounded_answer() -> None:
    chunks = [
        _chunk("acme_chunk_0001", "Acme Corporation reported revenue of 540 million dollars in 2024."),
        _chunk("acme_chunk_0002", "Jane Smith is the chief executive of Acme Corporation."),
    ]
    result = AnswerGenerator().generate("What was Acme's revenue in 2024?", _rr(chunks))
    assert not result.adversarial
    assert "540" in result.answer
    assert len(result.citations) == 2
