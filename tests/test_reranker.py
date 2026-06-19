"""Tests for Stage 11 (cross-encoder re-ranking) — src/retrieval/reranker.py.

A stub cross-encoder drives deterministic ranking/gate tests; pure tests cover
the sigmoid; and a real cross-encoder confirms relevant queries pass while
off-topic queries trip the adversarial gate.
"""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_retriever import RetrievedChunk
from src.retrieval.reranker import Reranker, _sigmoid


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, text=text, element_type="paragraph", page_number=1,
        document_id="d", document_name="d.pdf", section_title="", table_json=None,
        image_path=None, confidence=1.0,
    )


class _StubModel:
    """Returns preset logits keyed by chunk text order."""

    def __init__(self, logits: list[float]) -> None:
        self._logits = logits

    def predict(self, pairs):
        return self._logits[: len(pairs)]


# --------------------------------------------------------------------------- #
# Pure
# --------------------------------------------------------------------------- #
def test_sigmoid() -> None:
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(20.0) > 0.99
    assert _sigmoid(-20.0) < 0.01


# --------------------------------------------------------------------------- #
# Ranking + gate (stubbed)
# --------------------------------------------------------------------------- #
def test_ranks_by_score_and_takes_top_n() -> None:
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(6)]
    # Logits ascending → after sigmoid, c5 highest, c0 lowest.
    reranker = Reranker(model=_StubModel([-3, -1, 0, 1, 2, 5]))
    result = reranker.rerank("q", chunks, top_n=3)
    assert [c.chunk_id for c in result.chunks] == ["c5", "c4", "c3"]
    assert result.chunks[0].cross_score > result.chunks[1].cross_score
    assert result.max_score == pytest.approx(_sigmoid(5))
    assert result.adversarial is False


def test_adversarial_gate_trips_on_low_scores() -> None:
    chunks = [_chunk(f"c{i}", f"text {i}") for i in range(3)]
    # All strongly negative logits → sigmoid well below 0.3.
    reranker = Reranker(model=_StubModel([-5, -6, -4]))
    result = reranker.rerank("q", chunks)
    assert result.adversarial is True
    assert result.max_score < result.threshold


def test_empty_input_trips_gate() -> None:
    result = Reranker(model=_StubModel([])).rerank("q", [])
    assert result.chunks == [] and result.adversarial is True


# --------------------------------------------------------------------------- #
# Real cross-encoder
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker()


def test_real_relevant_query_passes_gate(reranker: Reranker) -> None:
    chunks = [
        _chunk("c0", "Acme Corporation reported revenue of 540 million dollars in 2024."),
        _chunk("c1", "Jane Smith is the chief executive of Acme Corporation."),
        _chunk("c2", "The weather was sunny over the coastal hills all afternoon."),
    ]
    result = reranker.rerank("What was Acme's revenue?", chunks, top_n=2)
    assert not result.adversarial
    assert result.chunks[0].chunk_id == "c0"  # the revenue chunk ranks first
    assert len(result.chunks) == 2


def test_real_offtopic_query_trips_gate(reranker: Reranker) -> None:
    chunks = [
        _chunk("c0", "Acme Corporation reported revenue of 540 million dollars in 2024."),
        _chunk("c1", "Jane Smith is the chief executive of Acme Corporation."),
    ]
    result = reranker.rerank("What is the recipe for sourdough bread?", chunks)
    assert result.adversarial is True  # nothing relevant → gate fires
