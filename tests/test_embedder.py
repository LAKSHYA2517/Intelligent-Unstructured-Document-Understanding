"""Tests for Stage 6 (embedding) — src/retrieval/embedder.py.

Real Ollama embedding of chunks and queries, verifying dimensionality, in-place
attachment to chunks, query/ingestion model consistency, empty-text handling, and
the dimension-mismatch guard.
"""

from __future__ import annotations

import math

import pytest

from src.config import config
from src.ingestion.chunker import Chunk, ChunkResult
from src.retrieval.embedder import Embedder


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc_chunk_{idx:04d}", text=text, table_json=None,
        element_type="paragraph", page_number=1, document_id="doc",
        document_name="doc.pdf", section_title="", position_in_doc=idx,
    )


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embed_attaches_vectors_in_place(embedder: Embedder) -> None:
    cr = ChunkResult(chunks=[_chunk(0, "Revenue grew this quarter."), _chunk(1, "Costs fell.")])
    res = embedder.embed(cr)
    assert res.embedded_count == 2
    assert res.dim == config.embedding_dim == 768
    assert res.model == config.ollama_embed_model
    assert all(c.embedding is not None and len(c.embedding) == 768 for c in cr.chunks)


def test_embed_handles_empty_text(embedder: Embedder) -> None:
    cr = ChunkResult(chunks=[_chunk(0, "   "), _chunk(1, "real content")])
    res = embedder.embed(cr)
    assert res.embedded_count == 2
    assert all(len(c.embedding) == 768 for c in cr.chunks)


def test_query_embedding_matches_ingestion_space(embedder: Embedder) -> None:
    """A query embeds with the same model/dim and is semantically comparable."""
    cr = ChunkResult(chunks=[
        _chunk(0, "Quarterly revenue increased across all regions."),
        _chunk(1, "The cat sat on the warm windowsill in the sun."),
    ])
    embedder.embed(cr)
    q = embedder.embed_query("How did revenue change this quarter?")
    assert len(q) == 768

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    sim_revenue = _cos(q, cr.chunks[0].embedding)
    sim_cat = _cos(q, cr.chunks[1].embedding)
    assert sim_revenue > sim_cat  # query is closer to the on-topic chunk


def test_dimension_guard_rejects_wrong_size(embedder: Embedder) -> None:
    with pytest.raises(ValueError):
        embedder._validate_dim([0.0, 1.0, 2.0])  # not 768
