"""Stage 11 — cross-encoder re-ranking + adversarial gate.

Re-scores the fused top-15 chunks from Stage 10 against the query with a
cross-encoder (``config.CROSS_ENCODER_MODEL`` via sentence-transformers), keeps
the top-5 for answer generation, and reports whether the **adversarial gate**
should fire — i.e. whether the best relevance score is below
``config.ADVERSARIAL_SCORE_THRESHOLD``, meaning no retrieved chunk is relevant
enough to answer from.

Cross-encoder models like ms-marco emit raw logits, so a sigmoid maps each score
to a 0-1 relevance probability that is directly comparable to the configured
threshold (the sigmoid is monotonic, so the ranking is unchanged).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from src.config import config
from src.retrieval.hybrid_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_TOP_N = 5


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid mapping a logit to 0-1."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class RerankResult:
    """Stage 11 output.

    Attributes:
        chunks: Top-N chunks by cross-encoder relevance, ``cross_score`` set.
        max_score: Best relevance probability over all candidates (0-1).
        adversarial: True if ``max_score`` < the adversarial threshold.
        threshold: The adversarial threshold used.
    """

    chunks: list[RetrievedChunk]
    max_score: float = 0.0
    adversarial: bool = False
    threshold: float = 0.0


class Reranker:
    """Cross-encoder re-ranker with an adversarial relevance gate."""

    def __init__(self, model=None) -> None:
        self._model = model
        self._threshold = config.adversarial_score_threshold

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int = _TOP_N
    ) -> RerankResult:
        """Re-rank chunks against the query and apply the adversarial gate.

        Args:
            query: The user query.
            chunks: Candidate chunks from Stage 10 (typically up to 15).
            top_n: Number of top chunks to keep for the answer.

        Returns:
            A :class:`RerankResult`. Empty input trips the adversarial gate.
        """
        if not chunks:
            return RerankResult(chunks=[], max_score=0.0, adversarial=True,
                                threshold=self._threshold)

        pairs = [(query, c.text) for c in chunks]
        raw_scores = self._get_model().predict(pairs)
        for chunk, raw in zip(chunks, raw_scores):
            chunk.cross_score = _sigmoid(float(raw))

        ranked = sorted(chunks, key=lambda c: c.cross_score, reverse=True)
        max_score = ranked[0].cross_score
        adversarial = max_score < self._threshold

        logger.info(
            "Re-ranked %d chunks; max_score=%.3f, adversarial=%s (threshold=%.2f)",
            len(chunks), max_score, adversarial, self._threshold,
        )
        return RerankResult(
            chunks=ranked[:top_n], max_score=max_score,
            adversarial=adversarial, threshold=self._threshold,
        )

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(config.cross_encoder_model)
            logger.info("Loaded cross-encoder from config: %s", config.cross_encoder_model)
        return self._model


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = " ".join(sys.argv[1:]) or "What is Acme Corporation's revenue and who is its CEO?"
    from src.retrieval.hybrid_retriever import HybridRetriever

    retrieved = HybridRetriever().retrieve(query)
    result = Reranker().rerank(query, retrieved.chunks)
    print(f"\nQuery: {query}")
    print(f"max_score={result.max_score:.3f} adversarial={result.adversarial}")
    for c in result.chunks:
        print(f"  [{c.cross_score:.3f}] {c.element_type:9s} p{c.page_number} "
              f"{c.document_name} :: {c.text[:60]!r}")
