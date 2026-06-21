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

        try:
            raw_scores = self._call_nvidia_nim_rerank(query, chunks)
        except Exception as exc:
            logger.error("Reranking failed, falling back to original order: %s", exc)
            raw_scores = [0.0] * len(chunks)  # Fallback to neutral 0.5 sigmoid

        for chunk, raw in zip(chunks, raw_scores):
            chunk.cross_score = _sigmoid(float(raw))

        ranked = sorted(chunks, key=lambda c: c.cross_score, reverse=True)
        max_score = ranked[0].cross_score
        # If threshold is 0.0 (disabled), only gate on truly empty results.
        # This handles the case where ms-marco scores generic questions near 0.
        if self._threshold <= 0.0:
            adversarial = False
        else:
            adversarial = max_score < self._threshold

        logger.info(
            "Re-ranked %d chunks via %s; max_score=%.3f, adversarial=%s (threshold=%.2f)",
            len(chunks), config.cross_encoder_model, max_score, adversarial, self._threshold,
        )
        return RerankResult(
            chunks=ranked[:top_n], max_score=max_score,
            adversarial=adversarial, threshold=self._threshold,
        )

    def _call_nvidia_nim_rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        import os
        import requests
        import time
        
        api_key = os.getenv("NVIDIA_API_KEY", "").strip().strip('"').strip("'")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is required for cloud reranking.")

        payload = {
            "model": config.cross_encoder_model,
            "query": {"text": query},
            "passages": [{"text": c.text} for c in chunks]
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        
        for attempt in range(1, 4):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
                
                scores = [0.0] * len(chunks)
                for item in data.get("rankings", []):
                    idx = int(item["index"])
                    scores[idx] = float(item["logit"])
                return scores
            except requests.RequestException as exc:
                if attempt == 3:
                    raise
                logger.warning("NVIDIA Reranker API error, retrying (%s)", exc)
                time.sleep(2 ** attempt)
        return [0.0] * len(chunks)


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
