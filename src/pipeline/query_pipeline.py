"""Stages 9-12 orchestration — query understanding through grounded answering.

:class:`QueryPipeline` classifies a query (visual / comparative / multi-hop) and
its scope (single document vs the whole graph), then runs hybrid retrieval →
cross-encoder re-ranking → answer generation, supplying the multi-hop
``retrieve_fn`` so sub-questions reuse the same scoped pipeline.

Query understanding is keyword/structure driven and domain-agnostic; document
scope is resolved by matching the query against the real Document names in the
graph. Components are lazily created and injectable for testing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.graph.falkordb_client import FalkorDBClient, get_client

logger = logging.getLogger(__name__)

_VISUAL_TERMS = {
    "show", "display", "chart", "graph", "figure", "visualise", "visualize",
    "image", "diagram", "plot", "picture", "infographic",
}
_COMPARATIVE_TERMS = (
    "compare", "comparison", "versus", " vs ", " vs.", "difference between",
    "across documents", "both documents", "compared to", "differ",
)
_MULTI_HOP_TERMS = (
    "who acquired", "what company", "which company", "which person", "who founded",
    "who owns", "who leads", "who is the ceo", "who works at", "who runs",
    "what acquired", "subsidiary of",
)


@dataclass
class QueryAnalysis:
    """Stage 9 output: query type and scope."""

    query: str
    query_type: str  # multi_hop | comparative | visual | standard
    visual: bool
    comparative: bool
    multi_hop: bool
    scope: str  # single_doc | cross_doc
    scope_doc_id: str | None = None
    scope_doc_name: str | None = None


@dataclass
class QueryResponse:
    """Full result of answering a query (for the UI)."""

    analysis: QueryAnalysis
    answer_result: object
    retrieval_result: object = None
    rerank_result: object = None


class QueryPipeline:
    """Understands a query and runs retrieve → rerank → generate."""

    def __init__(
        self, *, retriever=None, reranker=None, generator=None,
        client: FalkorDBClient | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._generator = generator
        self._db = client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def answer(self, query: str) -> QueryResponse:
        """Understand, retrieve, re-rank, and answer a query.

        Args:
            query: The natural-language question.

        Returns:
            A :class:`QueryResponse` with the analysis and all stage outputs.
        """
        analysis = self.understand(query)
        retriever, reranker, generator = (
            self._get_retriever(), self._get_reranker(), self._get_generator(),
        )

        retrieval = retriever.retrieve(query, scope_doc_id=analysis.scope_doc_id)
        rerank = reranker.rerank(query, retrieval.chunks)

        def retrieve_fn(sub_q: str):
            r = retriever.retrieve(sub_q, scope_doc_id=analysis.scope_doc_id)
            return reranker.rerank(sub_q, r.chunks)

        answer_result = generator.generate(
            query, rerank,
            query_type="multi_hop" if analysis.multi_hop else "standard",
            visual=analysis.visual, retrieve_fn=retrieve_fn,
        )
        logger.info(
            "Answered (%s, scope=%s): adversarial=%s",
            analysis.query_type, analysis.scope, answer_result.adversarial,
        )
        return QueryResponse(
            analysis=analysis, answer_result=answer_result,
            retrieval_result=retrieval, rerank_result=rerank,
        )

    # ------------------------------------------------------------------ #
    # Stage 9 — query understanding
    # ------------------------------------------------------------------ #
    def understand(self, query: str) -> QueryAnalysis:
        """Classify a query's type and resolve its document scope."""
        lower = f" {query.lower()} "
        visual = self._has_visual(lower)
        comparative = self._has_comparative(lower)
        multi_hop = self._has_multi_hop(lower)

        if multi_hop:
            query_type = "multi_hop"
        elif comparative:
            query_type = "comparative"
        elif visual:
            query_type = "visual"
        else:
            query_type = "standard"

        doc = self._match_document(query, self._documents())
        if doc is not None:
            scope, scope_doc_id, scope_doc_name = "single_doc", doc[0], doc[1]
        else:
            scope, scope_doc_id, scope_doc_name = "cross_doc", None, None

        return QueryAnalysis(
            query=query, query_type=query_type, visual=visual,
            comparative=comparative, multi_hop=multi_hop, scope=scope,
            scope_doc_id=scope_doc_id, scope_doc_name=scope_doc_name,
        )

    @staticmethod
    def _has_visual(padded_lower: str) -> bool:
        return any(f" {t} " in padded_lower for t in _VISUAL_TERMS)

    @staticmethod
    def _has_comparative(padded_lower: str) -> bool:
        return any(t in padded_lower for t in _COMPARATIVE_TERMS)

    @staticmethod
    def _has_multi_hop(padded_lower: str) -> bool:
        return any(t in padded_lower for t in _MULTI_HOP_TERMS)

    @staticmethod
    def _match_document(
        query: str, documents: list[tuple[str, str]]
    ) -> tuple[str, str] | None:
        """Match a query to a document by filename or its distinctive tokens.

        Returns ``(doc_id, doc_name)`` for a confident match, else ``None``
        (cross-document scope). Requires the exact filename or at least two
        stem tokens (and the majority of them) to avoid false positives.
        """
        q = query.lower()
        best: tuple[int, tuple[str, str]] | None = None
        for doc_id, name in documents:
            if not name:
                continue
            if name.lower() in q:
                return (doc_id, name)
            stem = re.sub(r"\.[a-z0-9]+$", "", name.lower())
            tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if len(t) > 2]
            if not tokens:
                continue
            present = [t for t in tokens if re.search(rf"\b{re.escape(t)}\b", q)]
            if len(present) >= 2 and len(present) >= (len(tokens) + 1) // 2:
                score = len(present)
                if best is None or score > best[0]:
                    best = (score, (doc_id, name))
        return best[1] if best else None

    def _documents(self) -> list[tuple[str, str]]:
        """Return ``(id, name)`` for every Document node in the graph."""
        try:
            rows = self._get_db().query(
                "MATCH (d:Document) RETURN d.id, d.name", read_only=True
            )
            return [(r[0], r[1]) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list documents for scope detection: %s", exc)
            return []

    # ------------------------------------------------------------------ #
    # Lazy components
    # ------------------------------------------------------------------ #
    def _get_db(self) -> FalkorDBClient:
        if self._db is None:
            self._db = get_client()
        return self._db

    def _get_retriever(self):
        if self._retriever is None:
            from src.retrieval.hybrid_retriever import HybridRetriever
            self._retriever = HybridRetriever(client=self._get_db())
        return self._retriever

    def _get_reranker(self):
        if self._reranker is None:
            from src.retrieval.reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    def _get_generator(self):
        if self._generator is None:
            from src.retrieval.answer_generator import AnswerGenerator
            self._generator = AnswerGenerator()
        return self._generator


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = " ".join(sys.argv[1:]) or "Who acquired Beta Labs and who leads that company?"
    response = QueryPipeline().answer(query)
    a = response.analysis
    print(f"\nQuery: {query}")
    print(f"Type: {a.query_type} | visual={a.visual} comparative={a.comparative} "
          f"multi_hop={a.multi_hop} | scope={a.scope} ({a.scope_doc_name})")
    print(f"\nAnswer:\n{response.answer_result.answer}")
    if response.answer_result.sub_questions:
        print(f"\nSub-questions: {response.answer_result.sub_questions}")
    print(f"\nCitations: {[(c['chunk_id'], c['page_number'], c['document_name']) for c in response.answer_result.citations]}")
