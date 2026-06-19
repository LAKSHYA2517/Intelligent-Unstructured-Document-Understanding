"""Stage 12 — grounded answer generation with an element-aware prompt.

Builds the LLM context entirely from runtime chunk metadata (element type,
section, page, document name, chunk id, text) and asks the configured LLM to
answer using only that context, citing every claim as
``[chunk_id, page, document]``. Three behaviours from the spec are enforced:

  * **Adversarial gate** — if re-ranking found nothing relevant enough, return a
    fixed "insufficient information" message *without* calling the LLM.
  * **Visual queries** — surface ``IMAGE_PATH`` values (from figure chunks) so the
    UI can render the image.
  * **Multi-hop** — decompose into sub-questions (1 LLM call), run the full
    retrieval pipeline per sub-question via an injected ``retrieve_fn``, then
    synthesise the sub-answers (1 LLM call).

The LLM name always comes from ``config.OLLAMA_LLM_MODEL``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from src.config import config
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.retrieval.reranker import RerankResult

logger = logging.getLogger(__name__)

ADVERSARIAL_MESSAGE = (
    "I could not find sufficient information in the uploaded documents "
    "to answer this question."
)

_ANSWER_INSTRUCTIONS = (
    "You are a careful analyst. Answer the question using ONLY the provided "
    "context.\n"
    "- Cite every factual claim with [chunk_id, page_number, document_name] using "
    "the values shown in each context block's header.\n"
    "- If the context contains contradictions across documents, state both "
    "versions and cite both sources.\n"
    "- Synthesise across all element types — do not ignore tables or figures.\n"
    "- If you cannot answer from the context, say so explicitly."
)

_IMAGE_PATH_RE = re.compile(r"IMAGE_PATH:\s*(\S+)")

# Callback that runs the full retrieve+rerank pipeline for a sub-question.
RetrieveFn = Callable[[str], RerankResult]


@dataclass
class AnswerResult:
    """Stage 12 output."""

    answer: str
    adversarial: bool = False
    citations: list[dict] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    chunks_used: list[RetrievedChunk] = field(default_factory=list)
    sub_questions: list[str] | None = None
    sub_answers: list[str] | None = None


class AnswerGenerator:
    """Generates grounded answers from re-ranked context."""

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client
        self._model = config.ollama_llm_model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        query: str,
        rerank_result: RerankResult | None = None,
        *,
        query_type: str = "standard",
        visual: bool = False,
        retrieve_fn: RetrieveFn | None = None,
    ) -> AnswerResult:
        """Generate an answer for a query.

        Args:
            query: The user question.
            rerank_result: Stage 11 output for the main query (standard path).
            query_type: ``"multi_hop"`` triggers decomposition; else standard.
            visual: Whether the query asks to show a visual.
            retrieve_fn: Required for multi-hop — runs retrieve+rerank per
                sub-question.

        Returns:
            An :class:`AnswerResult`.
        """
        if query_type == "multi_hop" and retrieve_fn is not None:
            return self._generate_multi_hop(query, retrieve_fn, visual)
        if rerank_result is None:
            raise ValueError("rerank_result is required for non-multi-hop generation")
        return self._generate_standard(query, rerank_result, visual)

    # ------------------------------------------------------------------ #
    # Context construction (every value from runtime chunk metadata)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_context(chunks: list[RetrievedChunk]) -> str:
        """Build the element-aware context block from runtime chunk metadata."""
        blocks = []
        for chunk in chunks:
            label = (
                f"[{chunk.element_type.upper()}"
                f" | chunk_id: {chunk.chunk_id}"
                f" | {chunk.section_title}"
                f", page {chunk.page_number}"
                f", doc: {chunk.document_name}]"
            )
            blocks.append(f"{label}: {chunk.text}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ #
    # Standard generation
    # ------------------------------------------------------------------ #
    def _generate_standard(
        self, query: str, rerank_result: RerankResult, visual: bool
    ) -> AnswerResult:
        """Generate from re-ranked chunks, honouring the adversarial gate."""
        chunks = rerank_result.chunks
        if rerank_result.adversarial or not chunks:
            logger.info("Adversarial gate fired (max_score=%.3f); skipping LLM",
                        rerank_result.max_score)
            return AnswerResult(answer=ADVERSARIAL_MESSAGE, adversarial=True,
                                chunks_used=chunks)

        prompt = self._build_prompt(query, chunks, visual)
        answer = self._call_llm(prompt)
        figure_paths = [c.image_path for c in chunks if c.element_type == "figure" and c.image_path]
        image_paths = self._collect_image_paths(answer, figure_paths, visual)
        return AnswerResult(
            answer=answer,
            adversarial=False,
            citations=[self._citation(c) for c in chunks],
            image_paths=image_paths,
            chunks_used=chunks,
        )

    def _build_prompt(self, query: str, chunks: list[RetrievedChunk], visual: bool) -> str:
        """Assemble the full element-aware prompt."""
        context = self.build_context(chunks)
        extra = ""
        if visual:
            available = [c.image_path for c in chunks if c.element_type == "figure" and c.image_path]
            if available:
                extra = (
                    "\nThe user asked to see a visual. If a figure answers the "
                    "question, end your reply with a line 'IMAGE_PATH: <path>' "
                    f"choosing from these paths: {available}."
                )
        return (
            f"{_ANSWER_INSTRUCTIONS}{extra}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    # ------------------------------------------------------------------ #
    # Multi-hop generation
    # ------------------------------------------------------------------ #
    def _generate_multi_hop(
        self, query: str, retrieve_fn: RetrieveFn, visual: bool
    ) -> AnswerResult:
        """Decompose → retrieve per sub-question → synthesise."""
        sub_questions = self._decompose(query)
        sub_answers: list[str] = []
        all_chunks: list[RetrievedChunk] = []
        image_paths: list[str] = []

        for sub_q in sub_questions:
            rr = retrieve_fn(sub_q)
            sub = self._generate_standard(sub_q, rr, visual)
            sub_answers.append(sub.answer)
            all_chunks.extend(sub.chunks_used)
            image_paths.extend(sub.image_paths)

        final = self._synthesise(query, sub_questions, sub_answers)
        # Deduplicate citations/images across sub-answers.
        seen: set[str] = set()
        citations = []
        for c in all_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                citations.append(self._citation(c))
        adversarial = all(a == ADVERSARIAL_MESSAGE for a in sub_answers)
        return AnswerResult(
            answer=final, adversarial=adversarial, citations=citations,
            image_paths=list(dict.fromkeys(image_paths)), chunks_used=all_chunks,
            sub_questions=sub_questions, sub_answers=sub_answers,
        )

    def _decompose(self, query: str) -> list[str]:
        """One LLM call to split a complex query into 2-3 sub-questions."""
        prompt = (
            "Break the following question into 2 or 3 simpler sub-questions that, "
            "answered together, fully answer it. "
            'Respond with JSON {"sub_questions": ["...", "..."]}.\n\n'
            f"Question: {query}\n\nJSON:"
        )
        raw = self._call_llm(prompt, json_mode=True)
        try:
            items = json.loads(raw).get("sub_questions", [])
        except (json.JSONDecodeError, AttributeError, TypeError):
            items = []
        subs = [self._coerce_subquestion(s) for s in items]
        subs = [s for s in subs if s]
        return subs[:3] or [query]

    @staticmethod
    def _coerce_subquestion(item) -> str:
        """Normalise a sub-question that the LLM may return as a string or dict."""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def _synthesise(
        self, query: str, sub_questions: list[str], sub_answers: list[str]
    ) -> str:
        """One LLM call to combine sub-answers into the final answer."""
        joined = "\n\n".join(
            f"Sub-question: {q}\nSub-answer: {a}"
            for q, a in zip(sub_questions, sub_answers)
        )
        prompt = (
            "Combine the sub-answers below into one coherent answer to the original "
            "question. Preserve all citations [chunk_id, page, document] from the "
            "sub-answers. If the sub-answers conflict, note the conflict.\n\n"
            f"Original question: {query}\n\n{joined}\n\nFinal answer:"
        )
        return self._call_llm(prompt)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _citation(chunk: RetrievedChunk) -> dict:
        """Citation record drawn entirely from runtime chunk metadata."""
        return {
            "chunk_id": chunk.chunk_id, "page_number": chunk.page_number,
            "document_name": chunk.document_name, "element_type": chunk.element_type,
            "confidence": chunk.confidence,
        }

    @staticmethod
    def _collect_image_paths(answer: str, figure_paths: list[str], visual: bool) -> list[str]:
        """Combine IMAGE_PATH lines emitted by the LLM with figure-chunk paths."""
        paths = _IMAGE_PATH_RE.findall(answer)
        if visual:
            paths.extend(figure_paths)
        # Preserve order, drop duplicates and empties.
        return [p for p in dict.fromkeys(paths) if p]

    def _call_llm(self, prompt: str, json_mode: bool = False) -> str:
        """Invoke the configured Ollama LLM and return the response text."""
        client = self._get_llm()
        try:
            kwargs = {"options": {"temperature": 0.0}}
            if json_mode:
                kwargs["format"] = "json"
            response = client.generate(model=self._model, prompt=prompt, **kwargs)
            return (getattr(response, "response", None) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Answer LLM call failed: %s", exc)
            return ADVERSARIAL_MESSAGE if not json_mode else "{}"

    def _get_llm(self):
        if self._llm is None:
            import ollama
            self._llm = ollama.Client(host=config.ollama_base_url)
        return self._llm


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = " ".join(sys.argv[1:]) or "What is Acme Corporation's revenue and who is its CEO?"
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import Reranker

    retrieved = HybridRetriever().retrieve(query)
    reranked = Reranker().rerank(query, retrieved.chunks)
    result = AnswerGenerator().generate(query, reranked, visual=False)
    print(f"\nQuery: {query}")
    print(f"Adversarial: {result.adversarial}")
    print(f"\nAnswer:\n{result.answer}")
    print(f"\nCitations: {[(c['chunk_id'], c['page_number'], c['document_name']) for c in result.citations]}")
    print(f"Image paths: {result.image_paths}")
