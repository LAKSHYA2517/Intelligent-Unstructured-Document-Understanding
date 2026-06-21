"""Stage 4 — document domain detection (one LLM call).

Classifies a document into exactly one of five domains —
``finance | legal | medical | technical | general`` — using a single call to the
configured Ollama LLM over the document's first 500 words. The returned domain is
the model's actual answer (parsed/normalised, never hardcoded); the confidence is
derived from how cleanly that answer matched a valid label. The detected domain
drives the runtime selection of GLiNER extension labels in Stage 5.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.ingestion.chunker import ChunkResult

logger = logging.getLogger(__name__)

VALID_DOMAINS: tuple[str, ...] = ("finance", "legal", "medical", "technical", "general")
_DEFAULT_DOMAIN = "general"
_SAMPLE_WORDS = 500

# Exact instruction mandated by the spec.
_INSTRUCTION = (
    "Classify this document into exactly one domain: finance, legal, medical, "
    "technical, general. Return only the single word domain label, nothing else."
)


@dataclass
class DomainResult:
    """Stage 4 output.

    Attributes:
        domain: One of :data:`VALID_DOMAINS` (the LLM's parsed answer).
        confidence: 0–1 score reflecting how cleanly the answer matched a label.
        raw_response: The model's raw text, kept for debugging/provenance.
    """

    domain: str
    confidence: float
    raw_response: str = ""


class DomainDetector:
    """Detects a document's domain with a single LLM Router call."""

    def __init__(self) -> None:
        logger.info("DomainDetector ready (via LLM Router: Cerebras -> Groq -> Gemini)")

    def detect(self, chunk_result: ChunkResult) -> DomainResult:
        """Classify a chunked document into one domain."""
        text = self._first_words(chunk_result, _SAMPLE_WORDS)
        if not text.strip():
            logger.warning("Empty document text; defaulting domain to '%s'", _DEFAULT_DOMAIN)
            return DomainResult(domain=_DEFAULT_DOMAIN, confidence=0.0)

        prompt = f"{_INSTRUCTION}\n\nDocument:\n{text}\n\nDomain:"
        try:
            from src.llm_router import generate_sync
            raw = generate_sync(prompt).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Domain LLM call failed (%s); defaulting to '%s'", exc, _DEFAULT_DOMAIN)
            return DomainResult(domain=_DEFAULT_DOMAIN, confidence=0.0)

        domain, confidence = self._parse_domain(raw)
        logger.info("Detected domain '%s' (confidence=%.2f, raw=%r)", domain, confidence, raw[:60])
        return DomainResult(domain=domain, confidence=confidence, raw_response=raw)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_words(chunk_result: ChunkResult, n: int) -> str:
        """Return the first ``n`` whitespace-delimited words across all chunks."""
        words: list[str] = []
        for chunk in chunk_result.chunks:
            for word in chunk.text.split():
                words.append(word)
                if len(words) >= n:
                    return " ".join(words)
        return " ".join(words)

    @staticmethod
    def _parse_domain(raw: str) -> tuple[str, float]:
        """Map a raw LLM response to a valid domain and a confidence score.

        Prefers an exact single-word match, then a whole-word match, then a
        substring match, before falling back to ``general``.
        """
        clean = raw.strip().lower()
        if clean in VALID_DOMAINS:
            return clean, 0.95
        tokens = set(re.findall(r"[a-z]+", clean))
        for domain in VALID_DOMAINS:
            if domain in tokens:
                return domain, 0.85
        for domain in VALID_DOMAINS:
            if domain in clean:
                return domain, 0.70
        return _DEFAULT_DOMAIN, 0.30


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.extraction.domain_detector <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser

    pr = DocumentParser().parse(sys.argv[1])
    cr = Chunker().chunk(pr)
    result = DomainDetector().detect(cr)
    print(f"\ndomain={result.domain} confidence={result.confidence} raw={result.raw_response!r}")
