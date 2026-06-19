"""Stage 5, Step 3 — LLM-fallback relation (triple) extraction.

Generates subject-predicate-object triples with the configured Ollama LLM, but
**only** for chunks that plausibly contain cross-element relationships — those
mentioning two or more distinct entities (a relation needs at least two
participants). The model is asked for strict JSON, which is parsed into
:class:`~src.extraction.gliner_extractor.Triple` objects that populate
``ExtractionResult.triples``.

The LLM name always comes from ``config.OLLAMA_LLM_MODEL``; nothing is hardcoded.
This is the sole source of triples in the pipeline (see the GLiNER-entities-only
decision), so the prompt is grounded in the entities already extracted upstream.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from src.config import config
from src.extraction.gliner_extractor import ExtractionResult, Triple
from src.extraction.ner import Entity
from src.ingestion.chunker import Chunk, ChunkResult

logger = logging.getLogger(__name__)

_MIN_ENTITIES_PER_CHUNK = 2
_MAX_ENTITIES_IN_PROMPT = 12
_LLM_TRIPLE_CONFIDENCE = 0.6

_INSTRUCTION = (
    "You extract factual relationships from text. Using ONLY the text provided, "
    "list the relationships between the entities as subject-predicate-object "
    "triples. Use short verb-phrase predicates (e.g. 'works at', 'located in', "
    "'acquired'). Do not invent facts not stated in the text. "
    'Respond with a JSON object of the form {"triples": [{"subject": "...", '
    '"predicate": "...", "object": "..."}]} and nothing else.'
)


@dataclass
class _Candidate:
    """A chunk selected for LLM relation extraction and its known entities."""

    chunk: Chunk
    entity_names: list[str]


class RelationExtractor:
    """LLM-based triple extractor over relationship-bearing chunks."""

    def __init__(self) -> None:
        import ollama

        self._client = ollama.Client(host=config.ollama_base_url)
        self._model = config.ollama_llm_model
        logger.info("RelationExtractor ready (LLM from config: %s)", self._model)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(
        self, chunk_result: ChunkResult, extraction_result: ExtractionResult
    ) -> ExtractionResult:
        """Populate ``extraction_result.triples`` with LLM-extracted triples.

        Args:
            chunk_result: Stage 3 output (for chunk text lookup).
            extraction_result: Stage 5 entities; its ``triples`` are filled and
                ``triple_count`` refreshed in place.

        Returns:
            The same :class:`ExtractionResult`, with triples added.
        """
        chunk_by_id = {c.chunk_id: c for c in chunk_result.chunks}
        candidates = self._select_candidates(extraction_result.entities, chunk_by_id)
        logger.info(
            "Relation extraction on %d/%d chunks (>=%d entities each)",
            len(candidates),
            len(chunk_by_id),
            _MIN_ENTITIES_PER_CHUNK,
        )

        triples: list[Triple] = []
        seen: set[tuple[str, str, str, str]] = set()
        for cand in candidates:
            for triple in self._extract_from_chunk(cand):
                dedup = (
                    triple.subject.lower(), triple.predicate,
                    triple.object.lower(), triple.chunk_id,
                )
                if dedup in seen:
                    continue
                seen.add(dedup)
                triples.append(triple)

        extraction_result.triples.extend(triples)
        extraction_result.recount()
        logger.info("Extracted %d triples", len(triples))
        return extraction_result

    # ------------------------------------------------------------------ #
    # Candidate selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _entity_chunk_index(entities: list[Entity]) -> dict[str, list[str]]:
        """Map each chunk_id to the distinct entity names mentioned in it."""
        index: dict[str, list[str]] = {}
        for entity in entities:
            for mention in entity.mentions:
                names = index.setdefault(mention.chunk_id, [])
                if entity.name not in names:
                    names.append(entity.name)
        return index

    def _select_candidates(
        self, entities: list[Entity], chunk_by_id: dict[str, Chunk]
    ) -> list[_Candidate]:
        """Pick chunks with >= the minimum number of distinct entities."""
        index = self._entity_chunk_index(entities)
        candidates: list[_Candidate] = []
        for chunk_id, names in index.items():
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None or not chunk.text.strip():
                continue
            if len(names) >= _MIN_ENTITIES_PER_CHUNK:
                candidates.append(_Candidate(chunk=chunk, entity_names=names[:_MAX_ENTITIES_IN_PROMPT]))
        return candidates

    # ------------------------------------------------------------------ #
    # Per-chunk extraction
    # ------------------------------------------------------------------ #
    def _extract_from_chunk(self, cand: _Candidate) -> list[Triple]:
        """Call the LLM for one chunk and parse the returned triples."""
        prompt = self._build_prompt(cand.chunk.text, cand.entity_names)
        raw = self._call_llm(prompt)
        if raw is None:
            return []
        return self._parse_triples(raw, cand.chunk)

    @staticmethod
    def _build_prompt(text: str, entity_names: list[str]) -> str:
        """Build a grounded relation-extraction prompt for one chunk."""
        entities = ", ".join(entity_names)
        return (
            f"{_INSTRUCTION}\n\n"
            f"Known entities: {entities}\n\n"
            f"Text:\n{text}\n\n"
            f"JSON:"
        )

    def _call_llm(self, prompt: str) -> str | None:
        """Invoke the configured Ollama LLM in JSON mode; return raw text."""
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                format="json",
                options={"temperature": 0.0},
            )
            return (getattr(response, "response", None) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Relation LLM call failed: %s", exc)
            return None

    def _parse_triples(self, raw: str, chunk: Chunk) -> list[Triple]:
        """Parse the LLM's JSON response into validated :class:`Triple` objects."""
        data = self._loads_lenient(raw)
        if data is None:
            logger.debug("Unparseable relation JSON for chunk %s: %r", chunk.chunk_id, raw[:120])
            return []
        items = data.get("triples") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []

        triples: list[Triple] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            subj_val, pred_val, obj_val = (
                item.get("subject"), item.get("predicate"), item.get("object"),
            )
            if subj_val is None or pred_val is None or obj_val is None:
                continue  # JSON null fields → incomplete triple
            subject = str(subj_val).strip()
            predicate = self._normalise_predicate(str(pred_val))
            obj = str(obj_val).strip()
            # Reject empty or literal null-like placeholders.
            if not subject or not predicate or not obj:
                continue
            if subject.lower() in {"none", "null"} or obj.lower() in {"none", "null"}:
                continue
            triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    confidence=_LLM_TRIPLE_CONFIDENCE,
                    extraction_method="llm",
                )
            )
        return triples

    @staticmethod
    def _loads_lenient(raw: str) -> dict | list | None:
        """Parse JSON, tolerating surrounding prose by extracting the JSON span."""
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        match = re.search(r"[\{\[].*[\}\]]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _normalise_predicate(predicate: str) -> str:
        """Lowercase, collapse whitespace, and snake_case a predicate phrase."""
        cleaned = re.sub(r"[^a-z0-9]+", "_", predicate.strip().lower())
        return cleaned.strip("_")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.extraction.relation_extractor <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.extraction.domain_detector import DomainDetector
    from src.extraction.gliner_extractor import GLiNERExtractor
    from src.extraction.ner import NERExtractor
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser

    pr = DocumentParser().parse(sys.argv[1])
    cr = Chunker().chunk(pr)
    dom = DomainDetector().detect(cr)
    ner = NERExtractor().extract(cr)
    er = GLiNERExtractor().extract(cr, dom, prior_entities=ner.entities)
    er = RelationExtractor().extract(cr, er)
    print(f"\nentities={er.entity_count} triples={er.triple_count}")
    for t in er.triples:
        print(f"  ({t.subject}) -[{t.predicate}]-> ({t.object})  @{t.chunk_id}")
