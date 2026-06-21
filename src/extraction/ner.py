"""Stage 5, Step 1 — NER via Gemini Flash batch extraction (Option B).

Replaces the previous spaCy-transformer + fastcoref pipeline with a single
Gemini Flash API call that processes ALL chunks in one request.

Gemini 2.0 Flash has a 1M-token context window, so for most documents the
entire chunked text fits in one call. The model returns a structured JSON list
of entities across the full document, achieving better cross-chunk coreference
naturally (the LLM sees the whole document at once) and doing it in seconds
instead of minutes.

Fallback: if the LLM call fails, a lightweight spaCy ``en_core_web_sm`` pass
is used (no fastcoref — still ~100x faster than the old transformer model).

This module preserves the exact same public interface (Entity, Mention,
NERResult, NERExtractor.extract()) so all downstream stages (GLiNER, resolver,
graph publisher) continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from src.ingestion.chunker import Chunk, ChunkResult

logger = logging.getLogger(__name__)

_NER_CONFIDENCE = 0.85
_FALLBACK_SPACY_MODEL = "en_core_web_sm"

# Label map for spaCy fallback (sm model labels)
_LABEL_MAP: dict[str, str] = {
    "PERSON": "Person",
    "ORG": "Organisation",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "EVENT": "Event",
    "PRODUCT": "Object",
    "WORK_OF_ART": "Object",
    "DATE": "Date",
    "TIME": "Date",
    "MONEY": "Quantity",
    "PERCENT": "Quantity",
    "QUANTITY": "Quantity",
    "LAW": "LegalDocument",
}

# Gemini system prompt for batch NER
_NER_SYSTEM = (
    "You are a precise Named Entity Recognition system. "
    "Extract all named entities from the provided document chunks. "
    "For each entity return: name (canonical form), type (one of: "
    "Person, Organisation, Location, Event, Object, Date, Quantity, "
    "LegalDocument, Concept, FinancialMetric, Product), "
    "and chunk_ids (list of chunk IDs where it appears). "
    "Resolve coreferences — if 'Apple Inc.', 'Apple', and 'the company' "
    "refer to the same entity, merge them under the canonical name. "
    'Respond ONLY with valid JSON: {"entities": [{"name": "...", "type": "...", "chunk_ids": [...]}]}'
)


@dataclass
class Mention:
    """A single occurrence of an entity in a specific chunk."""

    chunk_id: str
    page_number: int


@dataclass
class Entity:
    """A resolved entity with its canonical name, aliases, and mentions.

    Shared across Stages 5–8. The global resolver (Stage 7) later merges these
    across documents into canonical graph nodes.
    """

    name: str
    entity_type: str
    document_id: str
    document_name: str
    aliases: list[str] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    first_chunk_id: str | None = None
    first_page: int | None = None
    confidence: float = _NER_CONFIDENCE
    extraction_method: str = "gemini_ner"

    @property
    def key(self) -> tuple[str, str]:
        """Dedup/merge key: normalised lowercase name + type."""
        return (" ".join(self.name.lower().split()), self.entity_type)

    def add_alias(self, alias: str) -> None:
        alias = alias.strip()
        if not alias or alias.lower() == self.name.lower():
            return
        if alias.lower() not in {a.lower() for a in self.aliases}:
            self.aliases.append(alias)

    def add_mention(self, chunk_id: str, page_number: int) -> None:
        if any(m.chunk_id == chunk_id for m in self.mentions):
            return
        self.mentions.append(Mention(chunk_id=chunk_id, page_number=page_number))


@dataclass
class NERResult:
    """Stage 5 Step 1 output (entities only; triples are added downstream)."""

    entities: list[Entity]
    entity_count: int = 0

    def __post_init__(self) -> None:
        if not self.entity_count:
            self.entity_count = len(self.entities)


class NERExtractor:
    """Gemini Flash batch NER with spaCy-sm fallback.

    Primary path: all chunks are sent to Gemini Flash in a single API call.
    The model sees the entire document at once, enabling natural cross-chunk
    coreference resolution without a separate neural coref model.

    Fallback path: lightweight spaCy en_core_web_sm (no transformer, no coref).
    Used when no cloud API key is configured or if the API call fails.
    """

    def __init__(self) -> None:
        # Check if any cloud key is available
        import os
        has_cloud = any([
            os.getenv("CEREBRAS_API_KEY"),
            os.getenv("GROQ_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
        ])
        self._use_cloud = has_cloud
        if has_cloud:
            logger.info("NERExtractor ready (Gemini Flash batch mode via LLM Router)")
        else:
            logger.warning(
                "NERExtractor: No cloud API keys found — falling back to spaCy en_core_web_sm. "
                "Set CEREBRAS_API_KEY or GEMINI_API_KEY in .env for best quality."
            )

    def extract(self, chunk_result: ChunkResult) -> NERResult:
        """Extract entities from all chunks in one batch call.

        Args:
            chunk_result: Stage 3 output (chunked document).

        Returns:
            A :class:`NERResult` with all extracted entities.
        """
        text_chunks = [c for c in chunk_result.chunks if c.text.strip()]
        if not text_chunks:
            return NERResult(entities=[])

        if self._use_cloud:
            try:
                result = self._extract_via_cloud(text_chunks)
                logger.info(
                    "Gemini NER found %d entities across %d chunks",
                    result.entity_count, len(text_chunks),
                )
                return result
            except Exception as exc:
                logger.warning(
                    "Gemini NER failed (%s); falling back to spaCy sm", exc
                )

        # Fallback: spaCy sm
        result = self._extract_via_spacy(text_chunks)
        logger.info(
            "spaCy sm NER found %d entities across %d chunks",
            result.entity_count, len(text_chunks),
        )
        return result

    # ------------------------------------------------------------------ #
    # Cloud path (Gemini Flash)
    # ------------------------------------------------------------------ #

    def _extract_via_cloud(self, chunks: list[Chunk]) -> NERResult:
        """Send chunks to the LLM Router and parse the result.
        
        Sends all chunks in one batch to Gemini (1M context). If that fails
        (rate limit / quota exhausted), falls back to smaller batches
        that fit within Groq's 6K TPM limit (~4,500 tokens per batch).
        """
        chunk_index = {c.chunk_id: c for c in chunks}
        
        # First attempt: try sending all chunks to Gemini in one shot
        blocks = []
        for c in chunks:
            blocks.append(f"[CHUNK {c.chunk_id} | page {c.page_number}]\n{c.text}")
        full_text = "\n\n---\n\n".join(blocks)
        
        prompt = (
            f"{_NER_SYSTEM}\n\n"
            f"Document chunks to analyze:\n\n{full_text}\n\n"
            f"JSON response:"
        )

        from src.llm_router import agenerate, loads_lenient

        # Try full document first (works great with Gemini 1M context)
        try:
            raw = self._run_async(agenerate(prompt, json_mode=True, model="router-ner"))
            data = loads_lenient(raw)
            if data and "entities" in data:
                return self._parse_cloud_response(data["entities"], chunk_index, chunks)
        except Exception as exc:
            logger.warning("Full-document NER failed (%s); trying batched mode", exc)
        
        # Fallback: process in small batches (~10 chunks each) to fit Groq TPM limits
        logger.info("Switching to batched NER (10 chunks/batch) for Groq fallback")
        all_entities: list[dict] = []
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_blocks = [f"[CHUNK {c.chunk_id} | page {c.page_number}]\n{c.text}" for c in batch]
            batch_text = "\n\n---\n\n".join(batch_blocks)
            batch_prompt = (
                f"{_NER_SYSTEM}\n\n"
                f"Document chunks to analyze:\n\n{batch_text}\n\n"
                f"JSON response:"
            )
            try:
                raw = self._run_async(agenerate(batch_prompt, json_mode=True, model="router-ner"))
                data = loads_lenient(raw)
                if data and "entities" in data:
                    all_entities.extend(data["entities"])
            except Exception as exc:
                logger.warning("Batch NER failed for chunks %d-%d: %s", i, i+batch_size, exc)
                continue
        
        if all_entities:
            return self._parse_cloud_response(all_entities, chunk_index, chunks)
        
        raise ValueError("All cloud NER attempts failed")

    @staticmethod
    def _run_async(coro):
        """Run an async coroutine from a sync context safely."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result(timeout=180)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    @staticmethod
    def _parse_cloud_response(
        raw_entities: list[dict],
        chunk_index: dict[str, Chunk],
        all_chunks: list[Chunk],
    ) -> NERResult:
        """Convert Gemini's JSON entity list into Entity dataclass instances."""
        entities: dict[tuple[str, str], Entity] = {}

        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "Concept")).strip()
            chunk_ids = item.get("chunk_ids", [])
            if not name or not chunk_ids:
                continue

            key = (" ".join(name.lower().split()), etype)
            if key not in entities:
                first_chunk = chunk_index.get(str(chunk_ids[0])) if chunk_ids else None
                entities[key] = Entity(
                    name=name,
                    entity_type=etype,
                    document_id=all_chunks[0].document_id if all_chunks else "",
                    document_name=all_chunks[0].document_name if all_chunks else "",
                    first_chunk_id=first_chunk.chunk_id if first_chunk else None,
                    first_page=first_chunk.page_number if first_chunk else None,
                    confidence=_NER_CONFIDENCE,
                    extraction_method="gemini_ner",
                )

            entity = entities[key]
            for cid in chunk_ids:
                cid_str = str(cid)
                chunk = chunk_index.get(cid_str)
                if chunk:
                    entity.add_mention(chunk.chunk_id, chunk.page_number)

        return NERResult(entities=list(entities.values()))

    # ------------------------------------------------------------------ #
    # Fallback path (spaCy sm — CPU-friendly, no transformer)
    # ------------------------------------------------------------------ #

    def _extract_via_spacy(self, chunks: list[Chunk]) -> NERResult:
        """Lightweight fallback using spaCy en_core_web_sm (no transformer)."""
        import spacy
        try:
            nlp = spacy.load(_FALLBACK_SPACY_MODEL)
        except OSError:
            logger.error(
                "spaCy model '%s' not found. Run: uv sync", _FALLBACK_SPACY_MODEL
            )
            return NERResult(entities=[])

        entities: dict[tuple[str, str], Entity] = {}
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            try:
                doc = nlp(chunk.text[:10_000])  # spaCy sm has no context limit issue
            except Exception as exc:
                logger.warning("spaCy sm failed on chunk %s: %s", chunk.chunk_id, exc)
                continue

            for ent in doc.ents:
                etype = _LABEL_MAP.get(ent.label_)
                if not etype:
                    continue
                name = ent.text.strip()
                if not name:
                    continue
                key = (" ".join(name.lower().split()), etype)
                if key not in entities:
                    entities[key] = Entity(
                        name=name,
                        entity_type=etype,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        first_chunk_id=chunk.chunk_id,
                        first_page=chunk.page_number,
                        confidence=0.75,
                        extraction_method="spacy_sm",
                    )
                entities[key].add_mention(chunk.chunk_id, chunk.page_number)

        return NERResult(entities=list(entities.values()))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.extraction.ner <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser

    pr = DocumentParser().parse(sys.argv[1])
    cr = Chunker().chunk(pr)
    res = NERExtractor().extract(cr)
    print(f"\n{res.entity_count} entities:")
    for e in res.entities:
        print(
            f"  {e.entity_type:15s} {e.name!r} "
            f"method={e.extraction_method} "
            f"mentions={[m.chunk_id for m in e.mentions]}"
        )
