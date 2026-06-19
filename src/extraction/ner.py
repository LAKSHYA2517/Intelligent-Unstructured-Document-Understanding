"""Stage 5, Step 1 — NER + co-reference resolution (spaCy + fastcoref).

Runs the transformer spaCy pipeline (``en_core_web_trf``) together with neural
co-reference (``fastcoref``, the maintained replacement for coreferee, which is
version-incompatible with our docling) over a **sliding window of 3 consecutive
chunks**, sliding by 1 so adjacent windows overlap by 2 chunks. Within each
window, co-reference clusters resolve pronouns and aliases to a single canonical
entity name, and every chunk a mention appears in is recorded for the
``MENTIONED_IN`` / ``FIRST_MENTIONED_IN`` graph edges.

This module defines the shared :class:`Entity` dataclass consumed by the GLiNER
extractor, the global resolver, and the graph publisher. spaCy NER labels are
mapped onto the pipeline's base entity types; nothing here is domain-specific.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.ingestion.chunker import Chunk, ChunkResult

logger = logging.getLogger(__name__)

_SPACY_MODEL = "en_core_web_trf"
_WINDOW_SIZE = 3
_NER_CONFIDENCE = 0.8

# spaCy entity label → pipeline base entity type. Labels not present here are
# ignored as too noisy (e.g. CARDINAL, ORDINAL, NORP).
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

    Attributes:
        name: Canonical display name.
        entity_type: Base type (Person, Organisation, Location, Event, Object,
            Date, Quantity, LegalDocument, Concept, …).
        document_id: Owning document.
        document_name: Owning document's real filename.
        aliases: Alternate surface forms seen for this entity.
        mentions: Chunks (with page) where the entity appears.
        first_chunk_id: Chunk of the earliest mention (FIRST_MENTIONED_IN).
        first_page: Page of the earliest mention.
        confidence: Extraction confidence in 0–1.
        extraction_method: Provenance tag (``spacy_ner``, ``gliner``, ``llm`` …).
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
    extraction_method: str = "spacy_ner"

    @property
    def key(self) -> tuple[str, str]:
        """Dedup/merge key: normalised lowercase name + type."""
        return (" ".join(self.name.lower().split()), self.entity_type)

    def add_alias(self, alias: str) -> None:
        """Record an alternate surface form (case-insensitive, deduplicated)."""
        alias = alias.strip()
        if not alias or alias.lower() == self.name.lower():
            return
        if alias.lower() not in {a.lower() for a in self.aliases}:
            self.aliases.append(alias)

    def add_mention(self, chunk_id: str, page_number: int) -> None:
        """Record a chunk this entity is mentioned in (deduplicated)."""
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
    """spaCy-transformer NER + fastcoref co-reference over sliding windows."""

    def __init__(self) -> None:
        import spacy
        from fastcoref import spacy_component  # noqa: F401 - registers the pipe

        self._nlp = spacy.load(_SPACY_MODEL)
        if "fastcoref" not in self._nlp.pipe_names:
            self._nlp.add_pipe("fastcoref")
        logger.info("NERExtractor ready (%s + fastcoref)", _SPACY_MODEL)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract(self, chunk_result: ChunkResult) -> NERResult:
        """Extract and co-reference-resolve entities from a chunked document.

        Args:
            chunk_result: Stage 3 output.

        Returns:
            An :class:`NERResult` whose ``entity_count`` is a real runtime value.
        """
        text_chunks = [c for c in chunk_result.chunks if c.text.strip()]
        if not text_chunks:
            return NERResult(entities=[])

        entities: dict[tuple[str, str], Entity] = {}
        for window in self._windows(text_chunks):
            self._process_window(window, entities)

        result = NERResult(entities=list(entities.values()))
        logger.info(
            "spaCy+coref found %d entities across %d chunks",
            result.entity_count,
            len(text_chunks),
        )
        return result

    # ------------------------------------------------------------------ #
    # Windowing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _windows(chunks: list[Chunk]) -> list[list[Chunk]]:
        """Yield sliding windows of up to 3 chunks, sliding by 1."""
        if len(chunks) <= _WINDOW_SIZE:
            return [chunks]
        return [chunks[i : i + _WINDOW_SIZE] for i in range(len(chunks) - _WINDOW_SIZE + 1)]

    def _process_window(
        self, window: list[Chunk], entities: dict[tuple[str, str], Entity]
    ) -> None:
        """Run NER+coref on one window and merge results into ``entities``."""
        text, offsets = self._concat_with_offsets(window)
        try:
            doc = self._nlp(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("spaCy failed on a window (%s); skipping", exc)
            return

        clusters = self._char_clusters(doc)
        canonical_by_span = self._canonical_by_span(doc, clusters)

        for ent in doc.ents:
            etype = _LABEL_MAP.get(ent.label_)
            if etype is None:
                continue
            surface = ent.text.strip()
            if not surface:
                continue
            canonical = canonical_by_span.get((ent.start_char, ent.end_char), surface)
            chunk = self._chunk_for_offset(ent.start_char, offsets)
            if chunk is None:
                continue
            self._merge_entity(
                entities, canonical, etype, surface, chunk, window[0].document_id,
                window[0].document_name,
            )
            # Attribute every co-referent mention (incl. pronouns) of this entity.
            self._attach_cluster_mentions(
                entities, canonical, etype, ent, clusters, offsets, window
            )

    # ------------------------------------------------------------------ #
    # Entity merging
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_entity(
        entities: dict[tuple[str, str], Entity],
        canonical: str,
        etype: str,
        surface: str,
        chunk: Chunk,
        document_id: str,
        document_name: str,
    ) -> None:
        """Create or update the entity for ``canonical`` and record the mention."""
        key = (" ".join(canonical.lower().split()), etype)
        entity = entities.get(key)
        if entity is None:
            entity = Entity(
                name=canonical,
                entity_type=etype,
                document_id=document_id,
                document_name=document_name,
                first_chunk_id=chunk.chunk_id,
                first_page=chunk.page_number,
            )
            entities[key] = entity
        entity.add_alias(surface)
        entity.add_mention(chunk.chunk_id, chunk.page_number)

    def _attach_cluster_mentions(
        self,
        entities: dict[tuple[str, str], Entity],
        canonical: str,
        etype: str,
        ent,
        clusters: list[list[tuple[int, int]]],
        offsets: list[tuple[int, int, Chunk]],
        window: list[Chunk],
    ) -> None:
        """Add mentions for pronouns/aliases co-referring to a named entity."""
        key = (" ".join(canonical.lower().split()), etype)
        entity = entities.get(key)
        if entity is None:
            return
        for cluster in clusters:
            if not any(s <= ent.start_char < e for (s, e) in cluster):
                continue
            for (s, _e) in cluster:
                chunk = self._chunk_for_offset(s, offsets)
                if chunk is not None:
                    entity.add_mention(chunk.chunk_id, chunk.page_number)

    # ------------------------------------------------------------------ #
    # Co-reference helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _char_clusters(doc) -> list[list[tuple[int, int]]]:
        """Return fastcoref clusters as lists of (start_char, end_char) spans."""
        return list(getattr(doc._, "coref_clusters", []) or [])

    def _canonical_by_span(
        self, doc, clusters: list[list[tuple[int, int]]]
    ) -> dict[tuple[int, int], str]:
        """Map each clustered span to its cluster's canonical name."""
        mapping: dict[tuple[int, int], str] = {}
        for cluster in clusters:
            canonical = self._pick_canonical(doc, cluster)
            if not canonical:
                continue
            for span in cluster:
                mapping[span] = canonical
        return mapping

    @staticmethod
    def _pick_canonical(doc, cluster: list[tuple[int, int]]) -> str | None:
        """Choose the most name-like, longest span in a cluster as canonical."""
        best: str | None = None
        best_score = (-1, -1)  # (has_proper_noun, length)
        for s, e in cluster:
            span = doc.char_span(s, e, alignment_mode="expand")
            if span is None:
                continue
            text = span.text.strip()
            if not text:
                continue
            has_propn = any(t.pos_ == "PROPN" for t in span)
            is_pron = all(t.pos_ == "PRON" for t in span)
            if is_pron:
                continue
            score = (1 if has_propn else 0, len(text))
            if score > best_score:
                best_score = score
                best = text
        return best

    # ------------------------------------------------------------------ #
    # Offset mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _concat_with_offsets(
        window: list[Chunk],
    ) -> tuple[str, list[tuple[int, int, Chunk]]]:
        """Join window chunk texts, tracking each chunk's char span in the join."""
        parts: list[str] = []
        offsets: list[tuple[int, int, Chunk]] = []
        cursor = 0
        sep = "\n\n"
        for i, chunk in enumerate(window):
            text = chunk.text
            start = cursor
            end = start + len(text)
            offsets.append((start, end, chunk))
            parts.append(text)
            cursor = end + (len(sep) if i < len(window) - 1 else 0)
        return sep.join(parts), offsets

    @staticmethod
    def _chunk_for_offset(offset: int, offsets: list[tuple[int, int, Chunk]]) -> Chunk | None:
        """Return the chunk whose char span contains ``offset``."""
        for start, end, chunk in offsets:
            if start <= offset < end:
                return chunk
        return offsets[-1][2] if offsets else None


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
        print(f"  {e.entity_type:13s} {e.name!r} aliases={e.aliases} "
              f"mentions={[m.chunk_id for m in e.mentions]}")
