"""Stage 3 — element-aware chunking.

Turns the reading-order :class:`~src.ingestion.parser.ParsedElement` stream (plus
Stage 2 figure descriptions, when available) into a list of :class:`Chunk`
objects carrying everything downstream stages need: ``element_type``,
``table_json``, ``section_title`` (nearest parent heading), ``page_number``,
``document_id``/``document_name``, and ``position_in_doc`` for ``PRECEDES`` edges.

Chunking rules (from the spec):
  * a heading and the body paragraphs beneath it form one chunk group;
  * a table plus its caption is a single chunk, kept as both ``table_json`` and
    rendered text, never split mid-row, multi-page tables already merged upstream;
  * figures, captions (of figures), and footnotes are standalone chunks so the
    cross-modal/structural graph edges can attach to them;
  * long text groups are capped at ``config.chunk_max_tokens`` and split only on
    sentence boundaries — never mid-sentence, never mid-table.

The Chunk shape matches the spec exactly and adds two optional linkage fields
(``source_element_id``, ``image_path``) so figure crops and graph provenance
survive chunking; they default to ``None`` and never carry hardcoded values.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.config import config
from src.ingestion.parser import ParsedElement, ParseResult

logger = logging.getLogger(__name__)

# Rough upper-bound token estimate: count word-ish runs and standalone
# punctuation. This over-counts slightly versus BPE, keeping chunks safely under
# any model's real token limit without pulling in a tokenizer download.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# Sentence boundary: end punctuation followed by whitespace. Used so the token
# cap never splits mid-sentence.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _estimate_tokens(text: str) -> int:
    """Return an upper-bound token estimate for a piece of text."""
    return len(_TOKEN_RE.findall(text))


@dataclass
class Chunk:
    """A single retrievable unit of a document.

    Attributes:
        chunk_id: ``{document_id}_chunk_{index:04d}``.
        text: The chunk's text (table markdown for tables, description for figures).
        table_json: Structured table for table chunks, else ``None``.
        element_type: ``paragraph|table|figure|caption|footnote|heading``.
        page_number: Real 1-based page number from Docling.
        document_id: Owning document id.
        document_name: Real uploaded filename.
        section_title: Nearest parent heading text ("" if none).
        position_in_doc: Sequential 0-based index, drives ``PRECEDES`` edges.
        source_element_id: Originating :class:`ParsedElement` id (graph provenance).
        image_path: Saved figure crop path for figure chunks, else ``None``.
        embedding: Dense vector for this chunk, set by Stage 6, else ``None``.
    """

    chunk_id: str
    text: str
    table_json: dict[str, Any] | None
    element_type: str
    page_number: int
    document_id: str
    document_name: str
    section_title: str
    position_in_doc: int
    source_element_id: str | None = None
    image_path: str | None = None
    embedding: list[float] | None = None


@dataclass
class ChunkResult:
    """Stage 3 output. ``chunk_count`` is read directly by the Streamlit UI."""

    chunks: list[Chunk]
    chunk_count: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.chunk_count:
            self.chunk_count = len(self.chunks)


class Chunker:
    """Builds :class:`ChunkResult` from a :class:`ParseResult` (+ figure data)."""

    def __init__(self, max_tokens: int | None = None) -> None:
        self.max_tokens = max_tokens or config.chunk_max_tokens

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chunk(self, parse_result: ParseResult, image_result: Any | None = None) -> ChunkResult:
        """Chunk a parsed document.

        Args:
            parse_result: Stage 1 output (typed elements in reading order).
            image_result: Optional Stage 2 :class:`ImageProcessResult`. When
                present, figure descriptions/skips come from its
                ``processed_figures``; otherwise figure captions are used.

        Returns:
            A :class:`ChunkResult`. Never raises on empty input — returns zero
            chunks instead.
        """
        figure_overrides, skipped_elements = self._index_figures(image_result)
        ref_to_type = {
            el.self_ref: el.element_type for el in parse_result.elements if el.self_ref
        }

        chunks: list[Chunk] = []
        position = 0
        current_section = ""
        # Pending heading+paragraph group being accumulated.
        pending_parts: list[str] = []
        pending_page: int | None = None
        pending_is_heading_only = False
        pending_element_id: str | None = None

        def flush_group() -> None:
            nonlocal position, pending_parts, pending_page
            nonlocal pending_is_heading_only, pending_element_id
            if not pending_parts:
                return
            text = "\n".join(p for p in pending_parts if p).strip()
            if text:
                el_type = "heading" if pending_is_heading_only else "paragraph"
                for piece in self._split_to_token_cap(text):
                    chunks.append(
                        self._make_chunk(
                            parse_result,
                            position,
                            piece,
                            element_type=el_type,
                            page_number=pending_page or 1,
                            section_title=current_section,
                            source_element_id=pending_element_id,
                        )
                    )
                    position += 1
            pending_parts = []
            pending_page = None
            pending_is_heading_only = False
            pending_element_id = None

        for el in parse_result.elements:
            etype = el.element_type

            if etype == "heading":
                flush_group()
                current_section = el.text.strip()
                pending_parts = [el.text]
                pending_page = el.page_number
                pending_is_heading_only = True
                pending_element_id = el.element_id
                continue

            if etype == "paragraph":
                if pending_parts:
                    pending_parts.append(el.text)
                    pending_is_heading_only = False
                else:
                    pending_parts = [el.text]
                    pending_page = el.page_number
                    pending_element_id = el.element_id
                continue

            # Any non-paragraph element ends the current text group.
            flush_group()

            if etype == "table":
                chunks.append(self._make_table_chunk(parse_result, position, el, current_section))
                position += 1
            elif etype == "figure":
                if el.element_id in skipped_elements:
                    continue  # Tier 3 decorative figure dropped upstream.
                chunks.append(
                    self._make_figure_chunk(
                        parse_result, position, el, current_section, figure_overrides
                    )
                )
                position += 1
            elif etype == "caption":
                # Captions of tables are folded into the table chunk; keep the rest.
                if el.parent_ref and ref_to_type.get(el.parent_ref) == "table":
                    continue
                chunks.append(
                    self._make_chunk(
                        parse_result, position, el.text, element_type="caption",
                        page_number=el.page_number, section_title=current_section,
                        source_element_id=el.element_id,
                    )
                )
                position += 1
            elif etype == "footnote":
                chunks.append(
                    self._make_chunk(
                        parse_result, position, el.text, element_type="footnote",
                        page_number=el.page_number, section_title=current_section,
                        source_element_id=el.element_id,
                    )
                )
                position += 1

        flush_group()

        result = ChunkResult(chunks=chunks)
        logger.info(
            "Chunked '%s' into %d chunks (max_tokens=%d)",
            parse_result.doc_name,
            result.chunk_count,
            self.max_tokens,
        )
        return result

    # ------------------------------------------------------------------ #
    # Chunk builders
    # ------------------------------------------------------------------ #
    def _make_chunk(
        self,
        parse_result: ParseResult,
        position: int,
        text: str,
        *,
        element_type: str,
        page_number: int,
        section_title: str,
        source_element_id: str | None,
        table_json: dict[str, Any] | None = None,
        image_path: str | None = None,
    ) -> Chunk:
        """Construct a :class:`Chunk` with a positional id and full provenance."""
        return Chunk(
            chunk_id=f"{parse_result.doc_id}_chunk_{position:04d}",
            text=text.strip(),
            table_json=table_json,
            element_type=element_type,
            page_number=page_number,
            document_id=parse_result.doc_id,
            document_name=parse_result.doc_name,
            section_title=section_title,
            position_in_doc=position,
            source_element_id=source_element_id,
            image_path=image_path,
        )

    def _make_table_chunk(
        self, parse_result: ParseResult, position: int, el: ParsedElement, section: str
    ) -> Chunk:
        """A table + its caption as one chunk (text + structured ``table_json``)."""
        text = el.text or ""
        if el.caption:
            text = f"{el.caption}\n{text}".strip()
        return self._make_chunk(
            parse_result, position, text, element_type="table",
            page_number=el.page_number, section_title=section,
            source_element_id=el.element_id, table_json=el.table_json,
        )

    def _make_figure_chunk(
        self,
        parse_result: ParseResult,
        position: int,
        el: ParsedElement,
        section: str,
        figure_overrides: dict[str, tuple[str, str | None]],
    ) -> Chunk:
        """A figure chunk whose text is the Stage 2 description or its caption."""
        override = figure_overrides.get(el.element_id)
        if override is not None:
            text, image_path = override
        else:
            text, image_path = (el.caption or el.text or ""), el.image_path
        return self._make_chunk(
            parse_result, position, text, element_type="figure",
            page_number=el.page_number, section_title=section,
            source_element_id=el.element_id, image_path=image_path or el.image_path,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _split_to_token_cap(self, text: str) -> list[str]:
        """Split text into pieces under the token cap, only at sentence breaks.

        A single sentence longer than the cap is emitted whole rather than broken
        mid-sentence (the rule forbids mid-sentence splits).
        """
        if _estimate_tokens(text) <= self.max_tokens:
            return [text]
        sentences = _SENTENCE_RE.split(text)
        pieces: list[str] = []
        buf: list[str] = []
        buf_tokens = 0
        for sentence in sentences:
            stoks = _estimate_tokens(sentence)
            if buf and buf_tokens + stoks > self.max_tokens:
                pieces.append(" ".join(buf).strip())
                buf, buf_tokens = [], 0
            buf.append(sentence)
            buf_tokens += stoks
        if buf:
            pieces.append(" ".join(buf).strip())
        return [p for p in pieces if p]

    @staticmethod
    def _index_figures(
        image_result: Any | None,
    ) -> tuple[dict[str, tuple[str, str | None]], set[str]]:
        """Build figure description overrides and the set of skipped figure ids.

        Reads duck-typed ``image_result.processed_figures`` where each item
        exposes ``element_id``, ``text``, and ``image_path``; and an optional
        ``skipped_element_ids`` collection. Tolerates ``None`` (Stage 2 not run).
        """
        overrides: dict[str, tuple[str, str | None]] = {}
        skipped: set[str] = set()
        if image_result is None:
            return overrides, skipped
        for fig in getattr(image_result, "processed_figures", []) or []:
            element_id = getattr(fig, "element_id", None)
            if element_id is None:
                continue
            overrides[element_id] = (
                getattr(fig, "text", "") or "",
                getattr(fig, "image_path", None),
            )
        skipped = set(getattr(image_result, "skipped_element_ids", []) or [])
        return overrides, skipped


if __name__ == "__main__":
    import sys
    from collections import Counter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.chunker <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.ingestion.parser import DocumentParser

    pr = DocumentParser().parse(sys.argv[1])
    cr = Chunker().chunk(pr)
    print(f"\n{cr.chunk_count} chunks from '{pr.doc_name}':")
    print("element types:", dict(Counter(c.element_type for c in cr.chunks)))
    for c in cr.chunks:
        snippet = c.text.replace("\n", " ")[:64]
        print(f"  [{c.position_in_doc:02d}] {c.element_type:9s} p{c.page_number} "
              f"sec='{c.section_title[:24]}' tbl={'Y' if c.table_json else '-'} :: {snippet}")
