"""Standalone chunking utilities for retrieval-focused document indexing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


Metadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Chunk:
    """Small, dependency-free chunk object compatible with vector stores."""

    chunk_id: str
    content: str
    source: str = ""
    content_type: str = "text"
    title: str = ""
    sequence: int = 0
    metadata: Metadata = field(default_factory=dict)


class Chunker(Protocol):
    """Protocol implemented by all chunkers."""

    def chunk(self, document: Any, *, source: str = "", metadata: Mapping[str, Any] | None = None) -> list[Chunk]:
        """Split a document-like object into retrieval chunks."""


def stable_chunk_id(source: str, sequence: int, content: str) -> str:
    digest = hashlib.sha1(f"{source}:{sequence}:{content}".encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


class FixedWindowChunker:
    """Simple character-count splitter with optional overlapping context."""

    def __init__(self, chunk_size: int = 1600, overlap: int = 200) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Any, *, source: str = "", metadata: Mapping[str, Any] | None = None) -> list[Chunk]:
        text = normalize_whitespace(str(document or ""))
        if not text:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        base_metadata = dict(metadata or {})

        start = 0
        sequence = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            window = text[start:end].strip()
            if window:
                sequence += 1
                chunks.append(
                    Chunk(
                        chunk_id=stable_chunk_id(source or "fixed-window", sequence, window),
                        source=source,
                        content=window,
                        sequence=sequence,
                        metadata={**base_metadata, "start_char": start, "end_char": end},
                    )
                )
            if end >= len(text):
                break
            start += step
        return chunks


@dataclass(frozen=True, slots=True)
class StructuralElement:
    """Adapter-friendly representation of a Docling structural element."""

    text: str
    label: str = "text"
    level: int | None = None
    metadata: Metadata = field(default_factory=dict)


class StructureAwareChunker:
    """Chunk Docling-like structural elements without breaking markdown tables."""

    _HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    _TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
    _TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

    def __init__(
        self,
        max_chars: int = 1800,
        min_chars: int = 300,
        include_previous_paragraph: bool = True,
    ) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive.")
        if min_chars < 0:
            raise ValueError("min_chars must be non-negative.")
        self.max_chars = max_chars
        self.min_chars = min(min_chars, max_chars)
        self.include_previous_paragraph = include_previous_paragraph

    def chunk(
        self,
        document: Any,
        *,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        elements = list(self._coerce_elements(document))
        if not elements:
            return []

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_meta: list[Metadata] = []
        headers: dict[int, str] = {}
        previous_paragraph = ""
        sequence = 0
        base_metadata = dict(metadata or {})

        def current_header() -> str:
            if not headers:
                return ""
            return headers[max(headers)]

        def emit_buffer() -> None:
            nonlocal buffer, buffer_meta, sequence
            content = "\n\n".join(part for part in buffer if part.strip()).strip()
            if not content:
                buffer = []
                buffer_meta = []
                return
            sequence += 1
            title = current_header()
            merged_meta = self._merge_metadata(buffer_meta)
            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(source or "structured", sequence, content),
                    source=source,
                    content=content,
                    content_type="text",
                    title=title,
                    sequence=sequence,
                    metadata={**base_metadata, **merged_meta, "has_table": False},
                )
            )
            buffer = []
            buffer_meta = []

        for element in elements:
            text = normalize_whitespace(element.text)
            if not text:
                continue

            header_level, header_text = self._header_info(element)
            if header_text:
                emit_buffer()
                headers = {level: value for level, value in headers.items() if level < header_level}
                headers[header_level] = header_text
                previous_paragraph = ""
                continue

            if self._is_table(element):
                emit_buffer()
                table_parts = []
                header = current_header()
                if header:
                    table_parts.append(f"## {header}")
                if self.include_previous_paragraph and previous_paragraph:
                    table_parts.append(previous_paragraph)
                table_parts.append(text)
                table_content = "\n\n".join(table_parts).strip()
                sequence += 1
                chunks.append(
                    Chunk(
                        chunk_id=stable_chunk_id(source or "structured", sequence, table_content),
                        source=source,
                        content=table_content,
                        content_type="table",
                        title=header,
                        sequence=sequence,
                        metadata={
                            **base_metadata,
                            **element.metadata,
                            "has_table": True,
                            "table_atomic": True,
                            "preceding_paragraph_attached": bool(previous_paragraph),
                        },
                    )
                )
                previous_paragraph = ""
                continue

            candidate = "\n\n".join([*buffer, text]).strip()
            if buffer and len(candidate) > self.max_chars and len("\n\n".join(buffer)) >= self.min_chars:
                emit_buffer()
            buffer.append(text)
            buffer_meta.append(element.metadata)
            previous_paragraph = text if self._is_paragraph(element) else previous_paragraph

        emit_buffer()
        return chunks

    def _coerce_elements(self, document: Any) -> Iterable[StructuralElement]:
        if isinstance(document, str):
            yield from self._elements_from_markdown(document)
            return
        if isinstance(document, Iterable):
            for item in document:
                element = self._coerce_element(item)
                if element is not None:
                    yield element
            return
        exported = getattr(document, "export_to_markdown", None)
        if callable(exported):
            yield from self._elements_from_markdown(str(exported()))

    def _coerce_element(self, item: Any) -> StructuralElement | None:
        if isinstance(item, StructuralElement):
            return item
        if isinstance(item, str):
            return StructuralElement(text=item)
        if isinstance(item, Mapping):
            text = str(item.get("text") or item.get("content") or item.get("markdown") or "")
            label = str(item.get("label") or item.get("type") or item.get("content_type") or "text")
            level = item.get("level")
            return StructuralElement(text=text, label=label, level=int(level) if isinstance(level, int) else None)

        text_value = getattr(item, "text", None) or getattr(item, "content", None)
        if text_value is None and hasattr(item, "export_to_markdown"):
            text_value = item.export_to_markdown()
        if text_value is None:
            return None
        label = str(getattr(item, "label", "") or getattr(item, "type", "") or item.__class__.__name__)
        level = getattr(item, "level", None)
        metadata = getattr(item, "metadata", {}) if isinstance(getattr(item, "metadata", {}), Mapping) else {}
        return StructuralElement(text=str(text_value), label=label, level=level if isinstance(level, int) else None, metadata=dict(metadata))

    def _elements_from_markdown(self, markdown: str) -> Iterable[StructuralElement]:
        blocks = re.split(r"\n\s*\n", markdown)
        for block in blocks:
            text = block.strip()
            if text:
                yield StructuralElement(text=text, label="table" if self._looks_like_table(text) else "text")

    def _header_info(self, element: StructuralElement) -> tuple[int, str]:
        if element.level is not None and "header" in element.label.lower():
            return max(1, min(6, element.level)), normalize_whitespace(element.text).lstrip("#").strip()
        match = self._HEADER_RE.match(element.text.strip())
        if match:
            return len(match.group(1)), match.group(2).strip()
        return 0, ""

    def _is_table(self, element: StructuralElement) -> bool:
        return "table" in element.label.lower() or self._looks_like_table(element.text)

    def _looks_like_table(self, text: str) -> bool:
        lines = [line for line in text.splitlines() if line.strip()]
        table_lines = [line for line in lines if self._TABLE_LINE_RE.match(line)]
        return len(table_lines) >= 2 and any(self._TABLE_SEPARATOR_RE.match(line) for line in lines)

    def _is_paragraph(self, element: StructuralElement) -> bool:
        label = element.label.lower()
        return "paragraph" in label or (not self._is_table(element) and not self._header_info(element)[1])

    def _merge_metadata(self, items: Sequence[Metadata]) -> Metadata:
        merged: Metadata = {}
        for item in items:
            for key, value in item.items():
                if key not in merged:
                    merged[key] = value
        return merged
