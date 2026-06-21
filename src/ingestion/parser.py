"""Stage 1 — document parsing with Docling.

Converts a PDF or DOCX file into a :class:`ParseResult`: a flat, reading-order
list of typed :class:`ParsedElement` objects (heading, paragraph, table, figure,
caption, footnote) plus real document statistics recovered from Docling
(page/table/figure counts, reading order). Tables keep their full row/column
structure (TableFormer) and are never flattened here. Figure crops are saved to
``data/figures/{doc_id}_{fig_id}.png``. Multi-page tables whose column headers
continue across pages are merged into a single logical table element.

Nothing in this module is domain- or model-specific; it is pure structural
extraction. All downstream stages consume :class:`ParseResult`.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from docling_core.types.doc.labels import DocItemLabel

from src.config import config

logger = logging.getLogger(__name__)

# Map Docling's fine-grained labels onto the six element types the pipeline uses.
_LABEL_TO_ELEMENT_TYPE: dict[DocItemLabel, str] = {
    DocItemLabel.TITLE: "heading",
    DocItemLabel.SECTION_HEADER: "heading",
    DocItemLabel.TEXT: "paragraph",
    DocItemLabel.PARAGRAPH: "paragraph",
    DocItemLabel.LIST_ITEM: "paragraph",
    DocItemLabel.CODE: "paragraph",
    DocItemLabel.FORMULA: "paragraph",
    DocItemLabel.TABLE: "table",
    DocItemLabel.PICTURE: "figure",
    DocItemLabel.CHART: "figure",
    DocItemLabel.CAPTION: "caption",
    DocItemLabel.FOOTNOTE: "footnote",
    DocItemLabel.REFERENCE: "footnote",
}

# Labels treated as page furniture and dropped (not document content).
_SKIP_LABELS: set[DocItemLabel] = {
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
}

_SUPPORTED_SUFFIXES = {".pdf", ".docx"}


class ParserError(RuntimeError):
    """Raised when a document cannot be parsed."""


@dataclass
class ParsedElement:
    """One typed structural element recovered from a document.

    Attributes:
        element_id: Stable id, unique within the document.
        element_type: One of ``heading|paragraph|table|figure|caption|footnote``.
        text: Plain-text content (markdown for tables, caption text for figures).
        page_number: 1-based page the element starts on (real Docling value).
        reading_order: 0-based position in Docling's recovered reading order.
        level: Heading hierarchy level (1=H1 …) for headings, else ``None``.
        bbox: ``(left, top, right, bottom)`` on its page, if known.
        table_json: Structured table (headers/rows/grid) for tables, else ``None``.
        image_path: Saved crop path for figures, else ``None``.
        caption: Caption text linked to a figure/table, else ``None``.
        parent_ref: Docling ``self_ref`` of the figure/table a caption belongs to.
        self_ref: Docling ``self_ref`` of this element (for cross-linking).
    """

    element_id: str
    element_type: str
    text: str
    page_number: int
    reading_order: int
    level: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    table_json: dict[str, Any] | None = None
    image_path: str | None = None
    caption: str | None = None
    parent_ref: str | None = None
    self_ref: str | None = None


@dataclass
class ParseResult:
    """Result of parsing one document (Stage 1 output).

    Every count is a real value taken from the Docling document, never hardcoded;
    the Streamlit UI reads these fields directly for its progress display.
    """

    doc_id: str
    doc_name: str
    elements: list[ParsedElement]
    page_count: int
    table_count: int
    figure_count: int
    reading_order: list[str] = field(default_factory=list)


class DocumentParser:
    """Parses PDF/DOCX files into :class:`ParseResult` using Docling.

    The underlying :class:`~docling.document_converter.DocumentConverter` loads
    layout and TableFormer models on first use, so a single parser instance
    should be reused across a batch.
    """

    def __init__(self) -> None:
        pdf_options = PdfPipelineOptions()
        # TableFormer on so tables keep full row/col structure; OCR off for speed
        # on digital PDFs (flip do_ocr=True for scanned docs). images_scale=2 gives
        # crisp figure crops to save under data/figures/.
        pdf_options.do_table_structure = True
        pdf_options.table_structure_options.do_cell_matching = True
        pdf_options.do_ocr = False
        pdf_options.generate_picture_images = True
        pdf_options.images_scale = 2.0
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            }
        )
        logger.info("DocumentParser initialised (TableFormer on, OCR off, image crops on).")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse(self, file_path: str | Path, doc_name: str | None = None) -> ParseResult:
        """Parse one document into a :class:`ParseResult`.

        Args:
            file_path: Path to a ``.pdf`` or ``.docx`` file on disk.
            doc_name: Display name (defaults to the file's name). Always the real
                uploaded filename — never hardcoded downstream.

        Returns:
            A populated :class:`ParseResult`.

        Raises:
            ParserError: If the file is missing, unsupported, or fails to convert.
        """
        path = Path(file_path)
        if not path.exists():
            raise ParserError(f"File does not exist: {path}")
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ParserError(
                f"Unsupported file type '{path.suffix}'. Supported: {sorted(_SUPPORTED_SUFFIXES)}"
            )

        resolved_name = doc_name or path.name
        doc_id = self._make_doc_id(resolved_name)
        logger.info("Parsing '%s' (doc_id=%s)", resolved_name, doc_id)

        try:
            conversion = self._converter.convert(str(path))
        except Exception as exc:  # noqa: BLE001
            raise ParserError(f"Docling failed to convert '{path.name}': {exc}") from exc

        doc = conversion.document
        elements = self._extract_elements(doc, doc_id)
        elements = self._merge_multipage_tables(elements)

        # Re-number reading order after the merge so indices stay contiguous.
        for idx, el in enumerate(elements):
            el.reading_order = idx

        result = ParseResult(
            doc_id=doc_id,
            doc_name=resolved_name,
            elements=elements,
            page_count=len(doc.pages),
            table_count=len(doc.tables),
            figure_count=len(doc.pictures),
            reading_order=[el.element_id for el in elements],
        )
        logger.info(
            "Parsed '%s': %d pages, %d tables, %d figures, %d elements",
            resolved_name,
            result.page_count,
            result.table_count,
            result.figure_count,
            len(elements),
        )
        return result

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    def _extract_elements(self, doc: DoclingDocument, doc_id: str) -> list[ParsedElement]:
        """Walk the document in reading order and build typed elements."""
        caption_parents = self._build_caption_parent_map(doc)
        elements: list[ParsedElement] = []
        order = 0
        fig_index = 0

        for item, _level in doc.iterate_items():
            label = getattr(item, "label", None)
            if label in _SKIP_LABELS:
                continue
            element_type = _LABEL_TO_ELEMENT_TYPE.get(label)
            if element_type is None:
                # Unknown/structural label with text → treat as paragraph if it has any.
                if isinstance(item, TextItem) and item.text.strip():
                    element_type = "paragraph"
                else:
                    continue

            element_id = f"{doc_id}_el_{order:04d}"
            page_no = self._page_of(item)
            bbox = self._bbox_of(item)
            self_ref = getattr(item, "self_ref", None)

            if isinstance(item, TableItem):
                el = self._build_table_element(
                    item, doc, element_id, page_no, order, bbox, self_ref
                )
            elif isinstance(item, PictureItem):
                el = self._build_figure_element(
                    item, doc, doc_id, fig_index, element_id, page_no, order, bbox, self_ref
                )
                fig_index += 1
            else:
                text = getattr(item, "text", "") or ""
                if not text.strip() and element_type != "figure":
                    continue
                el = ParsedElement(
                    element_id=element_id,
                    element_type=element_type,
                    text=text,
                    page_number=page_no,
                    reading_order=order,
                    level=getattr(item, "level", None)
                    if isinstance(item, SectionHeaderItem)
                    else (1 if label == DocItemLabel.TITLE else None),
                    bbox=bbox,
                    self_ref=self_ref,
                    parent_ref=caption_parents.get(self_ref) if element_type == "caption" else None,
                )

            elements.append(el)
            order += 1

        return elements

    @staticmethod
    def _build_caption_parent_map(doc: DoclingDocument) -> dict[str, str]:
        """Map each caption's ``self_ref`` to the figure/table it describes."""
        mapping: dict[str, str] = {}
        for container in (*doc.tables, *doc.pictures):
            parent_ref = getattr(container, "self_ref", None)
            for cap_ref in getattr(container, "captions", []) or []:
                ref = getattr(cap_ref, "cref", None) or getattr(cap_ref, "self_ref", None)
                if ref and parent_ref:
                    mapping[ref] = parent_ref
        return mapping

    def _build_table_element(
        self,
        item: TableItem,
        doc: DoclingDocument,
        element_id: str,
        page_no: int,
        order: int,
        bbox: tuple[float, float, float, float] | None,
        self_ref: str | None,
    ) -> ParsedElement:
        """Build a table element preserving full row/column structure."""
        table_json = self._table_to_json(item)
        try:
            markdown = item.export_to_markdown(doc)
        except Exception:  # noqa: BLE001 - fall back to a plain grid join
            markdown = "\n".join(" | ".join(row) for row in table_json.get("rows", []))
        caption = (item.caption_text(doc) or None) if hasattr(item, "caption_text") else None
        return ParsedElement(
            element_id=element_id,
            element_type="table",
            text=markdown,
            page_number=page_no,
            reading_order=order,
            bbox=bbox,
            table_json=table_json,
            caption=caption,
            self_ref=self_ref,
        )

    def _build_figure_element(
        self,
        item: PictureItem,
        doc: DoclingDocument,
        doc_id: str,
        fig_index: int,
        element_id: str,
        page_no: int,
        order: int,
        bbox: tuple[float, float, float, float] | None,
        self_ref: str | None,
    ) -> ParsedElement:
        """Build a figure element and persist its image crop to disk."""
        fig_id = f"fig_{fig_index:04d}"
        image_path = self._save_figure_image(item, doc, doc_id, fig_id)
        caption = (item.caption_text(doc) or None) if hasattr(item, "caption_text") else None
        return ParsedElement(
            element_id=element_id,
            element_type="figure",
            text=caption or "",
            page_number=page_no,
            reading_order=order,
            bbox=bbox,
            image_path=image_path,
            caption=caption,
            self_ref=self_ref,
        )

    @staticmethod
    def _table_to_json(item: TableItem) -> dict[str, Any]:
        """Convert a TableFormer table into a structured dict (headers/rows/grid)."""
        data = item.data
        num_rows = int(getattr(data, "num_rows", 0) or 0)
        num_cols = int(getattr(data, "num_cols", 0) or 0)
        grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        header_rows: set[int] = set()

        for cell in getattr(data, "table_cells", []) or []:
            text = (cell.text or "").strip()
            r0 = cell.start_row_offset_idx
            r1 = cell.end_row_offset_idx
            c0 = cell.start_col_offset_idx
            c1 = cell.end_col_offset_idx
            for r in range(r0, min(r1, num_rows)):
                for c in range(c0, min(c1, num_cols)):
                    grid[r][c] = text
            if getattr(cell, "column_header", False):
                header_rows.add(r0)

        header_idx = min(header_rows) if header_rows else 0
        headers = grid[header_idx] if grid else []
        body = [grid[r] for r in range(len(grid)) if r != header_idx]
        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "headers": headers,
            "rows": body,
            "grid": grid,
        }

    def _save_figure_image(
        self, item: PictureItem, doc: DoclingDocument, doc_id: str, fig_id: str
    ) -> str | None:
        """Save a figure's image crop to ``data/figures/{doc_id}_{fig_id}.png``."""
        try:
            image = item.get_image(doc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("No image for %s_%s: %s", doc_id, fig_id, exc)
            return None
        if image is None:
            return None
        out_path = config.figures_dir / f"{doc_id}_{fig_id}.png"
        try:
            image.save(out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save figure crop %s: %s", out_path, exc)
            return None
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Multi-page table merging
    # ------------------------------------------------------------------ #
    def _merge_multipage_tables(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        """Merge consecutive tables whose column headers continue across pages.

        Two table elements merge when they share the same header signature and
        appear consecutively in reading order (only captions/footnotes may sit
        between them). Any heading or paragraph breaks the continuation.
        """
        merged: list[ParsedElement] = []
        last_table: ParsedElement | None = None

        for el in elements:
            if el.element_type == "table":
                sig = self._header_signature(el)
                if (
                    last_table is not None
                    and sig
                    and sig == self._header_signature(last_table)
                    and el.page_number != last_table.page_number
                ):
                    self._append_table_rows(last_table, el)
                    logger.debug(
                        "Merged multi-page table %s into %s",
                        el.element_id,
                        last_table.element_id,
                    )
                    continue
                merged.append(el)
                last_table = el
            else:
                # Captions/footnotes attached to the table don't break continuation.
                if el.element_type not in ("caption", "footnote"):
                    last_table = None
                merged.append(el)

        return merged

    @staticmethod
    def _header_signature(table_el: ParsedElement) -> tuple[str, ...]:
        """Normalised tuple of a table's header cell texts, for equality checks."""
        if not table_el.table_json:
            return ()
        return tuple(
            re.sub(r"\s+", " ", (h or "").strip().lower())
            for h in table_el.table_json.get("headers", [])
        )

    @staticmethod
    def _append_table_rows(target: ParsedElement, source: ParsedElement) -> None:
        """Append ``source``'s body rows onto ``target`` (in place)."""
        if not target.table_json or not source.table_json:
            return
        target.table_json["rows"].extend(source.table_json.get("rows", []))
        target.table_json["grid"].extend(source.table_json.get("rows", []))
        target.table_json["num_rows"] = len(target.table_json["rows"]) + 1
        if source.text:
            target.text = f"{target.text}\n{source.text}".strip()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_doc_id(name: str) -> str:
        """Generate a unique, filesystem-safe document id from a filename."""
        stem = Path(name).stem
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower() or "doc"
        return f"{slug[:40]}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _page_of(item: Any) -> int:
        """Return the 1-based page number of an item, defaulting to 1."""
        prov = getattr(item, "prov", None)
        if prov:
            return int(prov[0].page_no)
        return 1

    @staticmethod
    def _bbox_of(item: Any) -> tuple[float, float, float, float] | None:
        """Return ``(left, top, right, bottom)`` for an item, if available."""
        prov = getattr(item, "prov", None)
        if not prov:
            return None
        bbox = getattr(prov[0], "bbox", None)
        if bbox is None:
            return None
        return (float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.parser <file.pdf|file.docx>")
        raise SystemExit(1)
    parser = DocumentParser()
    res = parser.parse(sys.argv[1])
    print(f"\ndoc_id={res.doc_id} name={res.doc_name}")
    print(f"pages={res.page_count} tables={res.table_count} figures={res.figure_count}")
    from collections import Counter

    counts = Counter(e.element_type for e in res.elements)
    print("element types:", dict(counts))
    for e in res.elements[:12]:
        snippet = e.text.replace("\n", " ")[:70]
        print(f"  [{e.reading_order:02d}] {e.element_type:9s} p{e.page_number} "
              f"lvl={e.level} img={'Y' if e.image_path else '-'} :: {snippet}")
