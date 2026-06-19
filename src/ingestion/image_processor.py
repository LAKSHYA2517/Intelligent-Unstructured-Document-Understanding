"""Stage 2 — tiered figure image processing.

For every figure recovered in Stage 1 this module decides, per figure, the
cheapest way to obtain a useful textual description:

  * **Tier 3 (skip):** decorative/logo images — tiny file size or banner-like
    aspect ratio — are dropped entirely (no chunk, no vision call).
  * **Tier 1 (caption):** a figure with a descriptive caption (> 5 meaningful
    words) reuses that caption text — no vision call.
  * **Tier 2 (vision):** a figure with no caption or a vague one (≤ 5 words) is
    described by the configured vision model via Ollama.

The vision model name is **always** read from ``config.OLLAMA_VISION_MODEL`` —
never hardcoded here — keeping the stage model-agnostic. Every count returned
(``vision_calls``, ``skipped``) is a real runtime value the Streamlit UI reads.

The decorative (Tier 3) check runs first so logos never waste a vision call;
this orders the tiers by cost while preserving their intent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import config
from src.ingestion.parser import ParsedElement, ParseResult

logger = logging.getLogger(__name__)

# The exact vision prompt mandated by the spec.
_VISION_PROMPT = (
    "Describe this chart, diagram, or figure in detail. Include: the type of "
    "visual (bar chart/pie chart/flowchart/etc), what the axes or segments "
    "represent, key values visible, and the main insight conveyed."
)

# Tier-1 caption quality bar: strictly more than this many meaningful words.
_MIN_CAPTION_WORDS = 5
# Tier-3 decorative heuristics.
_DECORATIVE_MAX_BYTES = 5 * 1024  # < 5 KB → likely a logo/icon
_DECORATIVE_MAX_ASPECT = 5.0  # very wide/tall banner
_DECORATIVE_MIN_DIM = 32  # px; smaller than this is an icon
# Words ignored when judging caption descriptiveness.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "by", "with",
    "is", "are", "this", "that", "figure", "fig", "table", "chart", "image",
}


@dataclass
class FigureChunk:
    """A processed figure ready to become a chunk and a graph ``:Figure`` node.

    Attributes:
        element_id: Originating :class:`ParsedElement` id (links to the chunker).
        text: Description text (caption for Tier 1, vision output for Tier 2).
        image_path: Saved crop path on disk.
        page_number: Real page number of the figure.
        tier: 1 (caption) or 2 (vision).
        extraction_method: ``"caption"`` or ``"vision"`` for provenance.
        caption: Original caption text, if any.
    """

    element_id: str
    text: str
    image_path: str | None
    page_number: int
    tier: int
    extraction_method: str
    caption: str | None = None


@dataclass
class ImageProcessResult:
    """Stage 2 output. ``vision_calls`` and ``skipped`` are read by the UI."""

    processed_figures: list[FigureChunk]
    vision_calls: int = 0
    skipped: int = 0
    skipped_element_ids: list[str] = field(default_factory=list)


class ImageProcessor:
    """Runs the tiered figure-description policy over a :class:`ParseResult`."""

    def __init__(self) -> None:
        # Imported lazily-ish at construction; the client is cheap and makes no
        # network call until a Tier-2 figure actually needs describing.
        import ollama

        self._client = ollama.Client(host=config.ollama_base_url)
        self._vision_model = config.ollama_vision_model
        logger.info("ImageProcessor ready (vision model from config: %s)", self._vision_model)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, parse_result: ParseResult) -> ImageProcessResult:
        """Describe every figure in a document using the cheapest viable tier.

        Args:
            parse_result: Stage 1 output.

        Returns:
            An :class:`ImageProcessResult` whose counts are all real runtime values.
        """
        figures = [e for e in parse_result.elements if e.element_type == "figure"]
        processed: list[FigureChunk] = []
        vision_calls = 0
        skipped_ids: list[str] = []

        for el in figures:
            tier = self._classify_tier(el)
            if tier == 3:
                skipped_ids.append(el.element_id)
                logger.info("Figure %s skipped (Tier 3 decorative)", el.element_id)
                continue
            if tier == 1:
                processed.append(
                    FigureChunk(
                        element_id=el.element_id,
                        text=(el.caption or "").strip(),
                        image_path=el.image_path,
                        page_number=el.page_number,
                        tier=1,
                        extraction_method="caption",
                        caption=el.caption,
                    )
                )
                continue
            # Tier 2 — vision description.
            description = self._describe_image(el.image_path)
            vision_calls += 1
            text = description or (el.caption or "").strip() or "Figure (no description available)."
            processed.append(
                FigureChunk(
                    element_id=el.element_id,
                    text=text,
                    image_path=el.image_path,
                    page_number=el.page_number,
                    tier=2,
                    extraction_method="vision" if description else "vision_failed",
                    caption=el.caption,
                )
            )

        result = ImageProcessResult(
            processed_figures=processed,
            vision_calls=vision_calls,
            skipped=len(skipped_ids),
            skipped_element_ids=skipped_ids,
        )
        logger.info(
            "Processed %d figures in '%s': %d via caption, %d vision calls, %d skipped",
            len(figures),
            parse_result.doc_name,
            sum(1 for f in processed if f.tier == 1),
            result.vision_calls,
            result.skipped,
        )
        return result

    # ------------------------------------------------------------------ #
    # Tier classification
    # ------------------------------------------------------------------ #
    def _classify_tier(self, el: ParsedElement) -> int:
        """Return 1 (caption), 2 (vision), or 3 (skip) for a figure element."""
        # Tier 3 first so decorative images never trigger a vision call.
        if self._is_decorative(el.image_path):
            return 3
        if self._meaningful_word_count(el.caption) > _MIN_CAPTION_WORDS:
            return 1
        # No usable image to describe and no good caption → nothing we can do.
        if not el.image_path or not Path(el.image_path).exists():
            return 3
        return 2

    @staticmethod
    def _meaningful_word_count(caption: str | None) -> int:
        """Count descriptive words in a caption, ignoring stopwords/numbers."""
        if not caption:
            return 0
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", caption.lower())
        return sum(1 for w in words if w not in _STOPWORDS and len(w) > 1)

    @staticmethod
    def _is_decorative(image_path: str | None) -> bool:
        """Heuristically decide whether an image is a decorative logo/banner."""
        if not image_path:
            return False  # no crop to judge; caption/vision logic handles it
        path = Path(image_path)
        if not path.exists():
            return False
        try:
            if path.stat().st_size < _DECORATIVE_MAX_BYTES:
                return True
            from PIL import Image

            with Image.open(path) as img:
                w, h = img.size
            if min(w, h) < _DECORATIVE_MIN_DIM:
                return True
            if min(w, h) and max(w, h) / min(w, h) > _DECORATIVE_MAX_ASPECT:
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not inspect image %s: %s", image_path, exc)
            return False
        return False

    # ------------------------------------------------------------------ #
    # Vision call
    # ------------------------------------------------------------------ #
    def _describe_image(self, image_path: str | None) -> str | None:
        """Call the configured Ollama vision model on an image crop.

        Returns the description text, or ``None`` if the image is unusable or the
        call fails (the caller falls back to caption text).
        """
        if not image_path or not Path(image_path).exists():
            return None
        try:
            image_bytes = Path(image_path).read_bytes()
            response = self._client.generate(
                model=self._vision_model,
                prompt=_VISION_PROMPT,
                images=[image_bytes],
            )
            text = (getattr(response, "response", None) or "").strip()
            return text or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vision call failed for %s: %s", image_path, exc)
            return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.image_processor <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.ingestion.parser import DocumentParser

    pr = DocumentParser().parse(sys.argv[1])
    res = ImageProcessor().process(pr)
    print(f"\nvision_calls={res.vision_calls} skipped={res.skipped}")
    for f in res.processed_figures:
        print(f"  [{f.element_id}] tier={f.tier} method={f.extraction_method} "
              f"p{f.page_number}: {f.text[:120]!r}")
