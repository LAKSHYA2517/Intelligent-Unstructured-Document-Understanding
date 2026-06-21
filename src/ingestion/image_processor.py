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

import base64
import io
import logging
import os
import re
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from src.config import config
from src.ingestion.parser import ParsedElement, ParseResult

logger = logging.getLogger(__name__)

# Image size limits for safe API payloads
MAX_IMAGE_BYTES_FOR_API = 100_000   # 100KB hard limit
TARGET_IMAGE_BYTES      = 50_000    # 50KB target (extremely safe)
MAX_DIMENSION           = 300       # Further reduced to guarantee small payloads

import hashlib

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
    """
    Responsible for:
    1. Compressing images to a size that won't cause TCP connection drops
    2. Caching captions so we never re-caption the same image
    3. Skipping images that are clearly not worth captioning

    This implementation uses Gemini 2.0 Flash multimodal (via google.generativeai)
    to generate figure captions asynchronously instead of a local Ollama model.
    The external interface (process returns ImageProcessResult) remains sync for
    compatibility with the ingestion pipeline.
    """

    def __init__(self, cache_dir: str = ".image_caption_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # We rely on the unified llm_router for vision now.
        self._has_vision = bool(os.getenv("ROUTER_VISION_MODELS"))
        if self._has_vision:
            logger.info("ImageProcessor ready using cloud vision models")
        else:
            logger.warning("ImageProcessor: ROUTER_VISION_MODELS not set — image vision disabled")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, parse_result: ParseResult) -> ImageProcessResult:
        """Describe every figure in a document using the cheapest viable tier.

        Synchronous wrapper that keeps the previous public API. Internally runs
        the async Gemini captioning routine when needed.
        """
        figures = [e for e in parse_result.elements if e.element_type == "figure"]
        processed: list[FigureChunk] = []
        vision_calls = 0
        skipped_ids: list[str] = []

        # Decide tiers first so we can run all vision calls in one async batch.
        to_caption: list[ParsedElement] = []
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
            # Tier 2 — needs vision caption
            to_caption.append(el)

        # If we have a vision model configured, run async captioning in one batch
        captions: dict[str, str | None] = {}
        if to_caption and self._has_vision:
            try:
                captions = self._run_async(self._describe_images_async(to_caption))
            except Exception as exc:
                logger.warning("Cloud image captioning failed: %s", exc)
                captions = {el.element_id: None for el in to_caption}

        # Build final processed list
        for el in to_caption:
            description = captions.get(el.element_id)
            vision_calls += 1 if description is not None else 0
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
    # Async Gemini multimodal vision (batch)
    # ------------------------------------------------------------------ #
    def _run_async(self, coro):
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

    def prepare_image_for_api(self, image_bytes: bytes, aggressive: bool = False) -> Optional[str]:
        """
        Compress an image until it is under the size limit.
        Returns base64 string, or None if image is too small to be worth sending.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB (removes alpha channel, reduces size)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Check if image is large enough to be meaningful
            w, h = img.size
            if w < 50 or h < 50:
                logger.debug(f"Skipping tiny image ({w}x{h})")
                return None

            max_dim = MAX_DIMENSION if not aggressive else MAX_DIMENSION // 2

            # Try progressively lower quality until under size limit
            quality_levels = [70, 50, 35, 20] if not aggressive else [30, 15]

            for quality in quality_levels:

                # Resize if needed
                scale = min(max_dim / w, max_dim / h, 1.0)
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                else:
                    resized = img

                # Compress to JPEG
                buffer = io.BytesIO()
                resized.save(buffer, format="JPEG", quality=quality, optimize=True)
                compressed_bytes = buffer.getvalue()

                if len(compressed_bytes) <= TARGET_IMAGE_BYTES:
                    b64 = base64.b64encode(compressed_bytes).decode("utf-8")
                    logger.debug(
                        f"Image compressed: {len(image_bytes):,} → "
                        f"{len(compressed_bytes):,} bytes "
                        f"(quality={quality}, scale={scale:.2f})"
                    )
                    return b64

                # Make image smaller for next attempt
                max_dim = int(max_dim * 0.75)
                w, h = resized.size

            # If still over limit, force to thumbnail
            img.thumbnail((128, 128), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=15)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            logger.warning(
                "Image required extreme compression to fit under size limit"
            )
            return b64

        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            return None

    def get_cache_key(self, image_bytes: bytes) -> str:
        """SHA256 hash of image bytes as cache key."""
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def get_cached_caption(self, image_bytes: bytes) -> Optional[str]:
        """Return cached caption if available."""
        key = self.get_cache_key(image_bytes)
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")
        return None

    def save_caption_cache(self, image_bytes: bytes, caption: str) -> None:
        """Save caption to disk cache."""
        key = self.get_cache_key(image_bytes)
        cache_file = self.cache_dir / f"{key}.txt"
        cache_file.write_text(caption, encoding="utf-8")

    def should_caption(self, image_bytes: bytes) -> bool:
        """
        Decide if an image is worth sending to the vision API.
        Saves vision API quota.
        """
        if len(image_bytes) < 5_000:   # < 5KB
            return False

        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 20:  # Very elongated
                return False
            if w < 80 or h < 80:
                return False
        except Exception:
            pass

        return True

    async def _describe_images_async(self, elements: list[ParsedElement]) -> dict[str, str | None]:
        """Asynchronously caption multiple image elements using Gemini (best-effort).

        Returns a mapping element_id -> caption (or None on failure).
        """
        results: dict[str, str | None] = {}
        tasks = []
        semaphore = asyncio.Semaphore(5)

        async def _task(el: ParsedElement) -> None:
            nonlocal results
            if not el.image_path or not Path(el.image_path).exists():
                results[el.element_id] = None
                return
            image_bytes = Path(el.image_path).read_bytes()
            # cache
            cached = self.get_cached_caption(image_bytes)
            if cached is not None:
                results[el.element_id] = cached
                return
            # cheap heuristics
            if not self.should_caption(image_bytes):
                results[el.element_id] = None
                return

            b64 = self.prepare_image_for_api(image_bytes)
            if b64 is None:
                results[el.element_id] = None
                return

            async with semaphore:
                caption = await self._describe_single_image_async(b64)
                if caption:
                    try:
                        self.save_caption_cache(image_bytes, caption)
                    except Exception:
                        logger.debug("Failed to save caption cache for %s", el.image_path)
                results[el.element_id] = caption

        for el in elements:
            tasks.append(asyncio.create_task(_task(el)))

        await asyncio.gather(*tasks)
        return results

    async def _describe_single_image_async(self, b64_image: str) -> str | None:
        """Multimodal caption using the central LiteLLM router with rate limiting and retries."""
        if not self._has_vision:
            logger.warning("Vision models not configured - skipping image caption")
            return None

        from src.llm_router import agenerate
        
        try:
            # LiteLLM router handles rate-limiting and retries automatically
            response_text = await agenerate(
                _VISION_PROMPT,
                model="router-vision",
                image_b64=b64_image,
                temperature=0.1
            )
            return response_text.strip()
        except Exception as e:
            logger.error("Vision API error via router: %s", e)
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
