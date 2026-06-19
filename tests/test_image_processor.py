"""Tests for Stage 2 (tiered image processing) — src/ingestion/image_processor.py.

Mixes one real vision call against the configured Ollama model with network-free
unit tests for tier classification, decorative detection, caption scoring, and
the vision-failure fallback (via monkeypatching the vision method).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.ingestion.image_processor import FigureChunk, ImageProcessor
from src.ingestion.parser import DocumentParser, ParsedElement, ParseResult


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _save_png(path: Path, size: tuple[int, int], *, noisy: bool = False) -> Path:
    """Write a PNG of the given size; ``noisy`` keeps it above the 5 KB threshold."""
    from PIL import Image

    if noisy:
        img = Image.effect_noise(size, 48).convert("RGB")
    else:
        img = Image.new("RGB", size, "white")
    img.save(path)
    return path


def _fig(eid: str, caption: str | None, image_path: str | None, page: int = 1) -> ParsedElement:
    return ParsedElement(
        element_id=eid,
        element_type="figure",
        text=caption or "",
        page_number=page,
        reading_order=0,
        image_path=image_path,
        caption=caption,
        self_ref=f"#/{eid}",
    )


def _result(figs: list[ParsedElement]) -> ParseResult:
    return ParseResult(
        doc_id="d_0001", doc_name="doc.pdf", elements=figs,
        page_count=1, table_count=0, figure_count=len(figs),
        reading_order=[f.element_id for f in figs],
    )


@pytest.fixture(scope="module")
def processor() -> ImageProcessor:
    """A single ImageProcessor (constructs an Ollama client, no network call)."""
    return ImageProcessor()


# --------------------------------------------------------------------------- #
# Caption scoring & decorative detection (pure, no network)
# --------------------------------------------------------------------------- #
def test_meaningful_word_count(processor: ImageProcessor) -> None:
    assert processor._meaningful_word_count(None) == 0
    assert processor._meaningful_word_count("Figure 1: revenue by region") <= 5
    rich = "Bar chart comparing annual revenue growth across five global regions clearly"
    assert processor._meaningful_word_count(rich) > 5


def test_is_decorative(processor: ImageProcessor, tmp_path: Path) -> None:
    tiny = _save_png(tmp_path / "logo.png", (10, 10))  # < 5 KB
    banner = _save_png(tmp_path / "banner.png", (600, 40), noisy=True)  # aspect 15
    content = _save_png(tmp_path / "chart.png", (360, 240), noisy=True)
    assert processor._is_decorative(str(tiny)) is True
    assert processor._is_decorative(str(banner)) is True
    assert processor._is_decorative(str(content)) is False
    assert processor._is_decorative(None) is False


def test_classify_tier(processor: ImageProcessor, tmp_path: Path) -> None:
    content = str(_save_png(tmp_path / "c.png", (360, 240), noisy=True))
    tiny = str(_save_png(tmp_path / "t.png", (10, 10)))
    # Tier 1: descriptive caption, image not decorative.
    good_caption = "Bar chart comparing annual revenue growth across six regions worldwide"
    assert processor._classify_tier(_fig("a", good_caption, content)) == 1
    # Tier 2: weak caption but a real content image to describe.
    assert processor._classify_tier(_fig("b", "Figure 1", content)) == 2
    # Tier 3: decorative image.
    assert processor._classify_tier(_fig("c", None, tiny)) == 3


# --------------------------------------------------------------------------- #
# process() routing (network-free via monkeypatch)
# --------------------------------------------------------------------------- #
def test_process_routes_and_counts(processor: ImageProcessor, tmp_path: Path, monkeypatch) -> None:
    content = str(_save_png(tmp_path / "chart.png", (360, 240), noisy=True))
    tiny = str(_save_png(tmp_path / "logo.png", (8, 8)))
    figs = [
        _fig("cap", "Detailed bar chart of quarterly revenue across six regions", content),
        _fig("vis", "Fig 2", content),
        _fig("dec", None, tiny),
    ]
    monkeypatch.setattr(processor, "_describe_image", lambda p: "A described chart.")
    res = processor.process(_result(figs))

    assert res.vision_calls == 1  # only the Tier-2 figure
    assert res.skipped == 1 and res.skipped_element_ids == ["dec"]
    by_id = {f.element_id: f for f in res.processed_figures}
    assert set(by_id) == {"cap", "vis"}
    assert by_id["cap"].tier == 1 and by_id["cap"].extraction_method == "caption"
    assert by_id["vis"].tier == 2 and by_id["vis"].text == "A described chart."


def test_vision_failure_falls_back_to_caption(
    processor: ImageProcessor, tmp_path: Path, monkeypatch
) -> None:
    content = str(_save_png(tmp_path / "chart.png", (360, 240), noisy=True))
    fig = _fig("vis", "Fig 2", content)
    monkeypatch.setattr(processor, "_describe_image", lambda p: None)
    res = processor.process(_result([fig]))
    out = res.processed_figures[0]
    assert res.vision_calls == 1  # the call was still attempted
    assert out.extraction_method == "vision_failed"
    assert out.text == "Fig 2"  # fell back to caption


# --------------------------------------------------------------------------- #
# Real vision call against the configured Ollama model
# --------------------------------------------------------------------------- #
def test_real_vision_call_on_sample_pdf(processor: ImageProcessor, sample_pdf) -> None:
    """End-to-end: the sample chart is described by the real configured model."""
    pr = DocumentParser().parse(sample_pdf)
    res = processor.process(pr)
    assert res.vision_calls == 1 and res.skipped == 0
    fig = res.processed_figures[0]
    assert fig.tier == 2 and fig.extraction_method == "vision"
    assert len(fig.text) > 20  # a real, non-trivial description
    assert isinstance(fig, FigureChunk)
