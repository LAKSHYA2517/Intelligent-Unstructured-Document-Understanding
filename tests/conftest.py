"""Shared pytest fixtures: synthetic PDF/DOCX documents for the pipeline tests.

These generators build small but structurally rich documents (headings,
paragraphs, an embedded chart image with a caption, a footnote, and a table that
repeats its header across two pages) so the parser and downstream stages can be
exercised without shipping binary fixtures.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


def _make_chart_png() -> bytes:
    """Return PNG bytes of a labelled bar chart (a content image, > 5 KB).

    Subtle noise is blended in so the PNG does not compress below the Tier-3
    decorative size threshold, making this a realistic content figure that the
    image processor routes to the vision tier rather than skipping.
    """
    from PIL import Image, ImageDraw

    w, h = 360, 240
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    # Axes + gridlines.
    draw.line([40, 210, 340, 210], fill="black", width=2)
    draw.line([40, 20, 40, 210], fill="black", width=2)
    for gy in range(40, 210, 32):
        draw.line([40, gy, 340, gy], fill=(220, 220, 220), width=1)
    # Bars with distinct colours and value labels.
    values = [120, 90, 150, 70, 110, 85]
    colours = [(60, 120, 200), (200, 80, 80), (80, 180, 120),
               (220, 160, 40), (140, 100, 200), (90, 90, 90)]
    for i, (val, col) in enumerate(zip(values, colours)):
        x = 55 + i * 48
        bar_h = int(val * 1.2)
        draw.rectangle([x, 210 - bar_h, x + 32, 210], fill=col, outline="black")
        draw.text((x + 4, 210 - bar_h - 12), str(val), fill="black")
        draw.text((x + 6, 214), f"R{i + 1}", fill="black")
    draw.text((120, 4), "Revenue by Region", fill="black")
    # Blend faint noise so the chart is not trivially compressible.
    noise = Image.effect_noise((w, h), 32).convert("RGB")
    img = Image.blend(img, noise, 0.10)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_sample_pdf(path: Path) -> Path:
    """Write a multi-page sample PDF exercising every element type.

    Uses Platypus flowables so the tables carry real grid lines (TableFormer
    needs borders to recognise a table). The same header row appears on two
    separate table flowables across a page break to exercise multi-page merging.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    caption_style = ParagraphStyle("cap", parent=styles["Italic"], fontSize=9)
    footnote_style = ParagraphStyle("fn", parent=styles["Normal"], fontSize=7)

    header = ["Region", "Q1", "Q2", "Q3"]
    table_style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    )

    def _table(rows: list[list[str]]) -> Table:
        t = Table([header, *rows], colWidths=[1.4 * inch] * 4)
        t.setStyle(table_style)
        return t

    story = [
        Paragraph("Quarterly Performance Report", styles["Title"]),
        Paragraph("1. Executive Summary", styles["Heading1"]),
        Paragraph(
            "Acme Corporation reported strong growth this quarter across all "
            "regions. Revenue increased while operating costs remained broadly "
            "stable year over year.",
            styles["BodyText"],
        ),
        Spacer(1, 0.2 * inch),
        Image(io.BytesIO(_make_chart_png()), width=2.4 * inch, height=1.6 * inch),
        Paragraph("Figure 1: Quarterly revenue by region.", caption_style),
        Spacer(1, 0.2 * inch),
        Paragraph("1. All figures are unaudited and in USD millions.", footnote_style),
        PageBreak(),
        Paragraph("2. Regional Breakdown", styles["Heading1"]),
        _table([["North", "120", "130", "150"], ["South", "90", "95", "100"]]),
        PageBreak(),
        Paragraph("2. Regional Breakdown (continued)", styles["Heading1"]),
        _table([["East", "70", "80", "85"], ["West", "60", "65", "75"]]),
    ]
    SimpleDocTemplate(str(path), pagesize=letter).build(story)
    return path


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Path to a generated multi-page sample PDF (session-scoped)."""
    path = tmp_path_factory.mktemp("docs") / "quarterly_report.pdf"
    return _build_sample_pdf(path)


# --------------------------------------------------------------------------- #
# Multi-document corpus with deliberately overlapping entities
# --------------------------------------------------------------------------- #
# Both documents mention: Acme Corporation, Globex Industries, Jane Smith,
# New York, Beta Labs. This overlap is what makes global entity resolution
# (Splink) and the cross-document edges (SAME_AS / RELATED_TO / CORROBORATES)
# testable. Shared facts (Acme acquired Beta Labs; Jane Smith left Globex for
# Acme) are stated in both docs to exercise corroboration.
CORPUS_SPECS: list[dict] = [
    {
        "filename": "acme_annual_report.pdf",
        "title": "Acme Corporation Annual Report 2024",
        "sections": [
            ("1. Executive Summary", [
                "Acme Corporation, led by chief executive Jane Smith, reported "
                "record revenue of 540 million dollars in 2024. The company is "
                "headquartered in New York.",
                "Acme Corporation acquired Beta Labs, a software startup, in March "
                "2024. Acme Corporation competes with Globex Industries in cloud "
                "services.",
            ]),
            ("2. Leadership", [
                "Jane Smith joined Acme Corporation in 2019. She previously worked "
                "at Globex Industries. Under her leadership Acme Corporation "
                "expanded into Europe and Asia.",
            ]),
        ],
        "figure_caption": "Figure 1: Acme revenue by region for fiscal year 2024.",
        "table_header": ["Region", "2023", "2024"],
        "table_rows": [["Americas", "210", "260"], ["Europe", "120", "160"],
                       ["Asia", "80", "120"]],
    },
    {
        "filename": "globex_q3_filing.pdf",
        "title": "Globex Industries Quarterly Filing Q3 2024",
        "sections": [
            ("1. Overview", [
                "Globex Industries, headquartered in New York, reported revenue of "
                "480 million dollars in the third quarter of 2024. John Doe serves "
                "as the chief executive of Globex Industries.",
                "Globex Industries competes with Acme Corporation across several "
                "markets.",
            ]),
            ("2. Corporate Developments", [
                "Globex Industries attempted to acquire Beta Labs, but the "
                "acquisition was completed by Acme Corporation. Jane Smith, "
                "formerly of Globex Industries, now leads Acme Corporation.",
            ]),
        ],
        "figure_caption": "Figure 1: Globex quarterly revenue trend for 2024.",
        "table_header": ["Quarter", "Revenue", "Margin"],
        "table_rows": [["Q1", "440", "18%"], ["Q2", "460", "19%"],
                       ["Q3", "480", "20%"]],
    },
]


def _build_report_pdf(path: Path, spec: dict) -> Path:
    """Build an entity-rich, multi-page report PDF from a corpus spec."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    caption_style = ParagraphStyle("cap", parent=styles["Italic"], fontSize=9)

    story = [Paragraph(spec["title"], styles["Title"]), Spacer(1, 0.15 * inch)]
    for heading, paragraphs in spec["sections"]:
        story.append(Paragraph(heading, styles["Heading1"]))
        for para in paragraphs:
            story.append(Paragraph(para, styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))

    story += [
        Image(io.BytesIO(_make_chart_png()), width=2.4 * inch, height=1.6 * inch),
        Paragraph(spec["figure_caption"], caption_style),
        PageBreak(),
        Paragraph("Financial Tables", styles["Heading1"]),
    ]
    table = Table([spec["table_header"], *spec["table_rows"]],
                  colWidths=[1.6 * inch] * len(spec["table_header"]))
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    SimpleDocTemplate(str(path), pagesize=letter).build(story)
    return path


@pytest.fixture(scope="session")
def corpus_pdfs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """Paths to the generated multi-document corpus (overlapping entities)."""
    out_dir = tmp_path_factory.mktemp("corpus")
    return [_build_report_pdf(out_dir / s["filename"], s) for s in CORPUS_SPECS]


if __name__ == "__main__":
    out = _build_sample_pdf(Path("data/uploads/quarterly_report.pdf"))
    print(f"Wrote sample PDF to {out}")
    for spec in CORPUS_SPECS:
        p = _build_report_pdf(Path("data/uploads") / spec["filename"], spec)
        print(f"Wrote corpus PDF to {p}")
