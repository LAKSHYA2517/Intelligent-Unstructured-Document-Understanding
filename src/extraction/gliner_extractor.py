"""Stage 5, Step 2 — GLiNER entity extraction with runtime label sets.

Extracts entities from each chunk with GLiNER, using a label set assembled **at
runtime** from the always-on base labels plus the domain-extension labels chosen
by the Stage 4 :class:`~src.extraction.domain_detector.DomainResult`. The GLiNER
model name comes from ``config.gliner_model`` and the score cut-off from
``config.gliner_confidence_threshold`` — neither is hardcoded.

This module also owns the spec's :class:`ExtractionResult` (entities + triples).
GLiNER entities are merged with the spaCy/coref entities from Step 1; triples are
left empty here and populated by the LLM relation extractor (Step 11), keeping
the pipeline to a single lightweight GLiNER model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import config
from src.extraction.domain_detector import DomainResult
from src.extraction.ner import Entity
from src.ingestion.chunker import ChunkResult

logger = logging.getLogger(__name__)

# Always-on, domain-agnostic base labels (descriptive phrases improve GLiNER).
BASE_LABELS: list[str] = [
    "Person or individual or human",
    "Organisation or company or institution or agency",
    "Geographic location or place or region or country",
    "Event or incident or occurrence or happening",
    "Physical object or product or asset or item",
    "Date or time period or year or quarter",
    "Quantity or measurement or numerical value or percentage",
    "Legal or regulatory document or law or policy",
    "Concept or idea or topic or subject",
]

# Domain-specific extension labels, selected at runtime by detected domain.
DOMAIN_EXTENSION_LABELS: dict[str, list[str]] = {
    "finance": [
        "Monetary amount or currency value or price",
        "Financial instrument or security or stock",
        "Stock ticker or trading symbol",
        "Fiscal period or quarter or fiscal year",
        "Accounting metric or KPI or financial ratio",
    ],
    "legal": [
        "Legal clause or provision or article",
        "Court or jurisdiction or tribunal",
        "Case reference or citation or docket",
        "Contractual obligation or term or condition",
        "Legal party or signatory or plaintiff or defendant",
    ],
    "medical": [
        "Drug compound or medication or pharmaceutical",
        "Medical condition or diagnosis or disease",
        "Clinical trial or study or research",
        "Dosage or treatment protocol or procedure",
        "Anatomical structure or organ or tissue",
    ],
    "technical": [
        "Software system or platform or framework",
        "Technical specification or standard or protocol",
        "Algorithm or method or technique",
        "Hardware component or device or chip",
    ],
    "general": [],
}

# Map each descriptive label phrase to a short canonical entity type used on
# graph nodes. Domain types extend the base set without overlapping it.
_LABEL_TO_TYPE: dict[str, str] = {
    BASE_LABELS[0]: "Person",
    BASE_LABELS[1]: "Organisation",
    BASE_LABELS[2]: "Location",
    BASE_LABELS[3]: "Event",
    BASE_LABELS[4]: "Object",
    BASE_LABELS[5]: "Date",
    BASE_LABELS[6]: "Quantity",
    BASE_LABELS[7]: "LegalDocument",
    BASE_LABELS[8]: "Concept",
    # finance
    "Monetary amount or currency value or price": "MonetaryAmount",
    "Financial instrument or security or stock": "FinancialInstrument",
    "Stock ticker or trading symbol": "StockTicker",
    "Fiscal period or quarter or fiscal year": "FiscalPeriod",
    "Accounting metric or KPI or financial ratio": "AccountingMetric",
    # legal
    "Legal clause or provision or article": "LegalClause",
    "Court or jurisdiction or tribunal": "Court",
    "Case reference or citation or docket": "CaseReference",
    "Contractual obligation or term or condition": "ContractualObligation",
    "Legal party or signatory or plaintiff or defendant": "LegalParty",
    # medical
    "Drug compound or medication or pharmaceutical": "Drug",
    "Medical condition or diagnosis or disease": "MedicalCondition",
    "Clinical trial or study or research": "ClinicalTrial",
    "Dosage or treatment protocol or procedure": "Treatment",
    "Anatomical structure or organ or tissue": "Anatomy",
    # technical
    "Software system or platform or framework": "SoftwareSystem",
    "Technical specification or standard or protocol": "TechnicalSpec",
    "Algorithm or method or technique": "Algorithm",
    "Hardware component or device or chip": "Hardware",
}


@dataclass
class Triple:
    """A subject-predicate-object relation extracted from a chunk."""

    subject: str
    predicate: str
    object: str
    chunk_id: str
    document_id: str
    confidence: float = 0.5
    extraction_method: str = "llm"


@dataclass
class ExtractionResult:
    """Stage 5 output (entities + triples). Counts are read by the UI."""

    entities: list[Entity]
    triples: list[Triple] = field(default_factory=list)
    entity_count: int = 0
    triple_count: int = 0

    def __post_init__(self) -> None:
        self.entity_count = self.entity_count or len(self.entities)
        self.triple_count = self.triple_count or len(self.triples)

    def recount(self) -> None:
        """Refresh counts after entities/triples are mutated in place."""
        self.entity_count = len(self.entities)
        self.triple_count = len(self.triples)


def assemble_labels(domain: str) -> list[str]:
    """Return BASE_LABELS plus the extension labels for ``domain`` (runtime)."""
    return BASE_LABELS + DOMAIN_EXTENSION_LABELS.get(domain, [])


class GLiNERExtractor:
    """Runs GLiNER over chunks with a domain-aware, runtime-assembled label set."""

    def __init__(self) -> None:
        from gliner import GLiNER

        self._model = GLiNER.from_pretrained(config.gliner_model)
        self._threshold = config.gliner_confidence_threshold
        try:
            import torch

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
        except Exception:  # noqa: BLE001 - CPU fallback is fine
            pass
        self._model.eval()
        logger.info(
            "GLiNERExtractor ready (model from config: %s, threshold=%.2f)",
            config.gliner_model,
            self._threshold,
        )

    def extract(
        self,
        chunk_result: ChunkResult,
        domain_result: DomainResult,
        prior_entities: list[Entity] | None = None,
    ) -> ExtractionResult:
        """Extract entities per chunk and merge with any prior (spaCy) entities.

        Args:
            chunk_result: Stage 3 output.
            domain_result: Stage 4 output; selects the extension labels.
            prior_entities: Entities from Stage 5 Step 1 (spaCy + coref) to merge.

        Returns:
            An :class:`ExtractionResult` with merged entities and empty triples
            (triples are added by the LLM relation extractor in Step 11).
        """
        labels = assemble_labels(domain_result.domain)
        logger.info(
            "GLiNER labels for domain '%s': %d (%d base + %d extension)",
            domain_result.domain,
            len(labels),
            len(BASE_LABELS),
            len(labels) - len(BASE_LABELS),
        )

        merged: dict[tuple[str, str], Entity] = {}
        for entity in prior_entities or []:
            merged[entity.key] = entity

        for chunk in chunk_result.chunks:
            if not chunk.text.strip():
                continue
            try:
                spans = self._model.predict_entities(
                    chunk.text, labels, threshold=self._threshold
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("GLiNER failed on chunk %s (%s); skipping", chunk.chunk_id, exc)
                continue
            for span in spans:
                self._merge_span(merged, span, chunk)

        result = ExtractionResult(entities=list(merged.values()))
        logger.info(
            "GLiNER+spaCy merged to %d entities over %d chunks",
            result.entity_count,
            len(chunk_result.chunks),
        )
        return result

    @staticmethod
    def _merge_span(merged: dict[tuple[str, str], Entity], span: dict, chunk) -> None:
        """Merge one GLiNER span into the entity map (create or enrich)."""
        surface = (span.get("text") or "").strip()
        etype = _LABEL_TO_TYPE.get(span.get("label", ""))
        if not surface or etype is None:
            return
        key = (" ".join(surface.lower().split()), etype)
        entity = merged.get(key)
        if entity is None:
            entity = Entity(
                name=surface,
                entity_type=etype,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                first_chunk_id=chunk.chunk_id,
                first_page=chunk.page_number,
                confidence=float(span.get("score", 0.0)),
                extraction_method="gliner",
            )
            merged[key] = entity
        else:
            # Strengthen provenance: record GLiNER agreement and the mention.
            if entity.extraction_method != "gliner":
                entity.extraction_method = f"{entity.extraction_method}+gliner"
            entity.confidence = max(entity.confidence, float(span.get("score", 0.0)))
        entity.add_mention(chunk.chunk_id, chunk.page_number)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.extraction.gliner_extractor <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.extraction.domain_detector import DomainDetector
    from src.extraction.ner import NERExtractor
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser

    pr = DocumentParser().parse(sys.argv[1])
    cr = Chunker().chunk(pr)
    dom = DomainDetector().detect(cr)
    ner = NERExtractor().extract(cr)
    res = GLiNERExtractor().extract(cr, dom, prior_entities=ner.entities)
    print(f"\ndomain={dom.domain}  entities={res.entity_count}  triples={res.triple_count}")
    for e in res.entities:
        print(f"  {e.entity_type:16s} {e.name!r:30} method={e.extraction_method} conf={e.confidence:.2f}")
