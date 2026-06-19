"""Batch ingestion orchestration — Stages 1-8 then global resolution (Stage 7).

:class:`IngestionPipeline` runs every document in a batch through parsing, image
processing, chunking, domain detection, NER + GLiNER + relation extraction,
embedding, and graph publishing; then, once all documents are in, it runs global
entity resolution across the whole batch and writes the cross-document edges and
canonical metadata.

Heavy model-backed components are created lazily and reused across the batch, and
can be injected for testing. Every stage's typed result is surfaced — both in the
returned :class:`BatchIngestResult` and via an optional ``on_stage`` callback — so
the Streamlit progress tracker is driven entirely by real runtime values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Callback signature: (doc_index, doc_count, doc_name, stage, payload) -> None.
StageCallback = Callable[[int, int, str, str, object], None]


@dataclass
class DocumentIngestResult:
    """All per-stage results for one ingested document."""

    doc_id: str
    doc_name: str
    parse_result: object
    image_result: object
    chunk_result: object
    domain_result: object
    extraction_result: object
    embed_result: object
    publish_result: object


@dataclass
class BatchIngestResult:
    """Outcome of ingesting a whole batch plus global resolution."""

    documents: list[DocumentIngestResult] = field(default_factory=list)
    resolution_result: object = None
    cross_doc_edges_written: int = 0
    canonical_nodes_updated: int = 0

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def total_chunks(self) -> int:
        return sum(d.chunk_result.chunk_count for d in self.documents)

    @property
    def total_entities(self) -> int:
        return sum(d.extraction_result.entity_count for d in self.documents)

    @property
    def total_triples(self) -> int:
        return sum(d.extraction_result.triple_count for d in self.documents)


class IngestionPipeline:
    """Orchestrates batch ingestion and global entity resolution."""

    def __init__(
        self, *, parser=None, image_processor=None, chunker=None, domain_detector=None,
        ner=None, gliner=None, relation_extractor=None, embedder=None, publisher=None,
        resolver=None,
    ) -> None:
        self._parser = parser
        self._image_processor = image_processor
        self._chunker = chunker
        self._domain_detector = domain_detector
        self._ner = ner
        self._gliner = gliner
        self._relation_extractor = relation_extractor
        self._embedder = embedder
        self._publisher = publisher
        self._resolver = resolver
        # Batch accumulators (reset per ingest_batch).
        self._batch_entities: list = []
        self._batch_chunk_text: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest_batch(
        self,
        files: list[str | Path] | list[tuple[str | Path, str]],
        on_stage: StageCallback | None = None,
    ) -> BatchIngestResult:
        """Ingest a batch of documents, then resolve entities globally.

        Args:
            files: Paths, or ``(path, display_name)`` tuples.
            on_stage: Optional callback invoked after each stage with
                ``(doc_index, doc_count, doc_name, stage, payload)``.

        Returns:
            A :class:`BatchIngestResult` with every per-stage result.
        """
        normalised = [self._split_file(f) for f in files]
        count = len(normalised)
        self._batch_entities = []
        self._batch_chunk_text = {}

        batch = BatchIngestResult()
        for index, (path, name) in enumerate(normalised):
            doc_result = self._ingest_document(index, count, path, name, on_stage)
            batch.documents.append(doc_result)

        # Global resolution once all documents are ingested.
        resolver = self._get_resolver()
        resolution = resolver.resolve_all(self._batch_entities, self._batch_chunk_text)
        batch.resolution_result = resolution
        self._emit(on_stage, count, count, "", "resolution", resolution)

        publisher = self._get_publisher()
        batch.cross_doc_edges_written = publisher.publish_cross_document_edges(resolution)
        batch.canonical_nodes_updated = publisher.apply_canonical_metadata(
            resolution.canonical_entities
        )
        self._emit(on_stage, count, count, "", "cross_document", batch.cross_doc_edges_written)

        logger.info(
            "Batch ingested: %d docs, %d chunks, %d entities, %d merged, %d cross-doc edges",
            batch.document_count, batch.total_chunks, batch.total_entities,
            resolution.merged_count, batch.cross_doc_edges_written,
        )
        return batch

    # ------------------------------------------------------------------ #
    # Per-document ingestion (Stages 1-8)
    # ------------------------------------------------------------------ #
    def _ingest_document(
        self, index: int, count: int, path: Path, name: str, on_stage: StageCallback | None
    ) -> DocumentIngestResult:
        """Run one document through Stages 1-8 and accumulate batch state."""
        logger.info("[%d/%d] Ingesting '%s'", index + 1, count, name)

        parse_result = self._get_parser().parse(path, doc_name=name)
        self._emit(on_stage, index, count, name, "parse", parse_result)

        image_result = self._get_image_processor().process(parse_result)
        self._emit(on_stage, index, count, name, "image", image_result)

        chunk_result = self._get_chunker().chunk(parse_result, image_result)
        self._emit(on_stage, index, count, name, "chunk", chunk_result)

        domain_result = self._get_domain_detector().detect(chunk_result)
        self._emit(on_stage, index, count, name, "domain", domain_result)

        ner_result = self._get_ner().extract(chunk_result)
        extraction_result = self._get_gliner().extract(
            chunk_result, domain_result, prior_entities=ner_result.entities
        )
        extraction_result = self._get_relation_extractor().extract(chunk_result, extraction_result)
        self._emit(on_stage, index, count, name, "extract", extraction_result)

        embed_result = self._get_embedder().embed(chunk_result)
        self._emit(on_stage, index, count, name, "embed", embed_result)

        publish_result = self._get_publisher().publish_document(
            parse_result, chunk_result, extraction_result, domain_result.domain
        )
        self._emit(on_stage, index, count, name, "publish", publish_result)

        # Accumulate for global resolution.
        self._batch_entities.extend(extraction_result.entities)
        for chunk in chunk_result.chunks:
            self._batch_chunk_text[chunk.chunk_id] = chunk.text

        return DocumentIngestResult(
            doc_id=parse_result.doc_id, doc_name=name, parse_result=parse_result,
            image_result=image_result, chunk_result=chunk_result,
            domain_result=domain_result, extraction_result=extraction_result,
            embed_result=embed_result, publish_result=publish_result,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_file(item) -> tuple[Path, str]:
        """Normalise a file entry into ``(path, display_name)``."""
        if isinstance(item, (tuple, list)):
            path, name = item
            return Path(path), name
        path = Path(item)
        return path, path.name

    @staticmethod
    def _emit(
        cb: StageCallback | None, index: int, count: int, name: str, stage: str, payload: object
    ) -> None:
        """Invoke the progress callback, swallowing callback errors."""
        if cb is None:
            return
        try:
            cb(index, count, name, stage, payload)
        except Exception as exc:  # noqa: BLE001 - UI callbacks must never break ingestion
            logger.warning("on_stage callback raised for stage '%s': %s", stage, exc)

    # --- Lazy component accessors ------------------------------------- #
    def _get_parser(self):
        if self._parser is None:
            from src.ingestion.parser import DocumentParser
            self._parser = DocumentParser()
        return self._parser

    def _get_image_processor(self):
        if self._image_processor is None:
            from src.ingestion.image_processor import ImageProcessor
            self._image_processor = ImageProcessor()
        return self._image_processor

    def _get_chunker(self):
        if self._chunker is None:
            from src.ingestion.chunker import Chunker
            self._chunker = Chunker()
        return self._chunker

    def _get_domain_detector(self):
        if self._domain_detector is None:
            from src.extraction.domain_detector import DomainDetector
            self._domain_detector = DomainDetector()
        return self._domain_detector

    def _get_ner(self):
        if self._ner is None:
            from src.extraction.ner import NERExtractor
            self._ner = NERExtractor()
        return self._ner

    def _get_gliner(self):
        if self._gliner is None:
            from src.extraction.gliner_extractor import GLiNERExtractor
            self._gliner = GLiNERExtractor()
        return self._gliner

    def _get_relation_extractor(self):
        if self._relation_extractor is None:
            from src.extraction.relation_extractor import RelationExtractor
            self._relation_extractor = RelationExtractor()
        return self._relation_extractor

    def _get_embedder(self):
        if self._embedder is None:
            from src.retrieval.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _get_publisher(self):
        if self._publisher is None:
            from src.graph.graph_publisher import GraphPublisher
            self._publisher = GraphPublisher()
        return self._publisher

    def _get_resolver(self):
        if self._resolver is None:
            from src.resolution.entity_resolution import EntityResolver
            self._resolver = EntityResolver(embedder=self._get_embedder())
        return self._resolver


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = sys.argv[1:] or [
        "data/uploads/acme_annual_report.pdf", "data/uploads/globex_q3_filing.pdf",
    ]
    from src.graph.falkordb_client import get_client

    get_client().clear_graph()

    def _progress(i, n, name, stage, payload):
        print(f"  [{i + 1}/{n}] {name or '-'} :: {stage}")

    result = IngestionPipeline().ingest_batch(paths, on_stage=_progress)
    print(f"\nDocuments: {result.document_count}")
    print(f"Total chunks: {result.total_chunks}, entities: {result.total_entities}, "
          f"triples: {result.total_triples}")
    print(f"Merged entities: {result.resolution_result.merged_count}")
    print(f"Cross-doc edges: {result.cross_doc_edges_written}, "
          f"canonical nodes updated: {result.canonical_nodes_updated}")
