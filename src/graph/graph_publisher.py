"""Stage 8 — publishing documents to the FalkorDB knowledge graph.

:meth:`GraphPublisher.publish_document` writes one ingested document's complete
sub-graph: the Document/Section/Chunk/Figure nodes, the entity nodes (each with a
generic ``:Entity`` label plus a specific type label), and the structural,
cross-modal, entity→chunk, and entity→entity edges — all with provenance
properties. :meth:`publish_cross_document_edges` later applies the cross-document
edges produced by global entity resolution (Stage 7).

All writes are idempotent (``MERGE`` on stable ids) so re-publishing a document
is safe. Relationship types and node labels are inlined per group (Cypher cannot
parametrise them); every value is otherwise passed as a bound parameter.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from src.extraction.gliner_extractor import ExtractionResult, Triple
from src.extraction.ner import Entity
from src.graph.falkordb_client import FalkorDBClient, get_client
from src.ingestion.chunker import Chunk, ChunkResult
from src.ingestion.parser import ParsedElement, ParseResult

logger = logging.getLogger(__name__)

# Core entity types that become first-class node labels (beyond :Entity).
_CORE_TYPES = {"Person", "Organisation", "Location", "Event", "Object", "Date"}
_VALID_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Triple predicate (snake_case) → specific entity→entity edge type. Matched by
# substring; anything unmatched becomes a generic RELATED_TO carrying the
# predicate as a property.
_PREDICATE_EDGES: list[tuple[str, str]] = [
    ("subsidiar", "SUBSIDIARY_OF"),
    ("acquir", "ACQUIRED"),
    ("compet", "COMPETES_WITH"),
    ("found", "FOUNDED"),
    ("work", "WORKS_AT"),
    ("headquarter", "LOCATED_IN"),
    ("locat", "LOCATED_IN"),
    ("based", "LOCATED_IN"),
    ("led", "WORKS_AT"),
]
_ALL_ENTITY_EDGE_TYPES = (
    "WORKS_AT", "FOUNDED", "LOCATED_IN", "ACQUIRED", "SUBSIDIARY_OF",
    "COMPETES_WITH", "RELATED_TO",
)


@dataclass
class PublishResult:
    """Counts written for one document (for logging / pipeline return values)."""

    document_id: str
    sections: int = 0
    chunks: int = 0
    figures: int = 0
    entities: int = 0
    relationships: int = 0


class GraphPublisher:
    """Writes parsed/extracted documents into the FalkorDB graph."""

    def __init__(self, client: FalkorDBClient | None = None) -> None:
        self._db = client or get_client()
        self._db.ensure_indexes()

    # ================================================================== #
    # Per-document publishing
    # ================================================================== #
    def publish_document(
        self,
        parse_result: ParseResult,
        chunk_result: ChunkResult,
        extraction_result: ExtractionResult,
        domain: str,
    ) -> PublishResult:
        """Publish one document's full sub-graph.

        Args:
            parse_result: Stage 1 output (elements, counts, reading order).
            chunk_result: Stage 3 output with embeddings attached (Stage 6).
            extraction_result: Stage 5 entities and triples.
            domain: Detected domain (Stage 4) stored on the Document node.

        Returns:
            A :class:`PublishResult` with the real counts written.
        """
        ts = datetime.now(timezone.utc).isoformat()
        doc_id = parse_result.doc_id

        self._publish_document_node(parse_result, chunk_result, domain, ts)
        sections, title_to_section = self._publish_sections(parse_result, ts)
        self._publish_chunks(chunk_result, title_to_section, ts)
        figures = self._publish_figures(parse_result, chunk_result, ts)
        self._publish_structural_edges(chunk_result, title_to_section)
        self._publish_cross_modal_edges(parse_result, chunk_result)
        entity_count = self._publish_entities(extraction_result.entities, ts)
        self._publish_entity_chunk_edges(extraction_result.entities)
        rel_count = self._publish_entity_relations(extraction_result, doc_id, ts)

        result = PublishResult(
            document_id=doc_id,
            sections=len(sections),
            chunks=chunk_result.chunk_count,
            figures=figures,
            entities=entity_count,
            relationships=rel_count,
        )
        logger.info(
            "Published '%s': %d sections, %d chunks, %d figures, %d entities, %d relations",
            parse_result.doc_name, result.sections, result.chunks, result.figures,
            result.entities, result.relationships,
        )
        return result

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def _publish_document_node(
        self, parse_result: ParseResult, chunk_result: ChunkResult, domain: str, ts: str
    ) -> None:
        """MERGE the Document node with its real statistics."""
        self._db.query(
            """
            MERGE (d:Document {id: $id})
            SET d.name = $name, d.domain = $domain, d.page_count = $pages,
                d.table_count = $tables, d.figure_count = $figures,
                d.chunk_count = $chunks, d.ingested_at = $ts
            """,
            {
                "id": parse_result.doc_id, "name": parse_result.doc_name,
                "domain": domain, "pages": parse_result.page_count,
                "tables": parse_result.table_count, "figures": parse_result.figure_count,
                "chunks": chunk_result.chunk_count, "ts": ts,
            },
        )

    def _publish_sections(
        self, parse_result: ParseResult, ts: str
    ) -> tuple[list[dict], dict[str, str]]:
        """Create Section nodes + hierarchy; return sections and title→id map."""
        headings = [e for e in parse_result.elements if e.element_type == "heading"]
        sections: list[dict] = []
        title_to_section: dict[str, str] = {}
        stack: list[tuple[int, str]] = []  # (level, section_id)

        for i, h in enumerate(headings):
            level = h.level or 1
            section_id = f"{parse_result.doc_id}::sec::{i:04d}"
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None
            stack.append((level, section_id))
            sections.append(
                {"id": section_id, "title": h.text.strip(), "level": level,
                 "parent_id": parent_id, "page": h.page_number}
            )
            title_to_section[h.text.strip()] = section_id

        if not sections:
            return sections, title_to_section

        self._db.query(
            """
            UNWIND $sections AS s
            MERGE (sec:Section {id: s.id})
            SET sec.title = s.title, sec.level = s.level, sec.document_id = $doc_id,
                sec.page_number = s.page, sec.source_doc = $doc_id, sec.timestamp = $ts
            WITH sec, s
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:HAS_SECTION]->(sec)
            """,
            {"sections": sections, "doc_id": parse_result.doc_id, "ts": ts},
        )
        sub = [{"parent": s["parent_id"], "child": s["id"]} for s in sections if s["parent_id"]]
        if sub:
            self._db.query(
                """
                UNWIND $pairs AS p
                MATCH (parent:Section {id: p.parent}), (child:Section {id: p.child})
                MERGE (parent)-[:HAS_SUBSECTION]->(child)
                """,
                {"pairs": sub},
            )
        return sections, title_to_section

    def _publish_chunks(
        self, chunk_result: ChunkResult, title_to_section: dict[str, str], ts: str
    ) -> None:
        """MERGE Chunk nodes (table_json serialised, embedding as vecf32)."""
        rows = [
            {
                "id": c.chunk_id, "text": c.text,
                "table_json": json.dumps(c.table_json) if c.table_json else "",
                "element_type": c.element_type, "page_number": c.page_number,
                "document_id": c.document_id, "document_name": c.document_name,
                "section_title": c.section_title, "position": c.position_in_doc,
                "image_path": c.image_path or "", "confidence": 1.0,
                "extraction_method": "docling",
            }
            for c in chunk_result.chunks
        ]
        if not rows:
            return
        self._db.query(
            """
            UNWIND $rows AS r
            MERGE (c:Chunk {id: r.id})
            SET c.text = r.text, c.table_json = r.table_json,
                c.element_type = r.element_type, c.page_number = r.page_number,
                c.document_id = r.document_id, c.document_name = r.document_name,
                c.section_title = r.section_title, c.position = r.position,
                c.image_path = r.image_path, c.chunk_id = r.id,
                c.source_doc = r.document_id, c.confidence = r.confidence,
                c.extraction_method = r.extraction_method, c.timestamp = $ts
            """,
            {"rows": rows, "ts": ts},
        )
        # Vectors set separately so chunks without an embedding still publish.
        vec_rows = [
            {"id": c.chunk_id, "embedding": c.embedding}
            for c in chunk_result.chunks
            if c.embedding is not None
        ]
        if vec_rows:
            self._db.query(
                """
                UNWIND $rows AS r
                MATCH (c:Chunk {id: r.id})
                SET c.embedding = vecf32(r.embedding)
                """,
                {"rows": vec_rows},
            )

    def _publish_figures(
        self, parse_result: ParseResult, chunk_result: ChunkResult, ts: str
    ) -> int:
        """Create Figure nodes from figure elements + their description chunks."""
        desc_by_element = {
            c.source_element_id: c.text
            for c in chunk_result.chunks
            if c.element_type == "figure" and c.source_element_id
        }
        rows = []
        for el in parse_result.elements:
            if el.element_type != "figure":
                continue
            rows.append({
                "id": f"{parse_result.doc_id}::fig::{el.element_id}",
                "image_path": el.image_path or "",
                "description": desc_by_element.get(el.element_id, el.caption or ""),
                "page_number": el.page_number, "document_id": parse_result.doc_id,
            })
        if not rows:
            return 0
        self._db.query(
            """
            UNWIND $rows AS r
            MERGE (f:Figure {id: r.id})
            SET f.image_path = r.image_path, f.description = r.description,
                f.page_number = r.page_number, f.document_id = r.document_id,
                f.source_doc = r.document_id, f.timestamp = $ts
            WITH f, r
            MATCH (d:Document {id: r.document_id})
            MERGE (d)-[:HAS_FIGURE]->(f)
            """,
            {"rows": rows, "ts": ts},
        )
        return len(rows)

    def _publish_entities(self, entities: list[Entity], ts: str) -> int:
        """MERGE entity nodes grouped by type so each gets a specific label."""
        by_label: dict[str, list[dict]] = defaultdict(list)
        for e in entities:
            label = e.entity_type if _VALID_LABEL.match(e.entity_type) else "Concept"
            by_label[label].append({
                "id": self._entity_id(e),
                "canonical_name": e.name, "name": e.name, "type": e.entity_type,
                "aliases": e.aliases, "source_doc": e.document_id,
                "source_documents": [e.document_id],
                "confidence": e.confidence, "extraction_method": e.extraction_method,
            })

        total = 0
        for label, rows in by_label.items():
            extra = f":{label}" if label not in {"Entity"} else ""
            self._db.query(
                f"""
                UNWIND $rows AS r
                MERGE (e:Entity{extra} {{id: r.id}})
                SET e.canonical_name = r.canonical_name, e.name = r.name,
                    e.type = r.type, e.aliases = r.aliases, e.source_doc = r.source_doc,
                    e.source_documents = r.source_documents, e.confidence = r.confidence,
                    e.extraction_method = r.extraction_method, e.timestamp = $ts
                """,
                {"rows": rows, "ts": ts},
            )
            total += len(rows)
        return total

    # ------------------------------------------------------------------ #
    # Edges
    # ------------------------------------------------------------------ #
    def _publish_structural_edges(
        self, chunk_result: ChunkResult, title_to_section: dict[str, str]
    ) -> None:
        """HAS_CHUNK, PART_OF, PRECEDES, CONTAINS, BELONGS_TO_SECTION."""
        chunks = chunk_result.chunks
        if not chunks:
            return
        doc_id = chunks[0].document_id
        ids = [{"id": c.chunk_id} for c in chunks]
        self._db.query(
            """
            UNWIND $ids AS r
            MATCH (d:Document {id: $doc_id}), (c:Chunk {id: r.id})
            MERGE (d)-[:HAS_CHUNK]->(c)
            MERGE (c)-[:PART_OF]->(d)
            """,
            {"ids": ids, "doc_id": doc_id},
        )
        precedes = [
            {"a": chunks[i].chunk_id, "b": chunks[i + 1].chunk_id}
            for i in range(len(chunks) - 1)
        ]
        if precedes:
            self._db.query(
                """
                UNWIND $pairs AS p
                MATCH (a:Chunk {id: p.a}), (b:Chunk {id: p.b})
                MERGE (a)-[:PRECEDES]->(b)
                """,
                {"pairs": precedes},
            )
        sec_pairs = [
            {"chunk": c.chunk_id, "sec": title_to_section[c.section_title]}
            for c in chunks
            if c.section_title in title_to_section
        ]
        if sec_pairs:
            self._db.query(
                """
                UNWIND $pairs AS p
                MATCH (c:Chunk {id: p.chunk}), (s:Section {id: p.sec})
                MERGE (s)-[:CONTAINS]->(c)
                MERGE (c)-[:BELONGS_TO_SECTION]->(s)
                """,
                {"pairs": sec_pairs},
            )

    def _publish_cross_modal_edges(
        self, parse_result: ParseResult, chunk_result: ChunkResult
    ) -> None:
        """INTRODUCES, REFERENCES, ELABORATES, VISUALISED_BY, CAPTIONED_BY,
        DESCRIBES, ANNOTATES — inferred from element order and caption parentage."""
        chunks = chunk_result.chunks
        by_type: dict[str, list[dict]] = defaultdict(list)

        # element_id → chunk and caption parentage (figure element) lookups.
        chunk_by_element = {c.source_element_id: c for c in chunks if c.source_element_id}
        ref_to_element = {e.self_ref: e.element_id for e in parse_result.elements if e.self_ref}
        caption_parent_chunk: dict[str, Chunk] = {}
        for el in parse_result.elements:
            if el.element_type == "caption" and el.parent_ref:
                parent_el_id = ref_to_element.get(el.parent_ref)
                cap_chunk = chunk_by_element.get(el.element_id)
                fig_chunk = chunk_by_element.get(parent_el_id)
                if cap_chunk and fig_chunk:
                    by_type["CAPTIONED_BY"].append({"a": fig_chunk.chunk_id, "b": cap_chunk.chunk_id})
                    by_type["DESCRIBES"].append({"a": cap_chunk.chunk_id, "b": fig_chunk.chunk_id})

        last_paragraph: Chunk | None = None
        last_table: Chunk | None = None
        last_content: Chunk | None = None
        for c in chunks:
            if c.element_type == "table":
                if last_paragraph:
                    by_type["INTRODUCES"].append({"a": last_paragraph.chunk_id, "b": c.chunk_id})
                last_table = c
            elif c.element_type == "figure":
                if last_paragraph:
                    by_type["REFERENCES"].append({"a": last_paragraph.chunk_id, "b": c.chunk_id})
                if last_table and last_table.section_title == c.section_title:
                    by_type["VISUALISED_BY"].append({"a": last_table.chunk_id, "b": c.chunk_id})
            elif c.element_type == "paragraph":
                if last_paragraph and last_paragraph.section_title == c.section_title:
                    by_type["ELABORATES"].append({"a": last_paragraph.chunk_id, "b": c.chunk_id})
                last_paragraph = c
            elif c.element_type == "footnote":
                if last_content:
                    by_type["ANNOTATES"].append({"a": c.chunk_id, "b": last_content.chunk_id})
            if c.element_type != "footnote":
                last_content = c

        for edge_type, pairs in by_type.items():
            self._merge_chunk_edges(edge_type, pairs)

    def _merge_chunk_edges(self, edge_type: str, pairs: list[dict]) -> None:
        """MERGE a batch of Chunk→Chunk edges of one (inlined) type."""
        if not pairs:
            return
        self._db.query(
            f"""
            UNWIND $pairs AS p
            MATCH (a:Chunk {{id: p.a}}), (b:Chunk {{id: p.b}})
            MERGE (a)-[:{edge_type}]->(b)
            """,
            {"pairs": pairs},
        )

    def _publish_entity_chunk_edges(self, entities: list[Entity]) -> None:
        """MENTIONED_IN, FIRST_MENTIONED_IN, DEFINED_IN."""
        mentions, firsts = [], []
        for e in entities:
            eid = self._entity_id(e)
            for m in e.mentions:
                mentions.append({"eid": eid, "cid": m.chunk_id})
            if e.first_chunk_id:
                firsts.append({"eid": eid, "cid": e.first_chunk_id})
        if mentions:
            self._db.query(
                """
                UNWIND $rows AS r
                MATCH (e:Entity {id: r.eid}), (c:Chunk {id: r.cid})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                {"rows": mentions},
            )
        if firsts:
            self._db.query(
                """
                UNWIND $rows AS r
                MATCH (e:Entity {id: r.eid}), (c:Chunk {id: r.cid})
                MERGE (e)-[:FIRST_MENTIONED_IN]->(c)
                MERGE (e)-[:DEFINED_IN]->(c)
                """,
                {"rows": firsts},
            )

    def _publish_entity_relations(
        self, extraction_result: ExtractionResult, doc_id: str, ts: str
    ) -> int:
        """Create entity→entity edges from triples (typed by predicate)."""
        name_to_id = self._entity_name_index(extraction_result.entities)
        by_type: dict[str, list[dict]] = defaultdict(list)
        concept_nodes: dict[str, dict] = {}

        for t in extraction_result.triples:
            sid = self._resolve_or_concept(t.subject, doc_id, name_to_id, concept_nodes)
            oid = self._resolve_or_concept(t.object, doc_id, name_to_id, concept_nodes)
            if sid == oid:
                continue
            edge_type = self._edge_type_for(t.predicate)
            by_type[edge_type].append({
                "a": sid, "b": oid, "predicate": t.predicate,
                "confidence": t.confidence, "chunk_id": t.chunk_id, "source_doc": doc_id,
            })

        if concept_nodes:
            self._db.query(
                """
                UNWIND $rows AS r
                MERGE (e:Entity:Concept {id: r.id})
                SET e.canonical_name = r.name, e.name = r.name, e.type = 'Concept',
                    e.source_doc = $doc_id, e.source_documents = [$doc_id],
                    e.extraction_method = 'llm_triple', e.timestamp = $ts
                """,
                {"rows": list(concept_nodes.values()), "doc_id": doc_id, "ts": ts},
            )

        total = 0
        for edge_type, rows in by_type.items():
            self._db.query(
                f"""
                UNWIND $rows AS r
                MATCH (a:Entity {{id: r.a}}), (b:Entity {{id: r.b}})
                MERGE (a)-[rel:{edge_type}]->(b)
                SET rel.predicate = r.predicate, rel.confidence = r.confidence,
                    rel.chunk_id = r.chunk_id, rel.source_doc = r.source_doc
                """,
                {"rows": rows},
            )
            total += len(rows)
        return total

    # ================================================================== #
    # Cross-document edges (Stage 7 output)
    # ================================================================== #
    def publish_cross_document_edges(self, resolution_result) -> int:
        """Apply cross-document edges from global entity resolution.

        Expects ``resolution_result.cross_doc_edges`` as a list of dicts with
        ``type`` (one of SAME_AS, RELATED_TO, CORROBORATES, CONTRADICTS),
        ``from_id``, ``to_id``, and optional ``props``. The endpoint label is
        inferred from the edge type (entity edges match :Entity, document edges
        :Document, chunk edges :Chunk).

        Returns:
            The number of edges written.
        """
        edges = getattr(resolution_result, "cross_doc_edges", None) or []
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            by_type[e["type"]].append(e)

        label_for = {
            "SAME_AS": "Entity", "RELATED_TO": "Document",
            "CORROBORATES": "Chunk", "CONTRADICTS": "Chunk",
        }
        total = 0
        for edge_type, rows in by_type.items():
            label = label_for.get(edge_type, "Entity")
            payload = [
                {"a": r["from_id"], "b": r["to_id"], "props": r.get("props", {})}
                for r in rows
            ]
            self._db.query(
                f"""
                UNWIND $rows AS r
                MATCH (a:{label} {{id: r.a}}), (b:{label} {{id: r.b}})
                MERGE (a)-[rel:{edge_type}]->(b)
                SET rel += r.props
                """,
                {"rows": payload},
            )
            total += len(payload)
        logger.info("Published %d cross-document edges", total)
        return total

    def apply_canonical_metadata(self, canonical_entities) -> int:
        """Stamp resolved canonical metadata onto each cluster's member nodes.

        Writes ``canonical_name``, ``aliases``, ``source_documents``,
        ``cluster_id`` and ``confidence`` (from Stage 7) onto every member
        :class:`Entity` node so the cluster shares one canonical identity.

        Args:
            canonical_entities: ``ResolutionResult.canonical_entities``.

        Returns:
            The number of member nodes updated.
        """
        rows = [
            {
                "ids": c.member_ids, "canonical_name": c.canonical_name,
                "aliases": c.aliases, "source_documents": c.source_documents,
                "cluster_id": c.cluster_id, "confidence": c.confidence,
            }
            for c in canonical_entities
        ]
        if not rows:
            return 0
        self._db.query(
            """
            UNWIND $rows AS r
            UNWIND r.ids AS eid
            MATCH (e:Entity {id: eid})
            SET e.canonical_name = r.canonical_name, e.aliases = r.aliases,
                e.source_documents = r.source_documents, e.cluster_id = r.cluster_id,
                e.resolution_confidence = r.confidence
            """,
            {"rows": rows},
        )
        return sum(len(c.member_ids) for c in canonical_entities)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _entity_id(entity: Entity) -> str:
        """Stable, document-scoped id for a pre-resolution entity node."""
        norm = " ".join(entity.name.lower().split())
        return f"{entity.document_id}::{entity.entity_type}::{norm}"

    @staticmethod
    def _entity_name_index(entities: list[Entity]) -> dict[str, str]:
        """Map normalised names and aliases to entity ids for triple linking."""
        index: dict[str, str] = {}
        for e in entities:
            eid = GraphPublisher._entity_id(e)
            index[" ".join(e.name.lower().split())] = eid
            for alias in e.aliases:
                index.setdefault(" ".join(alias.lower().split()), eid)
        return index

    @staticmethod
    def _resolve_or_concept(
        surface: str, doc_id: str, name_to_id: dict[str, str], concept_nodes: dict[str, dict]
    ) -> str:
        """Return an existing entity id for ``surface`` or mint a Concept node."""
        norm = " ".join(surface.lower().split())
        if norm in name_to_id:
            return name_to_id[norm]
        cid = f"{doc_id}::Concept::{norm}"
        concept_nodes.setdefault(cid, {"id": cid, "name": surface})
        return cid

    @staticmethod
    def _edge_type_for(predicate: str) -> str:
        """Map a snake_case predicate to a specific edge type or RELATED_TO."""
        for needle, edge in _PREDICATE_EDGES:
            if needle in predicate:
                return edge
        return "RELATED_TO"


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.graph.graph_publisher <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.extraction.domain_detector import DomainDetector
    from src.extraction.gliner_extractor import GLiNERExtractor
    from src.extraction.ner import NERExtractor
    from src.extraction.relation_extractor import RelationExtractor
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser
    from src.retrieval.embedder import Embedder

    pr = DocumentParser().parse(sys.argv[1])
    cr = Chunker().chunk(pr)
    dom = DomainDetector().detect(cr)
    ner = NERExtractor().extract(cr)
    er = GLiNERExtractor().extract(cr, dom, prior_entities=ner.entities)
    er = RelationExtractor().extract(cr, er)
    Embedder().embed(cr)
    res = GraphPublisher().publish_document(pr, cr, er, dom.domain)
    print(f"\nPublished {res}")
