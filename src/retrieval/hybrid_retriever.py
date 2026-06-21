"""Stage 10 — hybrid retrieval (HNSW vector + BM25 full-text + graph) with RRF.

Runs three independent searches over the FalkorDB graph and fuses them with
Reciprocal Rank Fusion:

  1. **Vector**: HNSW KNN over ``Chunk.embedding`` (query embedded with the same
     model used at ingestion) → top-10 by cosine distance.
  2. **Full-text**: BM25 over ``Chunk.text`` (RediSearch) → top-10 by score.
  3. **Graph**: named entities in the query (spaCy) are matched to entity nodes,
     then connected chunks are collected up to two hops along the cross-modal /
     mention / corroboration edges, ranked by hop distance.

RRF combines the per-method ranks as ``score = Σ 1/(rank + 60)``, deduplicates by
chunk id, and returns the top-15 hydrated chunks plus the graph-traversal
metadata (entities, edge types, hops) for the UI's "Graph context" panel.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.graph.falkordb_client import FalkorDBClient, get_client

logger = logging.getLogger(__name__)

_RRF_K = 60
_PER_METHOD_TOP = 10
_FINAL_TOP = 15
_SPACY_MODEL = "en_core_web_sm"

# Edges traversed from query entities to collect related chunks (Search 3).
_TRAVERSAL_EDGES = (
    "MENTIONED_IN", "INTRODUCES", "VISUALISED_BY", "CAPTIONED_BY",
    "ANNOTATES", "SUPPORTS", "CORROBORATES", "REFERENCES", "DESCRIBES",
)
# RediSearch reserves these characters; strip them from query terms.
_FT_STRIP = re.compile(r"[^A-Za-z0-9 ]+")


@dataclass
class RetrievedChunk:
    """One chunk returned by retrieval, with fusion + provenance metadata."""

    chunk_id: str
    text: str
    element_type: str
    page_number: int
    document_id: str
    document_name: str
    section_title: str
    table_json: str | None
    image_path: str | None
    confidence: float
    rrf_score: float = 0.0
    methods: list[str] = field(default_factory=list)
    graph_hops: int | None = None
    cross_score: float = 0.0  # cross-encoder relevance (0-1), set by Stage 11


@dataclass
class RetrievalResult:
    """Stage 10 output: fused chunks plus graph-traversal metadata."""

    chunks: list[RetrievedChunk]
    query_entities: list[str] = field(default_factory=list)
    edge_types_used: list[str] = field(default_factory=list)
    max_hops: int = 0
    method_counts: dict[str, int] = field(default_factory=dict)


class HybridRetriever:
    """Three-way retrieval (vector + BM25 + graph) fused with RRF."""

    def __init__(self, embedder=None, client: FalkorDBClient | None = None, nlp=None) -> None:
        self._embedder = embedder
        self._db = client or get_client()
        self._nlp = nlp

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def retrieve(
        self, query: str, scope_doc_id: str | None = None, top_k: int = _FINAL_TOP
    ) -> RetrievalResult:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural-language query.
            scope_doc_id: If given, restrict results to this document.
            top_k: Number of fused chunks to return.

        Returns:
            A :class:`RetrievalResult` with fused chunks and graph metadata.
        """
        props: dict[str, dict] = {}
        vector_ids = self._vector_search(query, scope_doc_id, props)
        bm25_ids = self._bm25_search(query, scope_doc_id, props)
        graph_ids, query_entities, edge_types, hops_by_chunk = self._graph_search(
            query, scope_doc_id, props
        )

        rankings = {"vector": vector_ids, "bm25": bm25_ids, "graph": graph_ids}
        fused = self._reciprocal_rank_fusion(rankings)

        chunks: list[RetrievedChunk] = []
        for chunk_id, score, methods in fused[:top_k]:
            data = props.get(chunk_id)
            if not data:
                continue
            chunks.append(self._to_chunk(data, score, methods, hops_by_chunk.get(chunk_id)))

        result = RetrievalResult(
            chunks=chunks,
            query_entities=query_entities,
            edge_types_used=edge_types,
            max_hops=max(hops_by_chunk.values(), default=0),
            method_counts={m: len(ids) for m, ids in rankings.items()},
        )
        logger.info(
            "Retrieved %d chunks (vector=%d, bm25=%d, graph=%d, entities=%s)",
            len(chunks), len(vector_ids), len(bm25_ids), len(graph_ids), query_entities,
        )
        return result

    # ------------------------------------------------------------------ #
    # Search 1 — vector
    # ------------------------------------------------------------------ #
    def _vector_search(
        self, query: str, scope_doc_id: str | None, props: dict[str, dict]
    ) -> list[str]:
        """HNSW KNN over chunk embeddings; returns chunk ids ordered by closeness."""
        vector = self._get_embedder().embed_query(query)
        # HNSW is approximate and under-returns for small k, so over-fetch
        # candidates (more when scoping requires post-filtering) and slice.
        k = _PER_METHOD_TOP * (8 if scope_doc_id else 5)
        rows = self._db.query(
            """
            CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($vec))
            YIELD node, score
            RETURN node, score ORDER BY score ASC
            """,
            {"k": k, "vec": vector},
            read_only=True,
        )
        ids: list[str] = []
        for node, _score in rows:
            data = self._node_props(node)
            if scope_doc_id and data.get("document_id") != scope_doc_id:
                continue
            props.setdefault(data["id"], data)
            ids.append(data["id"])
            if len(ids) >= _PER_METHOD_TOP:
                break
        return ids

    # ------------------------------------------------------------------ #
    # Search 2 — BM25 full-text
    # ------------------------------------------------------------------ #
    def _bm25_search(
        self, query: str, scope_doc_id: str | None, props: dict[str, dict]
    ) -> list[str]:
        """BM25 full-text over chunk text; returns chunk ids ordered by score."""
        ft_query = self._to_fulltext_query(query)
        if not ft_query:
            return []
        try:
            rows = self._db.query(
                """
                CALL db.idx.fulltext.queryNodes('Chunk', $q) YIELD node, score
                RETURN node, score ORDER BY score DESC
                """,
                {"q": ft_query},
                read_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Full-text search failed for %r: %s", ft_query, exc)
            return []
        ids: list[str] = []
        for node, _score in rows:
            data = self._node_props(node)
            if scope_doc_id and data.get("document_id") != scope_doc_id:
                continue
            props.setdefault(data["id"], data)
            ids.append(data["id"])
            if len(ids) >= _PER_METHOD_TOP:
                break
        return ids

    @staticmethod
    def _to_fulltext_query(query: str) -> str:
        """Turn a natural-language query into a RediSearch OR query of terms."""
        terms = [t for t in _FT_STRIP.sub(" ", query).split() if len(t) > 2]
        return " | ".join(terms)

    # ------------------------------------------------------------------ #
    # Search 3 — graph traversal
    # ------------------------------------------------------------------ #
    def _graph_search(
        self, query: str, scope_doc_id: str | None, props: dict[str, dict]
    ) -> tuple[list[str], list[str], list[str], dict[str, int]]:
        """Find query entities, traverse up to 2 hops, collect ranked chunks."""
        entity_names = self._query_entities(query)
        if not entity_names:
            return [], [], [], {}

        edge_pattern = "|".join(_TRAVERSAL_EDGES)
        rows = self._db.query(
            f"""
            UNWIND $names AS qname
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower(qname)
               OR toLower(e.canonical_name) = toLower(qname)
            MATCH path = (e)-[:{edge_pattern}*1..2]-(c:Chunk)
            WITH c, min(length(path)) AS hops
            RETURN c, hops ORDER BY hops ASC
            """,
            {"names": entity_names},
            read_only=True,
        )
        hops_by_chunk: dict[str, int] = {}
        ids: list[str] = []
        edge_types_used = list(_TRAVERSAL_EDGES)
        for node, hops in rows:
            data = self._node_props(node)
            if scope_doc_id and data.get("document_id") != scope_doc_id:
                continue
            cid = data["id"]
            if cid in hops_by_chunk:
                continue
            props.setdefault(cid, data)
            hops_by_chunk[cid] = int(hops)
            ids.append(cid)
        return ids, entity_names, edge_types_used, hops_by_chunk

    def _query_entities(self, query: str) -> list[str]:
        """Extract named entities from the query with spaCy."""
        nlp = self._get_nlp()
        try:
            doc = nlp(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query NER failed: %s", exc)
            return []
        seen: list[str] = []
        for ent in doc.ents:
            text = self._clean_entity(ent.text)
            if text and text.lower() not in {s.lower() for s in seen}:
                seen.append(text)
        return seen

    @staticmethod
    def _clean_entity(text: str) -> str:
        """Strip possessives and surrounding punctuation so names match nodes."""
        text = text.strip().strip("\"'.,;:()[]")
        text = re.sub(r"[’']s$", "", text)  # drop trailing possessive
        return text.strip()

    # ------------------------------------------------------------------ #
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reciprocal_rank_fusion(
        rankings: dict[str, list[str]],
    ) -> list[tuple[str, float, list[str]]]:
        """Fuse per-method ranked id lists into one ranking via RRF."""
        scores: dict[str, float] = {}
        methods: dict[str, list[str]] = {}
        for method, ids in rankings.items():
            for rank, chunk_id in enumerate(ids, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + _RRF_K)
                methods.setdefault(chunk_id, []).append(method)
        fused = [(cid, scores[cid], methods[cid]) for cid in scores]
        fused.sort(key=lambda t: t[1], reverse=True)
        return fused

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _node_props(node) -> dict:
        """Extract a FalkorDB node's properties as a plain dict."""
        return dict(getattr(node, "properties", None) or node)

    @staticmethod
    def _to_chunk(
        data: dict, score: float, methods: list[str], hops: int | None
    ) -> RetrievedChunk:
        """Build a :class:`RetrievedChunk` from node properties + fusion data."""
        return RetrievedChunk(
            chunk_id=data.get("id", ""),
            text=data.get("text", ""),
            element_type=data.get("element_type", ""),
            page_number=int(data.get("page_number", 0) or 0),
            document_id=data.get("document_id", ""),
            document_name=data.get("document_name", ""),
            section_title=data.get("section_title", ""),
            table_json=data.get("table_json") or None,
            image_path=data.get("image_path") or None,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            rrf_score=score,
            methods=methods,
            graph_hops=hops,
        )

    def _get_embedder(self):
        if self._embedder is None:
            from src.retrieval.embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _get_nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(_SPACY_MODEL)
        return self._nlp


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = " ".join(sys.argv[1:]) or "What is Acme Corporation's revenue and who is its CEO?"
    result = HybridRetriever().retrieve(query)
    print(f"\nQuery: {query}")
    print(f"Query entities: {result.query_entities}")
    print(f"Top {len(result.chunks)} chunks:")
    for c in result.chunks:
        print(f"  [{c.rrf_score:.4f}] {c.element_type:9s} p{c.page_number} "
              f"hops={c.graph_hops} via={c.methods} doc={c.document_name} :: {c.text[:54]!r}")
