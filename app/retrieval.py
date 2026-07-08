"""Production-ready hybrid retrieval helpers for GraphRAG.

This module is intentionally standalone. It consumes existing ChromaDB
collections, NetworkX graphs, ingestion chunks, embedding clients, and optional
BM25 indexes through injection. It does not create persistence layers or change
storage contracts.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

try:  # Optional at import time so injected BM25 indexes keep this module usable.
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - exercised only when dependency absent.
    BM25Okapi = None  # type: ignore[assignment]


RRF_K = 60


@dataclass(frozen=True)
class RetrievalChunk:
    """Adapter-safe representation of an ingestion chunk."""

    chunk_id: str
    content: str
    source: str = ""
    content_type: str = "text"
    title: str = ""
    sequence: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedChunk:
    """Chunk plus Reciprocal Rank Fusion diagnostics."""

    chunk: RetrievalChunk
    fusion_score: float
    fused_rank: int
    semantic_rank: int | None = None
    lexical_rank: int | None = None


@dataclass(frozen=True)
class RetrievalResponse:
    """Return object for LLM context assembly and benchmark logging."""

    retrieved_chunks: list[RetrievalChunk]
    fusion_scores: dict[str, float]
    assembled_context: str
    fused_ranking: list[FusedChunk]


class Embedder(Protocol):
    async def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Return one embedding per input text."""


class VectorCollection(Protocol):
    def query(self, **kwargs: Any) -> Mapping[str, Any]:
        """Chroma-like collection query API."""


class GraphLike(Protocol):
    def has_node(self, node_for_adding: Any) -> bool:
        """Return whether a node exists."""

    def neighbors(self, n: Any) -> Iterable[Any]:
        """Return graph neighbors."""


class BM25Like(Protocol):
    def get_scores(self, query_tokens: list[str]) -> Sequence[float]:
        """Return one lexical score per indexed document."""


def tokenize(text: str) -> list[str]:
    """Deterministic tokenizer for BM25 with no heavy NLP dependencies."""

    return re.findall(r"[a-z0-9]+", text.lower())


def adapt_chunk(value: Any) -> RetrievalChunk:
    """Convert existing chunk objects or dicts to a stable local shape."""

    if isinstance(value, RetrievalChunk):
        return value
    if isinstance(value, Mapping):
        metadata = value.get("metadata", {})
        return RetrievalChunk(
            chunk_id=str(value.get("chunk_id") or value.get("id") or ""),
            content=str(value.get("content") or value.get("document") or value.get("text") or ""),
            source=str(value.get("source") or value.get("doc_id") or ""),
            content_type=str(value.get("content_type") or value.get("type") or "text"),
            title=str(value.get("title") or value.get("header") or ""),
            sequence=int(value.get("sequence") or 0),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
    return RetrievalChunk(
        chunk_id=str(getattr(value, "chunk_id", getattr(value, "id", ""))),
        content=str(getattr(value, "content", getattr(value, "text", ""))),
        source=str(getattr(value, "source", getattr(value, "doc_id", ""))),
        content_type=str(getattr(value, "content_type", getattr(value, "type", "text"))),
        title=str(getattr(value, "title", getattr(value, "header", ""))),
        sequence=int(getattr(value, "sequence", 0) or 0),
        metadata=dict(getattr(value, "metadata", {}) or {}),
    )


class HybridRetrievalService:
    """Semantic + BM25 + RRF retrieval with shallow graph expansion."""

    def __init__(
        self,
        *,
        collection: VectorCollection,
        graph: GraphLike,
        chunks: Sequence[Any],
        embedder: Embedder,
        bm25_index: BM25Like | None = None,
        semantic_top_k: int = 20,
        lexical_top_k: int = 20,
        final_k: int = 6,
        doc_id_field: str = "source",
    ) -> None:
        if semantic_top_k < 1:
            raise ValueError("semantic_top_k must be positive.")
        if lexical_top_k < 1:
            raise ValueError("lexical_top_k must be positive.")
        if final_k < 1:
            raise ValueError("final_k must be positive.")

        adapted = [adapt_chunk(chunk) for chunk in chunks]
        self.collection = collection
        self.graph = graph
        self.embedder = embedder
        self.chunks = adapted
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in adapted if chunk.chunk_id}
        self.semantic_top_k = semantic_top_k
        self.lexical_top_k = lexical_top_k
        self.final_k = final_k
        self.doc_id_field = doc_id_field
        self.bm25_index = bm25_index or self._build_bm25(adapted)

    async def retrieve_and_rerank(self, query: str, doc_id: str) -> RetrievalResponse:
        """Run semantic retrieval, BM25 retrieval, RRF fusion, and graph expansion."""

        clean_query = _clean_query(query)
        clean_doc_id = doc_id.strip()
        if not clean_query:
            return RetrievalResponse([], {}, "", [])

        semantic_task = asyncio.create_task(self._semantic_ranking(clean_query, clean_doc_id))
        lexical_task = asyncio.create_task(asyncio.to_thread(self._lexical_ranking, clean_query, clean_doc_id))
        semantic_ids, lexical_ids = await asyncio.gather(semantic_task, lexical_task)

        fused = self._fuse_rankings(semantic_ids, lexical_ids)
        retrieved = [item.chunk for item in fused[: self.final_k]]
        context = self._assemble_context(fused[0].chunk if fused else None, retrieved)
        return RetrievalResponse(
            retrieved_chunks=retrieved,
            fusion_scores={item.chunk.chunk_id: item.fusion_score for item in fused},
            assembled_context=context,
            fused_ranking=fused,
        )

    async def _semantic_ranking(self, query: str, doc_id: str) -> list[str]:
        embedding = await _embed_query(self.embedder, query)
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": max(1, self.semantic_top_k),
            "include": ["documents", "metadatas", "distances"],
        }
        if doc_id:
            kwargs["where"] = {self.doc_id_field: doc_id}
        response = await asyncio.to_thread(self.collection.query, **kwargs)
        return _extract_chroma_ids(response)

    def _lexical_ranking(self, query: str, doc_id: str) -> list[str]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = list(self.bm25_index.get_scores(query_tokens))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

        ranked_ids: list[str] = []
        for index in ranked_indices:
            if len(ranked_ids) >= self.lexical_top_k:
                break
            if index >= len(self.chunks) or scores[index] <= 0:
                continue
            chunk = self.chunks[index]
            if doc_id and not self._chunk_matches_doc_id(chunk, doc_id):
                continue
            ranked_ids.append(chunk.chunk_id)
        return ranked_ids

    def _chunk_matches_doc_id(self, chunk: RetrievalChunk, doc_id: str) -> bool:
        if chunk.source == doc_id:
            return True
        metadata_value = chunk.metadata.get(self.doc_id_field)
        return metadata_value == doc_id

    def _fuse_rankings(self, semantic_ids: Sequence[str], lexical_ids: Sequence[str]) -> list[FusedChunk]:
        scores: dict[str, float] = {}
        semantic_rank: dict[str, int] = {}
        lexical_rank: dict[str, int] = {}

        for rank, chunk_id in enumerate(_dedupe(semantic_ids), start=1):
            if chunk_id in self.chunks_by_id:
                semantic_rank[chunk_id] = rank
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank))

        for rank, chunk_id in enumerate(_dedupe(lexical_ids), start=1):
            if chunk_id in self.chunks_by_id:
                lexical_rank[chunk_id] = rank
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank))

        ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], semantic_rank.get(chunk_id, 10**9), lexical_rank.get(chunk_id, 10**9), chunk_id))
        return [
            FusedChunk(
                chunk=self.chunks_by_id[chunk_id],
                fusion_score=scores[chunk_id],
                fused_rank=rank,
                semantic_rank=semantic_rank.get(chunk_id),
                lexical_rank=lexical_rank.get(chunk_id),
            )
            for rank, chunk_id in enumerate(ordered_ids, start=1)
        ]

    def _assemble_context(self, top_chunk: RetrievalChunk | None, retrieved: Sequence[RetrievalChunk]) -> str:
        if top_chunk is None:
            return ""

        ordered: list[RetrievalChunk] = []
        seen: set[str] = set()

        def append(chunk: RetrievalChunk) -> None:
            if chunk.chunk_id and chunk.chunk_id not in seen:
                ordered.append(chunk)
                seen.add(chunk.chunk_id)

        for parent in self._parent_headers(top_chunk):
            append(parent)
        append(top_chunk)
        for sibling in self._immediate_siblings(top_chunk):
            append(sibling)
        for chunk in retrieved:
            append(chunk)

        blocks = []
        for index, chunk in enumerate(ordered, start=1):
            title = f" | {chunk.title}" if chunk.title else ""
            blocks.append(
                "\n".join(
                    [
                        f"[{index}] chunk_id={chunk.chunk_id} source={chunk.source} type={chunk.content_type}{title}",
                        chunk.content.strip(),
                    ]
                ).strip()
            )
        return "\n\n---\n\n".join(blocks)

    def _parent_headers(self, chunk: RetrievalChunk) -> list[RetrievalChunk]:
        parent_ids = _metadata_ids(chunk.metadata, ("parent_header_id", "parent_id", "header_id"))
        parent_ids.extend(self._graph_related_ids(chunk.chunk_id, relation_names={"parent", "parent_header", "header", "belongs_to"}))
        parents = [self.chunks_by_id[item_id] for item_id in _dedupe(parent_ids) if item_id in self.chunks_by_id]
        return sorted(parents, key=lambda item: item.sequence)

    def _immediate_siblings(self, chunk: RetrievalChunk) -> list[RetrievalChunk]:
        sibling_ids = _metadata_ids(chunk.metadata, ("previous_sibling_id", "next_sibling_id", "sibling_ids"))
        sibling_ids.extend(self._graph_related_ids(chunk.chunk_id, relation_names={"sibling", "previous_sibling", "next_sibling"}))

        parent_ids = _metadata_ids(chunk.metadata, ("parent_header_id", "parent_id", "header_id"))
        for parent_id in parent_ids:
            sibling_ids.extend(self._graph_related_ids(parent_id, relation_names={"child", "contains"}))

        siblings = [
            self.chunks_by_id[item_id]
            for item_id in _dedupe(sibling_ids)
            if item_id in self.chunks_by_id and item_id != chunk.chunk_id
        ]
        return sorted(siblings, key=lambda item: item.sequence)

    def _graph_related_ids(self, node_id: str, *, relation_names: set[str]) -> list[str]:
        if not node_id or not self.graph.has_node(node_id):
            return []

        related: list[str] = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = _edge_data(self.graph, node_id, neighbor)
            relation = str(edge_data.get("relation", edge_data.get("type", ""))).lower()
            if not relation or relation in relation_names:
                related.append(str(neighbor))

        predecessors = getattr(self.graph, "predecessors", None)
        if callable(predecessors):
            for neighbor in predecessors(node_id):
                edge_data = _edge_data(self.graph, neighbor, node_id)
                relation = str(edge_data.get("relation", edge_data.get("type", ""))).lower()
                if not relation or relation in relation_names:
                    related.append(str(neighbor))
        return related

    def _build_bm25(self, chunks: Sequence[RetrievalChunk]) -> BM25Like:
        if BM25Okapi is None:
            raise RuntimeError("rank_bm25 is required when bm25_index is not injected.")
        corpus = [tokenize(chunk.content) for chunk in chunks]
        return BM25Okapi(corpus)


async def retrieve_and_rerank(
    query: str,
    doc_id: str,
    *,
    collection: VectorCollection,
    graph: GraphLike,
    chunks: Sequence[Any],
    embedder: Embedder,
    bm25_index: BM25Like | None = None,
    semantic_top_k: int = 20,
    lexical_top_k: int = 20,
    final_k: int = 6,
    doc_id_field: str = "source",
) -> RetrievalResponse:
    """Function-parameter injection wrapper around ``HybridRetrievalService``."""

    service = HybridRetrievalService(
        collection=collection,
        graph=graph,
        chunks=chunks,
        embedder=embedder,
        bm25_index=bm25_index,
        semantic_top_k=semantic_top_k,
        lexical_top_k=lexical_top_k,
        final_k=final_k,
        doc_id_field=doc_id_field,
    )
    return await service.retrieve_and_rerank(query, doc_id)


async def _embed_query(embedder: Embedder, query: str) -> list[float]:
    """Call NVIDIA embeddings with input_type='query' when the client supports it."""

    embed_query = getattr(embedder, "embed_query", None)
    if callable(embed_query):
        value = embed_query(query, input_type="query")
        result = await value if inspect.isawaitable(value) else value
        return list(result)

    try:
        value = embedder.embed_texts([query], input_type="query")
        embeddings = await value if inspect.isawaitable(value) else value
    except TypeError:
        value = embedder.embed_texts([query])
        embeddings = await value if inspect.isawaitable(value) else value
    return list(embeddings[0])


def _extract_chroma_ids(response: Mapping[str, Any]) -> list[str]:
    ids = response.get("ids", [])
    if not ids:
        return []
    first = ids[0] if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)) else ids
    return [str(item) for item in first]


def _clean_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _metadata_ids(metadata: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    ids: list[str] = []
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            ids.extend(str(item) for item in value if item)
        else:
            ids.append(str(value))
    return ids


def _edge_data(graph: Any, source: str, target: str) -> Mapping[str, Any]:
    getter = getattr(graph, "get_edge_data", None)
    if not callable(getter):
        return {}
    data = getter(source, target, default={})
    if not isinstance(data, Mapping):
        return {}
    if "relation" in data or "type" in data:
        return data
    for value in data.values():
        if isinstance(value, Mapping):
            return value
    return data
