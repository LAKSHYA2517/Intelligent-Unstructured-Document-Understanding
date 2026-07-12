from __future__ import annotations

import asyncio
import json
import os
import pickle
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from injestion import (
    DEFAULT_CONCURRENCY_LIMIT,
    DEFAULT_TEXT_CHUNK_CHARS,
    DEFAULT_TEXT_CHUNK_OVERLAP,
    HybridIndex,
    NvidiaNIMClient,
    answer_query_robust,
    get_api_key,
    stream_ingest,
)
from pii import PiiMasker
from retrieval import HybridRetrievalService


load_dotenv()

app = FastAPI(title="Hybrid GraphRAG FastAPI SSE Wrapper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://intelligent-unstructured-document-u.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploads_dir = Path("backend/uploads")
if uploads_dir.exists():
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


# --- Index Save/Load (survives Space restarts) ---
INDEX_STORE_PATH = Path("index_store.pkl")

def save_index(index: HybridIndex) -> None:
    try:
        with open(INDEX_STORE_PATH, "wb") as f:
            pickle.dump(index, f)
        print("[index] Saved to disk:", INDEX_STORE_PATH)
    except Exception as e:
        print(f"[index] Could not save to disk: {e}")

def load_index() -> Optional[HybridIndex]:
    try:
        if INDEX_STORE_PATH.exists():
            with open(INDEX_STORE_PATH, "rb") as f:
                index = pickle.load(f)
            print("[index] Loaded from disk:", INDEX_STORE_PATH)
            return index
    except Exception as e:
        print(f"[index] Could not load from disk (will start fresh): {e}")
        # Remove corrupt file so it does not block future saves
        try:
            INDEX_STORE_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    return None

# Load index on startup automatically - prevents re-ingestion after Space restarts
hybrid_index: Optional[HybridIndex] = load_index()

index_lock = asyncio.Lock()
pii_masker = PiiMasker()

CACHE_MAX_ENTRIES = 128
CACHE_SIMILARITY_THRESHOLD = 0.82
CHUNK_SIZE = 8


@dataclass
class CacheEntry:
    query: str
    result: dict[str, Any]


class SemanticCache:
    def __init__(
        self,
        max_entries: int = CACHE_MAX_ENTRIES,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
    ) -> None:
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(text.lower().strip().split())

    @staticmethod
    def similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()

    async def get(self, query: str) -> Optional[dict[str, Any]]:
        normalized = self.normalize_text(query)
        async with self._lock:
            if normalized in self.store:
                entry = self.store.pop(normalized)
                self.store[normalized] = entry
                return entry.result

            best_key: str | None = None
            best_score = 0.0
            for key in self.store.keys():
                score = self.similarity(normalized, key)
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key is not None and best_score >= self.similarity_threshold:
                entry = self.store.pop(best_key)
                self.store[best_key] = entry
                return entry.result
        return None

    async def set(self, query: str, result: dict[str, Any]) -> None:
        normalized = self.normalize_text(query)
        async with self._lock:
            if normalized in self.store:
                self.store.pop(normalized)
            elif len(self.store) >= self.max_entries:
                self.store.popitem(last=False)
            self.store[normalized] = CacheEntry(query=query, result=result)

    async def clear(self) -> None:
        async with self._lock:
            self.store.clear()


semantic_cache = SemanticCache()


class ChatRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    agentic: bool = False
    evidence_threshold: Optional[float] = None
    mask_pii: bool = True


class RetrieveRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    semantic_top_k: int = 20
    lexical_top_k: int = 20
    final_k: int = 6
    mask_pii: bool = True


def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def preprocess_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r"[^\w\s\?\.\,\'\-]", "", query)
    if query:
        query = query[0].upper() + query[1:]
    return query


def resolve_api_key() -> str:
    try:
        return get_api_key()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def prepare_query(query: str, mask_pii: bool) -> tuple[str, dict[str, str]]:
    if not mask_pii:
        return query, {}
    return pii_masker.mask(query)


def default_doc_id(index: HybridIndex, requested_doc_id: str | None) -> str:
    if requested_doc_id:
        return requested_doc_id
    sources = index.source_names()
    return sources[0] if sources else ""


def serialize_chunk(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": getattr(chunk, "chunk_id", ""),
        "source": getattr(chunk, "source", ""),
        "content_type": getattr(chunk, "content_type", ""),
        "title": getattr(chunk, "title", ""),
        "sequence": getattr(chunk, "sequence", 0),
        "content": getattr(chunk, "content", ""),
        "metadata": getattr(chunk, "metadata", {}),
    }


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "has_index": hybrid_index is not None,
        "modules": {
            "pii": True,
            "retrieval": True,
            "chunking": True,
            "eval": True,
        },
    }

@app.get("/api/sources")
async def get_sources() -> dict[str, Any]:
    if hybrid_index is None:
        return {"sources": []}
    return {
        "sources": hybrid_index.source_names(),
        "chunk_count": len(hybrid_index.chunks)
    }


@app.get("/api/index")
async def get_index_status() -> dict[str, Any]:
    if hybrid_index is None:
        raise HTTPException(status_code=404, detail="No index has been built yet.")
    return {
        "sources": hybrid_index.source_names(),
        "chunks": len(hybrid_index.chunks),
        "nodes": hybrid_index.graph.number_of_nodes(),
        "edges": hybrid_index.graph.number_of_edges(),
    }


@app.post("/api/chat/reset")
async def reset_chat() -> dict[str, Any]:
    await semantic_cache.clear()
    return {
        "status": "ok",
        "message": "Chat history and cache cleared successfully"
    }


@app.post("/api/index/reset")
async def reset_index() -> dict[str, Any]:
    global hybrid_index
    async with index_lock:
        hybrid_index = None
    if INDEX_STORE_PATH.exists():
        INDEX_STORE_PATH.unlink()
    await semantic_cache.clear()
    return {
        "status": "ok",
        "message": "Index and cache cleared. You can now upload a fresh PDF."
    }


@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    global hybrid_index

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    async with index_lock:
        if hybrid_index is not None and file.filename in hybrid_index.source_names():
            return {
                "status": "already_indexed",
                "message": f"'{file.filename}' is already processed. No reprocessing needed.",
                "sources": hybrid_index.source_names(),
                "chunk_count": len(hybrid_index.chunks),
                "node_count": hybrid_index.graph.number_of_nodes(),
                "edge_count": hybrid_index.graph.number_of_edges(),
            }

    temp_path = Path(tempfile.mktemp(suffix=".pdf"))
    await asyncio.to_thread(temp_path.write_bytes, content)

    try:
        api_key = resolve_api_key()
        async with index_lock:
            nim = NvidiaNIMClient(
                api_key=api_key,
                concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
                status_callback=lambda _: None,
            )
            hybrid_index = await stream_ingest(
                pdf_path=temp_path,
                nim=nim,
                on_event=lambda **_: None,
                source_name=file.filename,
                index=hybrid_index,
                chunk_size_pages=5,
                text_chunk_chars=DEFAULT_TEXT_CHUNK_CHARS,
                text_chunk_overlap=DEFAULT_TEXT_CHUNK_OVERLAP,
            )
        return {
            "status": "indexed",
            "sources": hybrid_index.source_names(),
            "chunk_count": len(hybrid_index.chunks),
            "node_count": hybrid_index.graph.number_of_nodes(),
            "edge_count": hybrid_index.graph.number_of_edges(),
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/ingest/stream")
async def ingest_pdf_stream(file: UploadFile = File(...)) -> StreamingResponse:
    global hybrid_index

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    async with index_lock:
        if hybrid_index is not None and file.filename in hybrid_index.source_names():
            async def already_indexed_generator() -> Any:
                yield sse_event("progress", {
                    "stage": "Already Indexed",
                    "progress": 100,
                    "source": file.filename
                })
                yield sse_event("done", {
                    "status": "already_indexed",
                    "message": f"'{file.filename}' is already processed.",
                    "sources": hybrid_index.source_names(),
                    "chunk_count": len(hybrid_index.chunks),
                })
            return StreamingResponse(already_indexed_generator(), media_type="text/event-stream")

    temp_path = Path(tempfile.mktemp(suffix=".pdf"))
    await asyncio.to_thread(temp_path.write_bytes, content)
    source_name = file.filename

    async def event_generator() -> Any:
        global hybrid_index
        event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        emitted_stages: set[str] = set()
        last_stage: str = "Ingestion"

        def on_event(kind: str, **info: Any) -> None:
            event_queue.put_nowait((kind, info))

        try:
            api_key = resolve_api_key()
            async with index_lock:
                nim = NvidiaNIMClient(
                    api_key=api_key,
                    concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
                    status_callback=lambda message: on_event("status", message=message),
                )
                task = asyncio.create_task(
                    stream_ingest(
                        pdf_path=temp_path,
                        nim=nim,
                        on_event=on_event,
                        source_name=source_name,
                        index=hybrid_index,
                        chunk_size_pages=5,
                        text_chunk_chars=DEFAULT_TEXT_CHUNK_CHARS,
                        text_chunk_overlap=DEFAULT_TEXT_CHUNK_OVERLAP,
                    )
                )

            while not task.done() or not event_queue.empty():
                try:
                    kind, info = await asyncio.wait_for(event_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue

                if kind == "indexed":
                    if "OCR" not in emitted_stages:
                        emitted_stages.add("OCR")
                        yield sse_event("progress", {
                            "stage": "OCR",
                            "progress": 30,
                            "source": source_name,
                            "message": "OCR check complete",
                        })
                    last_stage = "Embedding Generation"
                    emitted_stages.add(last_stage)
                    yield sse_event("progress", {
                        "stage": "Embedding Generation",
                        "progress": 45,
                        "source": source_name,
                        **info,
                    })
                elif kind == "graph_updated":
                    if "Entity Extraction" not in emitted_stages:
                        emitted_stages.add("Entity Extraction")
                        yield sse_event("progress", {
                            "stage": "Entity Extraction",
                            "progress": 62,
                            "source": source_name,
                            **info,
                        })
                    last_stage = "Knowledge Graph Construction"
                    emitted_stages.add(last_stage)
                    yield sse_event("progress", {
                        "stage": "Knowledge Graph Construction",
                        "progress": 72,
                        "source": source_name,
                        **info,
                    })
                elif kind == "status":
                    message = str(info.get("message", ""))
                    if "ocr" in message.lower():
                        last_stage = "OCR"
                    elif "entity" in message.lower():
                        last_stage = "Entity Extraction"
                    emitted_stages.add(last_stage)
                    yield sse_event("progress", {
                        "stage": last_stage,
                        "progress": 55,
                        "source": source_name,
                        **info
                    })
                elif kind == "done":
                    emitted_stages.add("Ready")
                    yield sse_event("progress", {
                        "stage": "Ready",
                        "progress": 100,
                        "source": source_name,
                        **info,
                    })

            index = await task
            hybrid_index = index
            save_index(hybrid_index)  # persist to disk so restarts don't lose the index
            yield sse_event("done", {
                "status": "indexed",
                "sources": index.source_names(),
                "chunk_count": len(index.chunks),
                "node_count": index.graph.number_of_nodes(),
                "edge_count": index.graph.number_of_edges(),
            })
        except Exception as exc:
            yield sse_event("error", {
                "message": str(exc),
                "stage": last_stage,
                "source": source_name
            })
        finally:
            temp_path.unlink(missing_ok=True)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/retrieve")
async def retrieve_debug(request: RetrieveRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async with index_lock:
        index = hybrid_index
    if index is None:
        raise HTTPException(status_code=404, detail="No index is available. Ingest a PDF first.")

    sanitized_query, pii_audit = prepare_query(request.query, request.mask_pii)
    api_key = resolve_api_key()
    nim = NvidiaNIMClient(
        api_key=api_key,
        concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
        status_callback=lambda _: None,
    )
    service = HybridRetrievalService(
        collection=index.collection,
        graph=index.graph,
        chunks=index.chunks,
        embedder=nim,
        semantic_top_k=request.semantic_top_k,
        lexical_top_k=request.lexical_top_k,
        final_k=request.final_k,
    )
    response = await service.retrieve_and_rerank(
        sanitized_query,
        default_doc_id(index, request.doc_id),
    )
    return {
        "query": sanitized_query,
        "pii_audit": pii_audit,
        "retrieved_chunks": [serialize_chunk(chunk) for chunk in response.retrieved_chunks],
        "fusion_scores": response.fusion_scores,
        "assembled_context": response.assembled_context,
        "fused_ranking": [
            {
                "chunk_id": item.chunk.chunk_id,
                "fusion_score": item.fusion_score,
                "fused_rank": item.fused_rank,
                "semantic_rank": item.semantic_rank,
                "lexical_rank": item.lexical_rank,
            }
            for item in response.fused_ranking
        ],
    }


@app.post("/api/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    sanitized_query, pii_audit = prepare_query(request.query, request.mask_pii)

    async with index_lock:
        index = hybrid_index
    if index is None:
        raise HTTPException(status_code=404, detail="No index is available. Ingest a PDF first.")

    cache_hit = await semantic_cache.get(sanitized_query)

    async def event_generator() -> Any:
        yield sse_event("status", {"message": "request_received"})
        if pii_audit:
            yield sse_event("status", {"message": "pii_masked", "mapping": pii_audit})

        if cache_hit is not None:
            yield sse_event("status", {"message": "cache_hit"})
            answer = cache_hit.get("answer", "")
            for chunk in chunk_text(answer):
                yield sse_event("answer", {"text": chunk})
            metadata = {key: value for key, value in cache_hit.items() if key not in ("answer", "ranked_chunks")}
            metadata["pii_audit"] = pii_audit
            yield sse_event("done", {"cached": True, "metadata": metadata})
            return

        yield sse_event("status", {"message": "cache_miss"})
        yield sse_event("status", {"message": "starting_generation"})

        status_messages: list[dict[str, Any]] = []

        def status_callback(message: str) -> None:
            status_messages.append({"type": "status", "message": message})

        api_key = resolve_api_key()
        nim = NvidiaNIMClient(
            api_key=api_key,
            concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
            status_callback=status_callback,
        )

        answer_task = asyncio.create_task(
            answer_query_robust(
                sanitized_query,
                index,
                nim,
                source_filter=request.source_filter,
                agentic=request.agentic,
                evidence_threshold=request.evidence_threshold,
            )
        )

        sent_statuses = 0
        while not answer_task.done():
            while sent_statuses < len(status_messages):
                event = status_messages[sent_statuses]
                yield sse_event(event["type"], {"message": event["message"]})
                sent_statuses += 1
            await asyncio.sleep(0.05)

        try:
            result = await answer_task
        except Exception as exc:
            while sent_statuses < len(status_messages):
                event = status_messages[sent_statuses]
                yield sse_event(event["type"], {"message": event["message"]})
                sent_statuses += 1
            yield sse_event("error", {"message": str(exc)})
            yield sse_event("done", {"error": True})
            return

        while sent_statuses < len(status_messages):
            event = status_messages[sent_statuses]
            yield sse_event(event["type"], {"message": event["message"]})
            sent_statuses += 1

        result["pii_audit"] = pii_audit
        await semantic_cache.set(sanitized_query, result)

        answer_text = result.get("answer", "")
        for chunk in chunk_text(answer_text):
            yield sse_event("answer", {"text": chunk})
            await asyncio.sleep(0.01)

        yield sse_event("done", {"cached": False, "metadata": {k: v for k, v in result.items() if k not in ("answer", "ranked_chunks")}})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

print("KEY PREFIX =", os.getenv("NVIDIA_API_KEY", "")[:15])
