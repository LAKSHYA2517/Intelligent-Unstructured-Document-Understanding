import asyncio
import json
import os
import queue
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from injestion import (
    DEFAULT_CHUNK_SIZE_LIMIT,
    DEFAULT_CONCURRENCY_LIMIT,
    DEFAULT_TEXT_CHUNK_CHARS,
    DEFAULT_TEXT_CHUNK_OVERLAP,
    HybridIndex,
    NvidiaNIMClient,
    answer_query_robust,
    build_hybrid_index,
    get_api_key,
    stream_ingest,
)

load_dotenv()

app = FastAPI(title="Hybrid GraphRAG FastAPI SSE Wrapper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hybrid_index: Optional[HybridIndex] = None
index_lock = asyncio.Lock()

CACHE_MAX_ENTRIES = 128
CACHE_SIMILARITY_THRESHOLD = 0.88
CHUNK_SIZE = 256


@dataclass
class CacheEntry:
    query: str
    result: dict[str, Any]


class SemanticCache:
    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD) -> None:
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    async def get(self, query: str) -> Optional[dict[str, Any]]:
        normalized = self.normalize_text(query)
        async with self._lock:
            if normalized in self.store:
                entry = self.store.pop(normalized)
                self.store[normalized] = entry
                return entry.result

            best_key = None
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


semantic_cache = SemanticCache()


class ChatRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    agentic: bool = False
    evidence_threshold: Optional[float] = None


def sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def resolve_api_key() -> str:
    try:
        return get_api_key()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {"status": "ok", "has_index": hybrid_index is not None}


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


@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    temp_path = Path(tempfile.mktemp(suffix=".pdf"))
    await asyncio.to_thread(temp_path.write_bytes, content)

    api_key = resolve_api_key()
    async with index_lock:
        global hybrid_index
        nim = NvidiaNIMClient(api_key=api_key, concurrency_limit=DEFAULT_CONCURRENCY_LIMIT, status_callback=lambda _: None)
        hybrid_index = await stream_ingest(
            pdf_path=temp_path,
            nim=nim,
            on_event=lambda *args, **kwargs: None,
            source_name=file.filename,
            index=hybrid_index,
            chunk_size_pages=5,
            text_chunk_chars=DEFAULT_TEXT_CHUNK_CHARS,
            text_chunk_overlap=DEFAULT_TEXT_CHUNK_OVERLAP,
        )

    temp_path.unlink(missing_ok=True)
    return {
        "status": "indexed",
        "sources": hybrid_index.source_names(),
        "chunk_count": len(hybrid_index.chunks),
        "node_count": hybrid_index.graph.number_of_nodes(),
        "edge_count": hybrid_index.graph.number_of_edges(),
    }


@app.post("/api/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async with index_lock:
        index = hybrid_index
    if index is None:
        raise HTTPException(status_code=404, detail="No index is available. Ingest a PDF first.")

    cache_hit = await semantic_cache.get(request.query)

    async def event_generator() -> Any:
        yield sse_event("status", {"message": "request_received"})

        if cache_hit is not None:
            yield sse_event("status", {"message": "cache_hit"})
            answer = cache_hit.get("answer", "")
            for chunk in chunk_text(answer):
                yield sse_event("answer", {"text": chunk})
            yield sse_event("done", {"cached": True, "metadata": {k: v for k, v in cache_hit.items() if k != "answer"}})
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
                request.query,
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
        await semantic_cache.set(request.query, result)

        answer_text = result.get("answer", "")
        for chunk in chunk_text(answer_text):
            yield sse_event("answer", {"text": chunk})

        SKIP_KEYS = {"answer", "ranked_chunks", "graph_context"}
        metadata = {k: v for k, v in result.items() if k not in SKIP_KEYS}
        yield sse_event("done", {"cached": False, "metadata": metadata})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
