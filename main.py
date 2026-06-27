import asyncio
import json
import os
import pickle
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
from fastapi.staticfiles import StaticFiles
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

# ─── FIX 1: Doc Store — Save/Load Index to Disk ──────────────
INDEX_STORE_PATH = Path("index_store.pkl")

def save_index(index: HybridIndex) -> None:
    try:
        with open(INDEX_STORE_PATH, "wb") as f:
            pickle.dump(index, f)
        print("✅ Index saved to disk")
    except Exception as e:
        print(f"❌ Could not save index: {e}")

def load_index() -> Optional[HybridIndex]:
    try:
        if INDEX_STORE_PATH.exists():
            with open(INDEX_STORE_PATH, "rb") as f:
                index = pickle.load(f)
            print("✅ Index loaded from disk")
            return index
    except Exception as e:
        print(f"❌ Could not load index: {e}")
    return None

# Load index on startup automatically
hybrid_index: Optional[HybridIndex] = load_index()
index_lock = asyncio.Lock()

# ─── FIX 4+6: Improved Constants ─────────────────────────────
CACHE_MAX_ENTRIES = 128
CACHE_SIMILARITY_THRESHOLD = 0.82
CHUNK_SIZE = 8


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

    async def clear(self) -> None:
        async with self._lock:
            self.store.clear()


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


# ─── FIX 5: Query Preprocessing ──────────────────────────────
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


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {"status": "ok", "has_index": hybrid_index is not None}

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


# ─── FIX 2: New Chat Reset Endpoint ──────────────────────────
@app.post("/api/chat/reset")
async def reset_chat() -> dict[str, Any]:
    await semantic_cache.clear()
    return {
        "status": "ok",
        "message": "Chat history and cache cleared successfully"
    }


# ─── FIX 3: Clear Index / Fresh Start ────────────────────────
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

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    # FIX 3: Skip if already indexed — no 15-20 min reprocess
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

    api_key = resolve_api_key()
    async with index_lock:
        nim = NvidiaNIMClient(
            api_key=api_key,
            concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
            status_callback=lambda _: None
        )
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

    # FIX 1: Save index to disk after processing
    save_index(hybrid_index)
    temp_path.unlink(missing_ok=True)

    return {
        "status": "indexed",
        "sources": hybrid_index.source_names(),
        "chunk_count": len(hybrid_index.chunks),
        "node_count": hybrid_index.graph.number_of_nodes(),
        "edge_count": hybrid_index.graph.number_of_edges(),
    }


@app.post("/api/ingest/stream")
async def ingest_pdf_stream(file: UploadFile = File(...)) -> StreamingResponse:
    global hybrid_index

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    # FIX 3: Skip if already indexed in stream endpoint too
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
        api_key = resolve_api_key()
        event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

        def on_event(kind: str, **info: Any) -> None:
            event_queue.put_nowait((kind, info))

        async def run_ingest() -> HybridIndex:
            global hybrid_index
            async with index_lock:
                nim = NvidiaNIMClient(
                    api_key=api_key,
                    concurrency_limit=DEFAULT_CONCURRENCY_LIMIT,
                    status_callback=lambda message: on_event("status", message=message),
                )
                hybrid_index = await stream_ingest(
                    pdf_path=temp_path,
                    nim=nim,
                    on_event=on_event,
                    source_name=source_name,
                    index=hybrid_index,
                    chunk_size_pages=5,
                    text_chunk_chars=DEFAULT_TEXT_CHUNK_CHARS,
                    text_chunk_overlap=DEFAULT_TEXT_CHUNK_OVERLAP,
                )
                # FIX 1: Save index to disk after streaming ingest
                save_index(hybrid_index)
                return hybrid_index

        yield sse_event("progress", {"stage": "Uploading", "progress": 8, "source": source_name})
        yield sse_event("progress", {"stage": "Parsing", "progress": 18, "source": source_name})
        task = asyncio.create_task(run_ingest())
        last_stage = "Parsing"
        emitted_stages: set[str] = {"Uploading", "Parsing"}

        try:
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


@app.post("/api/chat")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    # FIX 5: Preprocess query
    request.query = preprocess_query(request.query)

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
                await asyncio.sleep(0.01)
            yield sse_event("done", {
                "cached": True,
                "metadata": {k: v for k, v in cache_hit.items() if k != "answer"}
            })
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
            await asyncio.sleep(0.01)  # FIX 6: Smooth streaming

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

print("KEY PREFIX =", os.getenv("NVIDIA_API_KEY", "")[:15])
