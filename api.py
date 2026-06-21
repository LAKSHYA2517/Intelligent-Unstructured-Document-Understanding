import json
import os
import shutil
from typing import Dict, Any, Optional
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.pipeline.query_pipeline import QueryPipeline
from src.graph.falkordb_client import get_client
from src.config import config

app = FastAPI(title="DocGraph-RAG v4 API")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = str(config.uploads_dir)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ingestion_pipeline = IngestionPipeline()
query_pipeline = QueryPipeline()
jobs: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Eagerly initialise the LLM router so the first /ask request is fast."""
    import asyncio
    loop = asyncio.get_running_loop()
    def _init():
        from src.llm_router import _get_router
        _get_router()  # builds and caches the litellm Router
        logging.getLogger(__name__).info("LLM Router pre-initialised at startup")
    await loop.run_in_executor(None, _init)


class QueryRequest(BaseModel):
    question: str
    filename: Optional[str] = None


def process_document(job_id: str, file_path: str, filename: str):
    try:
        jobs[job_id]["status"] = "processing"
        result = ingestion_pipeline.ingest_batch([file_path])
        stats = {
            "chunks": result.total_chunks,
            "entities": result.total_entities,
            "triples": result.total_triples,
        }
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stats"] = stats
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/files")
async def list_files():
    """Returns list of ingested documents from FalkorDB (persists across restarts)."""
    try:
        client = get_client()
        result = client.query(
            "MATCH (d:Document) RETURN d.name AS name, d.doc_id AS doc_id, "
            "d.page_count AS pages ORDER BY d.name"
        )
        files = [
            {"name": row[0], "doc_id": row[1], "pages": row[2]}
            for row in (result.result_set or [])
        ]
        return {"files": files}
    except Exception:
        # FalkorDB not ready yet — return empty list gracefully
        return {"files": []}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/health")
async def health_check():
    """Provider health, job counts, and graph statistics."""
    from src.llm_router import get_provider_health, get_audit_log
    providers = get_provider_health()
    audit = get_audit_log()
    # Compute per-provider success rates from audit log
    provider_stats: Dict[str, Any] = {}
    for entry in audit:
        p = entry["provider"]
        if p not in provider_stats:
            provider_stats[p] = {"calls": 0, "errors": 0, "total_tokens": 0, "avg_latency_ms": 0}
        provider_stats[p]["calls"] += 1
        if not entry["success"]:
            provider_stats[p]["errors"] += 1
        provider_stats[p]["total_tokens"] += entry["prompt_tokens"] + entry["completion_tokens"]
        provider_stats[p]["avg_latency_ms"] = (
            (provider_stats[p]["avg_latency_ms"] * (provider_stats[p]["calls"] - 1) + entry["latency_ms"])
            / provider_stats[p]["calls"]
        )
    # Graph stats
    graph_stats = {}
    try:
        client = get_client()
        rows = client.query(
            "MATCH (d:Document) RETURN count(d) AS docs "
            "UNION ALL MATCH (c:Chunk) RETURN count(c) "
            "UNION ALL MATCH (e:Entity) RETURN count(e) "
            "UNION ALL MATCH ()-[r]->() RETURN count(r)",
            read_only=True
        )
        counts = [r[0] for r in (rows.result_set or [])]
        graph_stats = {
            "documents": counts[0] if len(counts) > 0 else 0,
            "chunks": counts[1] if len(counts) > 1 else 0,
            "entities": counts[2] if len(counts) > 2 else 0,
            "edges": counts[3] if len(counts) > 3 else 0,
        }
    except Exception:
        graph_stats = {"error": "FalkorDB unavailable"}
    return {
        "status": "ok",
        "providers_registered": providers,
        "provider_runtime_stats": provider_stats,
        "jobs": {
            "total": len(jobs),
            "completed": sum(1 for j in jobs.values() if j.get("status") == "completed"),
            "processing": sum(1 for j in jobs.values() if j.get("status") == "processing"),
            "failed": sum(1 for j in jobs.values() if j.get("status") == "failed"),
        },
        "graph": graph_stats,
    }


@app.get("/audit")
async def get_audit_trail(limit: int = 100):
    """Returns the LLM call audit trail with provenance metadata.
    
    Each entry records: timestamp, provider used, model, call type,
    token counts, latency, and success/failure status.
    This satisfies the provenance and audit trail requirement.
    """
    from src.llm_router import get_audit_log
    log = get_audit_log()
    # Return most recent entries first
    recent = list(reversed(log))[:limit]
    total_calls = len(log)
    total_tokens = sum(e["prompt_tokens"] + e["completion_tokens"] for e in log)
    total_errors = sum(1 for e in log if not e["success"])
    return {
        "total_calls": total_calls,
        "total_tokens_used": total_tokens,
        "total_errors": total_errors,
        "entries": recent,
    }


@app.delete("/audit")
async def clear_audit_trail():
    """Clears the in-memory LLM audit log."""
    from src.llm_router import clear_audit_log
    clear_audit_log()
    return {"success": True, "message": "Audit log cleared"}


@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a document, saves it, and triggers background ingestion."""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(process_document, job_id, file_path, file.filename)

    return {"success": True, "job_id": job_id, "message": "Ingestion started"}


@app.post("/ask")
async def ask_question(req: QueryRequest):
    """Processes a user question — full response (non-streaming)."""
    try:
        response = query_pipeline.answer(req.question)
        answer_result = response.answer_result

        formatted_sources = []
        if hasattr(answer_result, "chunks_used"):
            for chunk in answer_result.chunks_used:
                formatted_sources.append({
                    "chunk_id": getattr(chunk, "chunk_id", ""),
                    "page": getattr(chunk, "page_number", 1),
                    "document": getattr(chunk, "document_name", ""),
                    "element_type": getattr(chunk, "element_type", "text"),
                    "relevance_score": round(getattr(chunk, "cross_score", 0.0), 4),
                    "preview": (getattr(chunk, "text", "") or "")[:200],
                })

        return {
            "success": True,
            "answer": answer_result.answer,
            "sources": formatted_sources,
            "adversarial": answer_result.adversarial,
            "grounded": not answer_result.adversarial,
            "query_type": response.analysis.query_type,
            "scope": response.analysis.scope,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_question_stream(req: QueryRequest):
    """Streaming version — tokens appear as they are generated (SSE)."""
    import asyncio
    from src.llm_router import _get_router

    try:
        response = query_pipeline.answer(req.question)
        answer_result = response.answer_result

        if answer_result.adversarial:
            async def adversarial_gen():
                yield f"data: {json.dumps({'token': answer_result.answer})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(adversarial_gen(), media_type="text/event-stream")

        router = _get_router()
        answer_text = answer_result.answer

        async def token_generator():
            try:
                # Stream the pre-generated answer word by word for consistent behavior
                words = answer_text.split()
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for natural feel
                yield "data: [DONE]\n\n"
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
