"""
Hybrid GraphRAG ingestion and demo application.

This file preserves the working Docling + NVIDIA Vision extraction core and
adds:
  - ChromaDB vector storage with NVIDIA NIM embeddings
  - NVIDIA NIM reranking
  - NetworkX knowledge graph construction with deterministic entity resolution
  - Llama 4 Maverick answer synthesis
  - Streamlit demo interface

Run as a CLI:
    python injestion.py --pdf test-5.pdf --build

Run as a Streamlit demo:
    streamlit run injestion.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import networkx as nx
from dotenv import load_dotenv

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    import chromadb
except ImportError:  # pragma: no cover - handled at runtime for clearer UX
    chromadb = None

ChromaCollection = Any
MetadataValue = str | int | float | bool

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError:  # pragma: no cover
    InputFormat = None
    PdfPipelineOptions = None
    TableFormerMode = None
    DocumentConverter = None
    PdfFormatOption = None


load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_PDF_FILENAME = "test-5.pdf"
MD_FILENAME = "parsed_text_clean.md"
REPORT_FILENAME = "nvidia_vision_report.md"
CHROMA_DIR = WORKSPACE / "chroma_store"
IMAGE_DIR = WORKSPACE / "extracted_charts"

DEFAULT_CHUNK_SIZE_LIMIT = 10
DEFAULT_CONCURRENCY_LIMIT = 3
DEFAULT_TEXT_CHUNK_CHARS = 1600
DEFAULT_TEXT_CHUNK_OVERLAP = 200

DEFAULT_NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
DEFAULT_NVIDIA_RERANK_URL = "https://integrate.api.nvidia.com/v1/ranking"


def env_url(name: str, default: str, required_suffix: str) -> str:
    url = os.getenv(name, default).strip().strip('"').strip("'")
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"{name} must be a valid HTTP(S) URL, got: {url}")
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return f"{url}{required_suffix}"
    return url


NVIDIA_CHAT_URL = env_url("NVIDIA_CHAT_URL", DEFAULT_NVIDIA_CHAT_URL, "/chat/completions")
NVIDIA_EMBED_URL = env_url("NVIDIA_EMBED_URL", DEFAULT_NVIDIA_EMBED_URL, "/embeddings")
NVIDIA_RERANK_URL = env_url("NVIDIA_RERANK_URL", DEFAULT_NVIDIA_RERANK_URL, "/ranking")
NVIDIA_EMBED_URLS = [
    NVIDIA_EMBED_URL,
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-embed-v1/embeddings",
]
NVIDIA_RERANK_URLS = [
    NVIDIA_RERANK_URL,
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/rerank-qa-mistral-4b/reranking",
]


VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
EMBED_MODEL = "nv-embed-v1"
RERANK_MODEL = "rerank-qa-mistral-4b"
REASONING_MODEL = "meta/llama-4-maverick-17b-128e-instruct"
GRAPH_EXTRACTION_MODEL = "meta/llama-4-maverick-17b-128e-instruct"


StatusCallback = Callable[[str], None]


def default_status(message: str) -> None:
    print(message)


def get_api_key(explicit_key: str | None = None) -> str:
    """Resolve NVIDIA API key from UI input or .env without leaking it."""
    key = (explicit_key or "").strip() or os.getenv("NVIDIA_API_KEY", "").strip()
    key = key.strip('"').strip("'")
    if not key:
        raise ValueError("NVIDIA_API_KEY is required. Add it in the sidebar or .env file.")
    return key


def ensure_dependency(name: str, module: Any) -> None:
    if module is None:
        raise RuntimeError(
            f"Missing dependency '{name}'. Install project requirements before running this feature."
        )


# =============================================================================
# Data Models
# =============================================================================


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source: str
    content_type: str
    content: str
    title: str = ""
    sequence: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, MetadataValue]:
        meta: dict[str, MetadataValue] = {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "content_type": self.content_type,
            "title": self.title,
            "sequence": self.sequence,
        }
        meta.update({k: str(v) for k, v in self.metadata.items() if v is not None})
        return meta


@dataclass(slots=True)
class HybridIndex:
    chunks: list[DocumentChunk]
    collection: ChromaCollection
    graph: nx.DiGraph
    entity_to_node_id: dict[str, str]
    chart_results: list[dict[str, str]]
    markdown_text: str
    vision_report: str


# =============================================================================
# NVIDIA NIM Client
# =============================================================================


class NvidiaNIMClient:
    """Small defensive wrapper around NVIDIA NIM endpoints used by this demo."""

    def __init__(
        self,
        api_key: str,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        status_callback: StatusCallback = default_status,
    ) -> None:
        self.api_key = api_key
        self.status = status_callback
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.semaphore = asyncio.Semaphore(max(1, concurrency_limit))

    async def post_with_backoff(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        label: str,
        timeout: float = 90.0,
        max_retries: int = 4,
    ) -> dict[str, Any]:
        """Retry transient NVIDIA API failures, tuned for 429 and 502 drops."""
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(
                    url, headers=self.headers, json=payload, timeout=timeout
                )
                if response.status_code == 200:
                    return response.json()

                if response.status_code == 429:
                    wait_time = attempt * 5
                    self.status(
                        f"Rate limited by NVIDIA NIM while processing {label}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code in {500, 502, 503, 504}:
                    wait_time = attempt * 2
                    self.status(
                        f"NVIDIA NIM returned {response.status_code} for {label}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue

                raise RuntimeError(
                    f"NVIDIA NIM request failed for {label}: "
                    f"HTTP {response.status_code} - {response.text[:500]}"
                )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                wait_time = attempt * 2
                self.status(
                    f"Connection dropped while processing {label}: {exc}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"NVIDIA NIM request failed permanently for {label}.")

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: float = 120.0,
        label: str = "chat completion",
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stream": False,
        }
        async with httpx.AsyncClient() as client:
            data = await self.post_with_backoff(
                client, NVIDIA_CHAT_URL, payload, label=label, timeout=timeout
            )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat completion response for {label}: {data}") from exc

    async def embed_texts(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        async with httpx.AsyncClient() as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                payload = {
                    "model": EMBED_MODEL,
                    "input": batch,
                    "encoding_format": "float",
                    "input_type": "passage",
                }
                last_error: Exception | None = None
                for endpoint in NVIDIA_EMBED_URLS:
                    try:
                        data = await self.post_with_backoff(
                            client,
                            endpoint,
                            payload,
                            label=f"embedding batch {start // batch_size + 1}",
                            timeout=120.0,
                        )
                        embeddings.extend(parse_embedding_response(data))
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    raise RuntimeError(
                        f"Embedding batch {start // batch_size + 1} failed: {last_error}"
                    ) from last_error
        return embeddings

    async def rerank(self, query: str, chunks: list[DocumentChunk]) -> list[tuple[DocumentChunk, float]]:
        if not chunks:
            return []

        passages = [{"text": chunk.content} for chunk in chunks]
        payload_variants = [
            {
                "model": RERANK_MODEL,
                "query": {"text": query},
                "passages": passages,
                "truncate": "END",
            },
            {
                "model": RERANK_MODEL,
                "query": query,
                "documents": [chunk.content for chunk in chunks],
            },
        ]

        async with httpx.AsyncClient() as client:
            last_error: Exception | None = None
            for endpoint in NVIDIA_RERANK_URLS:
                for payload in payload_variants:
                    try:
                        data = await self.post_with_backoff(
                            client,
                            endpoint,
                            payload,
                            label="reranking",
                            timeout=90.0,
                            max_retries=3,
                        )
                        scores = parse_rerank_response(data, len(chunks))
                        return sorted(
                            zip(chunks, scores), key=lambda item: item[1], reverse=True
                        )
                    except Exception as exc:  # endpoint schema differs across NIM deployments
                        last_error = exc
            raise RuntimeError(f"Reranking failed: {last_error}")

    async def extract_graph_facts(self, chunk: DocumentChunk) -> dict[str, Any]:
        text = truncate_text(chunk.content, 3200)
        prompt = f"""
Extract a compact knowledge graph from the document segment below.

Return only valid JSON with this schema:
{{
  "entities": [
    {{"name": "Entity Name", "type": "organization|person|metric|concept|date|place|other"}}
  ],
  "relations": [
    {{"source": "Entity Name", "target": "Entity Name", "relation": "short_label"}}
  ]
}}

Rules:
- Use canonical entity names.
- Do not invent entities that are not present or directly implied.
- Prefer business, technical, financial, chart, table, and metric entities.
- Keep relation labels concise, lowercase, and underscore_separated.

Segment title: {chunk.title or chunk.chunk_id}
Segment:
{text}
""".strip()
        raw = await self.chat_completion(
            [{"role": "user", "content": prompt}],
            model=GRAPH_EXTRACTION_MODEL,
            max_tokens=900,
            temperature=0.0,
            label=f"graph extraction {chunk.chunk_id}",
        )
        return parse_json_object(raw)


def parse_embedding_response(data: dict[str, Any]) -> list[list[float]]:
    if "data" in data:
        ordered = sorted(data["data"], key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in ordered]
    if "embeddings" in data:
        return data["embeddings"]
    raise RuntimeError(f"Unexpected embedding response: {data}")


def parse_rerank_response(data: dict[str, Any], expected_count: int) -> list[float]:
    if "rankings" in data:
        scores = [0.0] * expected_count
        for item in data["rankings"]:
            idx = int(item.get("index", item.get("document_index", 0)))
            scores[idx] = float(item.get("logit", item.get("score", item.get("relevance_score", 0.0))))
        return scores

    if "results" in data:
        scores = [0.0] * expected_count
        for item in data["results"]:
            idx = int(item.get("index", item.get("document_index", 0)))
            scores[idx] = float(item.get("relevance_score", item.get("score", 0.0)))
        return scores

    if "scores" in data:
        return [float(score) for score in data["scores"]]

    raise RuntimeError(f"Unexpected rerank response: {data}")


def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {"entities": [], "relations": []}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}


# =============================================================================
# Existing Docling + Vision Core
# =============================================================================


def generate_dynamic_chunks(pdf_path: str | Path, chunk_size: int = DEFAULT_CHUNK_SIZE_LIMIT) -> list[tuple[int, int]]:
    """Read PDF metadata and create memory-safe dynamic page ranges."""
    ensure_dependency("pypdf", PdfReader)
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    chunks = []
    for start in range(1, total_pages + 1, max(1, chunk_size)):
        end = min(start + chunk_size - 1, total_pages)
        chunks.append((start, end))
    return chunks


def run_local_docling_pipeline(
    pdf_path: str | Path = DEFAULT_PDF_FILENAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE_LIMIT,
    output_markdown_path: str | Path = MD_FILENAME,
    image_output_dir: str | Path = IMAGE_DIR,
    status_callback: StatusCallback = default_status,
) -> list[str]:
    """Run Docling text and chart extraction over dynamic page chunks."""
    ensure_dependency("pypdf", PdfReader)
    ensure_dependency("docling", DocumentConverter)
    ensure_dependency("docling", PdfPipelineOptions)
    ensure_dependency("docling", TableFormerMode)
    ensure_dependency("docling", PdfFormatOption)
    ensure_dependency("docling", InputFormat)

    pdf_path = Path(pdf_path)
    output_markdown_path = Path(output_markdown_path)
    image_output_dir = Path(image_output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    image_output_dir.mkdir(parents=True, exist_ok=True)

    status_callback("Initializing Docling layout models...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.generate_picture_images = True
    pipeline_options.do_ocr = False
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    page_chunks = generate_dynamic_chunks(pdf_path, chunk_size=chunk_size)
    status_callback(f"Generated Docling page chunks: {page_chunks}")
    output_markdown_path.write_text("", encoding="utf-8")

    saved_images: list[str] = []
    skipped_count = 0

    for start_page, end_page in page_chunks:
        status_callback(f"Processing pages {start_page} to {end_page} with Docling...")
        doc = converter.convert(str(pdf_path), page_range=(start_page, end_page)).document

        with output_markdown_path.open("a", encoding="utf-8") as md_file:
            md_file.write(doc.export_to_markdown() + "\n\n")

        for index, picture in enumerate(doc.pictures):
            image_obj = picture.get_image(doc)
            if not image_obj:
                continue

            width, height = image_obj.size
            aspect_ratio = width / max(1, height)
            if aspect_ratio <= 1.1:
                skipped_count += 1
                continue

            colors = image_obj.getcolors(maxcolors=250000)
            if colors is None or len(colors) > 15000:
                skipped_count += 1
                continue

            image_path = image_output_dir / (
                f"extracted_graphic_pages_{start_page}_to_{end_page}_img_{index}.png"
            )
            image_obj.save(image_path)
            saved_images.append(str(image_path))
            status_callback(f"Saved chart candidate: {image_path.name}")

        del doc
        gc.collect()

    status_callback(
        f"Docling extraction complete. Saved {len(saved_images)} charts; "
        f"filtered {skipped_count} decorative/photo artifacts."
    )
    return saved_images


def encode_image_to_base64(image_path: str | Path) -> str:
    with Path(image_path).open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def analyze_chart_async(
    nim: NvidiaNIMClient,
    client: httpx.AsyncClient,
    image_path: str,
) -> dict[str, str]:
    """Send an image to NVIDIA NIM VLM with explicit 429/502 backoff handling."""
    async with nim.semaphore:
        base64_image = encode_image_to_base64(image_path)
        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this graph or image in high detail. "
                                "State the axes, the main trends, and extract any "
                                "specific data points highlighted."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 1.0,
            "stream": False,
        }

        try:
            nim.status(f"Uploading chart to NVIDIA Vision model: {Path(image_path).name}")
            response_json = await nim.post_with_backoff(
                client,
                NVIDIA_CHAT_URL,
                payload,
                label=Path(image_path).name,
                timeout=60.0,
                max_retries=4,
            )
            summary = response_json["choices"][0]["message"]["content"]
            return {"file": image_path, "summary": summary}
        except Exception as exc:
            nim.status(f"Vision analysis failed for {Path(image_path).name}: {exc}")
            return {"file": image_path, "summary": f"Vision analysis failed: {exc}"}


async def run_async_vlm_pipeline(
    image_paths: list[str],
    nim: NvidiaNIMClient,
    report_path: str | Path = REPORT_FILENAME,
) -> list[dict[str, str]]:
    """Orchestrate concurrent async VLM calls for all extracted charts."""
    report_path = Path(report_path)
    if not image_paths:
        report_path.write_text("# NVIDIA NIM Vision Extraction Report\n\nNo chart images found.\n", encoding="utf-8")
        return []

    nim.status(f"Sending {len(image_paths)} charts to NVIDIA Vision concurrently...")
    async with httpx.AsyncClient() as client:
        tasks = [analyze_chart_async(nim, client, path) for path in image_paths]
        results = await asyncio.gather(*tasks)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# NVIDIA NIM Vision Extraction Report\n\n")
        for res in results:
            f.write(f"## File: `{res['file']}`\n")
            f.write(f"{res['summary']}\n\n")
            f.write("---\n\n")

    nim.status(f"VLM analysis complete. Saved report to {report_path}.")
    return results


# =============================================================================
# Chunking
# =============================================================================


def stable_chunk_id(source: str, sequence: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}:{sequence}:{content}".encode("utf-8")).hexdigest()[:16]
    return f"chunk_{digest}"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title = "Document"
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+", line):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_lines = []
            current_title = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip() or "Untitled Section"
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(title, body) for title, body in sections if body.strip()]


def sliding_text_windows(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    start = 0
    step_back = max(0, min(overlap, max_chars // 2))
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - step_back)
    return [window for window in windows if window]


def parse_vision_report(report_text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    pattern = re.compile(
        r"^## File:\s*`(?P<file>[^`]+)`\s*\n(?P<body>.*?)(?=^---\s*$|^## File:|\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(report_text):
        filename = match.group("file").strip()
        body = match.group("body").strip()
        if body:
            entries.append((filename, body))
    return entries


def build_document_chunks(
    markdown_path: str | Path = MD_FILENAME,
    report_path: str | Path = REPORT_FILENAME,
    text_chunk_chars: int = DEFAULT_TEXT_CHUNK_CHARS,
    text_chunk_overlap: int = DEFAULT_TEXT_CHUNK_OVERLAP,
) -> tuple[list[DocumentChunk], str, str]:
    markdown_path = Path(markdown_path)
    report_path = Path(report_path)
    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown output not found: {markdown_path}")

    markdown_text = markdown_path.read_text(encoding="utf-8")
    vision_report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    chunks: list[DocumentChunk] = []
    sequence = 0

    for title, body in split_markdown_sections(markdown_text):
        for window in sliding_text_windows(body, max(300, text_chunk_chars), text_chunk_overlap):
            sequence += 1
            content = f"{title}\n\n{window}".strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=stable_chunk_id("parsed_text_clean.md", sequence, content),
                    source="parsed_text_clean.md",
                    content_type="text",
                    content=content,
                    title=title,
                    sequence=sequence,
                )
            )

    for filename, summary in parse_vision_report(vision_report):
        sequence += 1
        content = f"Chart/Image file: {filename}\n\nVision summary:\n{summary}".strip()
        chunks.append(
            DocumentChunk(
                chunk_id=stable_chunk_id("nvidia_vision_report.md", sequence, content),
                source="nvidia_vision_report.md",
                content_type="chart_summary",
                content=content,
                title=Path(filename).name,
                sequence=sequence,
                metadata={"image_file": filename},
            )
        )

    if not chunks:
        raise RuntimeError("No chunks were produced from parsed markdown or vision report.")

    return chunks, markdown_text, vision_report


# =============================================================================
# ChromaDB Vector Layer
# =============================================================================


def init_chroma_collection(
    reset: bool = True,
    persist_dir: str | Path | None = None,
) -> ChromaCollection:
    ensure_dependency("chromadb", chromadb)
    if persist_dir is None:
        persist_dir = CHROMA_DIR
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_name = "hybrid_graphrag_chunks"
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


async def vectorize_chunks(
    chunks: list[DocumentChunk],
    nim: NvidiaNIMClient,
    reset_collection: bool = True,
    status_callback: StatusCallback = default_status,
) -> ChromaCollection:
    collection = init_chroma_collection(reset=reset_collection)
    status_callback(f"Embedding {len(chunks)} chunks with NVIDIA {EMBED_MODEL}...")
    embeddings = await nim.embed_texts([chunk.content for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings)}"
        )

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk.content for chunk in chunks],
        metadatas=[chunk.as_metadata() for chunk in chunks],
    )
    status_callback("ChromaDB vector index is ready.")
    return collection


async def retrieve_and_rerank(
    query: str,
    index: HybridIndex,
    nim: NvidiaNIMClient,
    top_k: int = 10,
    final_k: int = 3,
    min_score: float | None = None,
) -> list[tuple[DocumentChunk, float]]:
    query_embedding = (await nim.embed_texts([query]))[0]
    results = index.collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, len(index.chunks)),
        include=["documents", "metadatas", "distances"],
    )
    ids = results.get("ids", [[]])[0]
    chunk_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    retrieved = [chunk_by_id[item_id] for item_id in ids if item_id in chunk_by_id]
    if not retrieved:
        return []

    ranked = await nim.rerank(query, retrieved)
    if min_score is not None:
        ranked = [item for item in ranked if item[1] >= min_score]
    return ranked[:final_k]


# =============================================================================
# Knowledge Graph Layer
# =============================================================================


def canonical_entity_key(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", " ", name).strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    aliases = {
        "tcs": "tata consultancy services",
        "tata consultancy services ltd": "tata consultancy services",
        "tata consultancy services limited": "tata consultancy services",
    }
    return aliases.get(cleaned, cleaned)


def deterministic_entity_uuid(name: str) -> str:
    canonical = canonical_entity_key(name)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hybrid-graphrag/entity/{canonical}"))


def display_entity_name(name: str) -> str:
    canonical = canonical_entity_key(name)
    if not canonical:
        return "UNKNOWN_ENTITY"
    return canonical.upper()


def safe_relation_label(label: str) -> str:
    label = re.sub(r"[^A-Za-z0-9]+", "_", label.strip().lower()).strip("_")
    return label or "related_to"


async def build_knowledge_graph(
    chunks: list[DocumentChunk],
    nim: NvidiaNIMClient,
    status_callback: StatusCallback = default_status,
) -> tuple[nx.DiGraph, dict[str, str]]:
    graph = nx.DiGraph()
    entity_to_node_id: dict[str, str] = {}

    for chunk in chunks:
        graph.add_node(
            chunk.chunk_id,
            label=chunk.title or chunk.chunk_id,
            node_type="chunk",
            content_type=chunk.content_type,
            source=chunk.source,
            content=truncate_text(chunk.content, 600),
        )

    for chunk in chunks:
        status_callback(f"Extracting entities and relations from {chunk.chunk_id}...")
        facts = await nim.extract_graph_facts(chunk)
        entities = facts.get("entities", [])
        relations = facts.get("relations", [])

        for entity in entities:
            if isinstance(entity, str):
                name = entity
                entity_type = "other"
            else:
                name = str(entity.get("name", "")).strip()
                entity_type = str(entity.get("type", "other")).strip() or "other"
            if not name:
                continue

            canonical = canonical_entity_key(name)
            node_id = deterministic_entity_uuid(name)
            entity_to_node_id[canonical] = node_id
            graph.add_node(
                node_id,
                label=display_entity_name(name),
                canonical=canonical,
                node_type="entity",
                entity_type=entity_type,
            )
            graph.add_edge(chunk.chunk_id, node_id, relation="mentions")

        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            label = safe_relation_label(str(relation.get("relation", "related_to")))
            if not source or not target:
                continue

            source_id = deterministic_entity_uuid(source)
            target_id = deterministic_entity_uuid(target)
            source_key = canonical_entity_key(source)
            target_key = canonical_entity_key(target)
            entity_to_node_id[source_key] = source_id
            entity_to_node_id[target_key] = target_id

            graph.add_node(
                source_id,
                label=display_entity_name(source),
                canonical=source_key,
                node_type="entity",
                entity_type=graph.nodes.get(source_id, {}).get("entity_type", "other"),
            )
            graph.add_node(
                target_id,
                label=display_entity_name(target),
                canonical=target_key,
                node_type="entity",
                entity_type=graph.nodes.get(target_id, {}).get("entity_type", "other"),
            )
            graph.add_edge(source_id, target_id, relation=label)
            graph.add_edge(chunk.chunk_id, source_id, relation="supports_relation")
            graph.add_edge(chunk.chunk_id, target_id, relation="supports_relation")

    status_callback(
        f"NetworkX graph ready: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges."
    )
    return graph, entity_to_node_id


def extract_query_entity_candidates(query: str, entity_to_node_id: dict[str, str]) -> list[str]:
    query_norm = canonical_entity_key(query)
    candidates = []
    for canonical, node_id in entity_to_node_id.items():
        if canonical and (canonical in query_norm or query_norm in canonical):
            candidates.append(node_id)
    return candidates[:8]


def localized_subgraph_context(
    graph: nx.DiGraph,
    seed_chunk_ids: list[str],
    query: str,
    entity_to_node_id: dict[str, str],
    depth: int = 1,
    max_edges: int = 40,
) -> str:
    seeds = set(seed_chunk_ids)
    seeds.update(extract_query_entity_candidates(query, entity_to_node_id))

    visited = set(seeds)
    frontier = set(seeds)
    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            if node not in graph:
                continue
            next_frontier.update(graph.successors(node))
            next_frontier.update(graph.predecessors(node))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier

    lines: list[str] = []
    edge_count = 0
    for source, target, data in graph.edges(data=True):
        if source in visited and target in visited:
            source_label = graph.nodes[source].get("label", source)
            target_label = graph.nodes[target].get("label", target)
            relation = data.get("relation", "related_to")
            lines.append(f"{source_label} --[{relation}]--> {target_label}")
            edge_count += 1
            if edge_count >= max_edges:
                break

    if not lines:
        return "No localized graph context was found for this query."
    return "\n".join(lines)


def graph_nodes_table(graph: nx.DiGraph) -> list[dict[str, Any]]:
    return [
        {
            "node_id": node_id,
            "label": data.get("label", node_id),
            "node_type": data.get("node_type", ""),
            "entity_type": data.get("entity_type", ""),
            "source": data.get("source", ""),
        }
        for node_id, data in graph.nodes(data=True)
    ]


def graph_edges_table(graph: nx.DiGraph) -> list[dict[str, Any]]:
    return [
        {
            "source": graph.nodes[source].get("label", source),
            "relation": data.get("relation", "related_to"),
            "target": graph.nodes[target].get("label", target),
        }
        for source, target, data in graph.edges(data=True)
    ]


# =============================================================================
# Final Hybrid Reasoning Layer
# =============================================================================


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n...[truncated]"


async def answer_query(
    query: str,
    index: HybridIndex,
    nim: NvidiaNIMClient,
    min_rerank_score: float | None = None,
) -> dict[str, Any]:
    ranked_chunks = await retrieve_and_rerank(
        query,
        index,
        nim,
        top_k=10,
        final_k=3,
        min_score=min_rerank_score,
    )
    vector_context = "\n\n".join(
        [
            (
                f"[Chunk {i + 1} | id={chunk.chunk_id} | score={score:.4f} | "
                f"source={chunk.source}]\n{truncate_text(chunk.content, 1800)}"
            )
            for i, (chunk, score) in enumerate(ranked_chunks)
        ]
    )
    subgraph = localized_subgraph_context(
        index.graph,
        seed_chunk_ids=[chunk.chunk_id for chunk, _ in ranked_chunks],
        query=query,
        entity_to_node_id=index.entity_to_node_id,
        depth=1,
    )

    prompt = f"""
You are an enterprise Hybrid GraphRAG reasoning engine.

Answer the user's question using only the supplied vector chunks and knowledge
graph context. Be accurate, concise, and cite chunk IDs when making factual
claims. If the evidence is insufficient, say exactly what is missing instead of
guessing.

User question:
{query}

Top reranked vector chunks:
{vector_context or "No vector chunks passed reranking."}

Localized NetworkX graph context:
{subgraph}

Final answer:
""".strip()

    answer = await nim.chat_completion(
        [{"role": "user", "content": prompt}],
        model=REASONING_MODEL,
        max_tokens=1400,
        temperature=0.05,
        timeout=160.0,
        label="final hybrid answer",
    )
    return {
        "answer": answer,
        "ranked_chunks": ranked_chunks,
        "graph_context": subgraph,
    }


# =============================================================================
# End-to-End Build Orchestration
# =============================================================================


def copy_uploaded_pdf(uploaded_file: Any) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="hybrid_graphrag_pdf_"))
    target = temp_dir / uploaded_file.name
    with target.open("wb") as f:
        f.write(uploaded_file.getbuffer())
    return target


async def build_hybrid_index(
    api_key: str,
    pdf_path: str | Path | None = None,
    chunk_size_pages: int = DEFAULT_CHUNK_SIZE_LIMIT,
    text_chunk_chars: int = DEFAULT_TEXT_CHUNK_CHARS,
    text_chunk_overlap: int = DEFAULT_TEXT_CHUNK_OVERLAP,
    concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    reuse_existing_outputs: bool = False,
    status_callback: StatusCallback = default_status,
) -> HybridIndex:
    nim = NvidiaNIMClient(
        api_key=api_key,
        concurrency_limit=concurrency_limit,
        status_callback=status_callback,
    )

    chart_results: list[dict[str, str]] = []
    if pdf_path is not None and not reuse_existing_outputs:
        status_callback("Docling Parsing")
        if IMAGE_DIR.exists():
            shutil.rmtree(IMAGE_DIR)
        image_paths = run_local_docling_pipeline(
            pdf_path=pdf_path,
            chunk_size=chunk_size_pages,
            output_markdown_path=WORKSPACE / MD_FILENAME,
            image_output_dir=IMAGE_DIR,
            status_callback=status_callback,
        )

        status_callback("VLM Chart Extraction")
        chart_results = await run_async_vlm_pipeline(
            image_paths=image_paths,
            nim=nim,
            report_path=WORKSPACE / REPORT_FILENAME,
        )
    elif not (WORKSPACE / MD_FILENAME).exists():
        raise FileNotFoundError(
            "No PDF was supplied and parsed_text_clean.md does not exist. "
            "Upload a PDF or run the Docling phase first."
        )

    status_callback("Chunking parsed markdown and vision report")
    chunks, markdown_text, vision_report = build_document_chunks(
        markdown_path=WORKSPACE / MD_FILENAME,
        report_path=WORKSPACE / REPORT_FILENAME,
        text_chunk_chars=text_chunk_chars,
        text_chunk_overlap=text_chunk_overlap,
    )

    status_callback("ChromaDB Vectorizing")
    collection = await vectorize_chunks(
        chunks=chunks,
        nim=nim,
        reset_collection=True,
        status_callback=status_callback,
    )

    status_callback("NetworkX Graph Construction")
    graph, entity_to_node_id = await build_knowledge_graph(
        chunks=chunks,
        nim=nim,
        status_callback=status_callback,
    )

    return HybridIndex(
        chunks=chunks,
        collection=collection,
        graph=graph,
        entity_to_node_id=entity_to_node_id,
        chart_results=chart_results,
        markdown_text=markdown_text,
        vision_report=vision_report,
    )


# =============================================================================
# Streamlit Demo
# =============================================================================


def run_streamlit_app() -> None:
    ensure_dependency("streamlit", st)
    ensure_dependency("pandas", pd)

    st.set_page_config(
        page_title="Hybrid GraphRAG Document Intelligence",
        page_icon="",
        layout="wide",
    )
    st.title("Hybrid GraphRAG Document Intelligence")

    with st.sidebar:
        st.header("Configuration")
        env_key_present = bool(os.getenv("NVIDIA_API_KEY"))
        api_key = st.text_input(
            "NVIDIA_API_KEY",
            type="password",
            value="",
            placeholder="Falls back to .env when blank",
        )
        if env_key_present and not api_key:
            st.caption("Using NVIDIA_API_KEY from .env.")

        chunk_size_pages = st.slider("Docling pages per chunk", 1, 25, DEFAULT_CHUNK_SIZE_LIMIT)
        text_chunk_chars = st.slider("Text chunk size", 600, 4000, DEFAULT_TEXT_CHUNK_CHARS, 100)
        concurrency_limit = st.slider("NVIDIA API concurrency ceiling", 1, 8, DEFAULT_CONCURRENCY_LIMIT)
        reuse_existing = st.checkbox(
            "Reuse existing parsed_text_clean.md and nvidia_vision_report.md",
            value=(WORKSPACE / MD_FILENAME).exists(),
        )

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if "hybrid_index" not in st.session_state:
        st.session_state.hybrid_index = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    can_build_from_existing = reuse_existing and (WORKSPACE / MD_FILENAME).exists()
    build_disabled = uploaded_pdf is None and not can_build_from_existing

    if st.button("Build Hybrid RAG Index", type="primary", disabled=build_disabled):
        try:
            resolved_key = get_api_key(api_key)
            pdf_path = copy_uploaded_pdf(uploaded_pdf) if uploaded_pdf is not None else None

            with st.status("Starting pipeline...", expanded=True) as status:
                def ui_status(message: str) -> None:
                    status.write(message)

                index = asyncio.run(
                    build_hybrid_index(
                        api_key=resolved_key,
                        pdf_path=pdf_path,
                        chunk_size_pages=chunk_size_pages,
                        text_chunk_chars=text_chunk_chars,
                        text_chunk_overlap=min(300, text_chunk_chars // 4),
                        concurrency_limit=concurrency_limit,
                        reuse_existing_outputs=bool(pdf_path is None and can_build_from_existing),
                        status_callback=ui_status,
                    )
                )
                st.session_state.hybrid_index = index
                st.session_state.nim_key = resolved_key
                st.session_state.concurrency_limit = concurrency_limit
                status.update(label="Pipeline complete", state="complete")
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

    index: HybridIndex | None = st.session_state.hybrid_index
    if index is None:
        st.info("Upload a PDF or reuse existing parsed outputs, then build the Hybrid RAG index.")
        return

    tab_chat, tab_context, tab_graph = st.tabs(
        ["Interactive Chat", "Document Context", "Graph Metrics"]
    )

    with tab_chat:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("Ask a question about the indexed document")
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving vector chunks, expanding graph context, and reasoning..."):
                    try:
                        nim = NvidiaNIMClient(
                            api_key=st.session_state.nim_key,
                            concurrency_limit=st.session_state.concurrency_limit,
                            status_callback=lambda _: None,
                        )
                        result = asyncio.run(answer_query(user_query, index, nim))
                        st.markdown(result["answer"])
                        with st.expander("Retrieval details"):
                            for chunk, score in result["ranked_chunks"]:
                                st.write(f"{chunk.chunk_id} | score={score:.4f} | {chunk.source}")
                                st.caption(truncate_text(chunk.content, 500))
                            st.code(result["graph_context"])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result["answer"]}
                        )
                    except Exception as exc:
                        error_msg = f"Answer generation failed: {exc}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

    with tab_context:
        left, right = st.columns(2)
        with left:
            st.subheader("Parsed Markdown")
            st.markdown(index.markdown_text or "_No markdown text available._")
        with right:
            st.subheader("Extracted Charts and Vision Descriptions")
            entries = parse_vision_report(index.vision_report)
            if not entries:
                st.info("No chart summaries available.")
            for image_file, summary in entries:
                st.markdown(f"**{Path(image_file).name}**")
                if Path(image_file).exists():
                    st.image(str(image_file), use_container_width=True)
                st.write(summary)

    with tab_graph:
        st.subheader("Graph Density Statistics")
        graph = index.graph
        density = nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
        metrics = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": density,
            "weak_components": nx.number_weakly_connected_components(graph)
            if graph.number_of_nodes()
            else 0,
        }
        st.dataframe(pd.DataFrame([metrics]), use_container_width=True)

        st.subheader("Resolved Nodes")
        st.dataframe(pd.DataFrame(graph_nodes_table(graph)), use_container_width=True)

        st.subheader("Labeled Edges")
        st.dataframe(pd.DataFrame(graph_edges_table(graph)), use_container_width=True)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid GraphRAG ingestion pipeline")
    parser.add_argument("--pdf", default=DEFAULT_PDF_FILENAME, help="PDF path to parse")
    parser.add_argument("--build", action="store_true", help="Build full hybrid index")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip Docling/VLM and build from existing markdown/report outputs",
    )
    parser.add_argument("--page-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_LIMIT)
    parser.add_argument("--text-chunk-chars", type=int, default=DEFAULT_TEXT_CHUNK_CHARS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY_LIMIT)
    parser.add_argument("--query", help="Optional query to run after building the index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = get_api_key()

    index = asyncio.run(
        build_hybrid_index(
            api_key=api_key,
            pdf_path=args.pdf,
            chunk_size_pages=args.page_chunk_size,
            text_chunk_chars=args.text_chunk_chars,
            concurrency_limit=args.concurrency,
            reuse_existing_outputs=args.reuse_existing,
            status_callback=default_status,
        )
    )

    if args.query:
        nim = NvidiaNIMClient(api_key=api_key, concurrency_limit=args.concurrency)
        result = asyncio.run(answer_query(args.query, index, nim))
        print("\nFinal answer:\n")
        print(result["answer"])


if __name__ == "__main__":
    if st is not None:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx

            if get_script_run_ctx(suppress_warning=True) is not None:
                run_streamlit_app()
            else:
                main()
        except Exception:
            main()
    else:
        main()
