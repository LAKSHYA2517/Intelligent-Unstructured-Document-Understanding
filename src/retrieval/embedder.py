"""Stage 6 — chunk embedding via Ollama.

Embeds every chunk's text with the configured embedding model
(``config.OLLAMA_EMBED_MODEL``) into fixed-dimension vectors and attaches each
vector to its :class:`~src.ingestion.chunker.Chunk`. The very same model and
method are reused at query time (:meth:`Embedder.embed_query`) so ingestion and
retrieval share one vector space — a hard requirement for the HNSW index.

The model name and expected dimensionality both come from config; nothing here
is hardcoded, keeping the stage model-agnostic.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from src.config import config
from src.ingestion.chunker import ChunkResult

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32


@dataclass
class EmbedResult:
    """Stage 6 output summary (the vectors live on the chunks themselves)."""

    embedded_count: int
    dim: int
    model: str


class Embedder:
    """Embeds chunk text (and queries) with the configured Ollama model."""

    def __init__(self) -> None:
        # Attempt to use a cloud embedding provider if configured, otherwise
        # fall back to the local Ollama embedder. This keeps the Embedder a
        # drop-in replacement while enabling scale-up via cloud providers.
        self._cloud_provider = os.getenv("EMBED_PROVIDER", "").lower()
        self._client = None
        self._model = ""
        self._dim = config.embedding_dim
        
        if self._cloud_provider == "gemini":
            # Use Gemini text-embedding-3-large via google.generativeai
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self._client = genai
                self._model = "models/embedding-001"
                logger.info("Embedder configured to use Gemini cloud embeddings")
            except Exception as e:
                logger.warning("Gemini embeddings unavailable: %s - falling back to Ollama", str(e))
                self._cloud_provider = ""
                
        elif self._cloud_provider == "voyage":
            # Use Voyage AI embeddings
            try:
                import voyageai
                self._client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
                self._model = "voyage-2"
                logger.info("Embedder configured to use Voyage AI cloud embeddings")
            except Exception as e:
                logger.warning("Voyage AI embeddings unavailable: %s - falling back to Ollama", str(e))
                self._cloud_provider = ""
                
        elif self._cloud_provider == "openai":
            # Use OpenAI embeddings
            try:
                import openai
                self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self._model = "text-embedding-3-small"
                logger.info("Embedder configured to use OpenAI cloud embeddings")
            except Exception as e:
                logger.warning("OpenAI embeddings unavailable: %s - falling back to Ollama", str(e))
                self._cloud_provider = ""

        if not self._cloud_provider:
            try:
                import ollama
                self._client = ollama.Client(host=config.ollama_base_url)
                self._model = config.ollama_embed_model
                logger.info("Embedder using local Ollama model: %s", self._model)
            except Exception as e:
                logger.error("Ollama client unavailable: %s", str(e))
                raise RuntimeError("No embedding provider available") from e

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed(self, chunk_result: ChunkResult) -> EmbedResult:
        """Embed every chunk in place and return a summary.

        Args:
            chunk_result: Stage 3 output; each chunk's ``embedding`` is set.

        Returns:
            An :class:`EmbedResult` with the real embedded count and dimension.

        Raises:
            ValueError: If the model returns vectors of an unexpected dimension.
        """
        chunks = chunk_result.chunks
        embedded = 0
        for start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[start : start + _BATCH_SIZE]
            vectors = self._embed_texts([c.text for c in batch])
            for chunk, vector in zip(batch, vectors):
                self._validate_dim(vector)
                chunk.embedding = vector
                embedded += 1
        logger.info("Embedded %d chunks with %s (dim=%d)", embedded, self._model, self._dim)
        return EmbedResult(embedded_count=embedded, dim=self._dim, model=self._model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed an arbitrary list of strings (e.g. entity names for resolution).

        Args:
            texts: Strings to embed.

        Returns:
            One validated vector per input, in order.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            for vector in self._embed_texts(texts[start : start + _BATCH_SIZE]):
                self._validate_dim(vector)
                vectors.append(vector)
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string with the same model used at ingestion.

        Args:
            query: The natural-language query.

        Returns:
            The query's embedding vector.
        """
        vector = self._embed_texts([query])[0]
        self._validate_dim(vector)
        return vector

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the configured provider."""
        if not texts:
            return []
            
        # Handle empty strings
        safe = [t if t.strip() else " " for t in texts]
        
        try:
            if self._cloud_provider == "gemini":
                response = self._client.embed_content(
                    model=self._model,
                    content=safe,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                return [list(embedding["embedding"]) for embedding in response["embedding"]]
                
            elif self._cloud_provider == "voyage":
                response = self._client.embed(safe, model=self._model)
                return [list(embedding) for embedding in response.embeddings]
                
            elif self._cloud_provider == "openai":
                response = self._client.embeddings.create(
                    model=self._model,
                    input=safe
                )
                return [list(embedding.embedding) for embedding in response.data]
                
            else:  # Ollama fallback
                response = self._client.embed(model=self._model, input=safe)
                vectors = getattr(response, "embeddings", None)
                if vectors is None and isinstance(response, dict):
                    vectors = response.get("embeddings")
                return [list(v) for v in vectors]
                
        except Exception as exc:
            # Retry once after a short delay
            logger.warning("Embedding failed, retrying: %s", str(exc))
            time.sleep(1)
            try:
                if self._cloud_provider == "gemini":
                    response = self._client.embed_content(
                        model=self._model,
                        content=safe,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    return [list(embedding["embedding"]) for embedding in response["embedding"]]
                # Other providers similar...
            except Exception as retry_exc:
                raise RuntimeError(
                    f"Embedding call failed after retry for {len(texts)} texts: {str(retry_exc)}"
                ) from retry_exc

    def _validate_dim(self, vector: list[float]) -> None:
        """Raise if a vector's length does not match the configured dimension."""
        if len(vector) != self._dim:
            raise ValueError(
                f"Embedding model '{self._model}' returned dim {len(vector)}, "
                f"but config.embedding_dim is {self._dim}. Align EMBEDDING dim or model."
            )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m src.retrieval.embedder <file.pdf|file.docx>")
        raise SystemExit(1)
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser

    pr = DocumentParser().parse(sys.argv[1])
    cr = Chunker().chunk(pr)
    res = Embedder().embed(cr)
    print(f"\nembedded={res.embedded_count} dim={res.dim} model={res.model}")
    print(f"chunk[0] embedding length: {len(cr.chunks[0].embedding)}")
    print(f"all chunks embedded: {all(c.embedding is not None for c in cr.chunks)}")
