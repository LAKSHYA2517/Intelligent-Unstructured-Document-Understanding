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
        import ollama

        self._client = ollama.Client(host=config.ollama_base_url)
        self._model = config.ollama_embed_model
        self._dim = config.embedding_dim
        logger.info("Embedder ready (embed model from config: %s, dim=%d)", self._model, self._dim)

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
        """Embed a batch of texts, substituting a placeholder for empty strings."""
        if not texts:
            return []
        # Ollama rejects empty input; embed a single space to keep alignment.
        safe = [t if t.strip() else " " for t in texts]
        try:
            response = self._client.embed(model=self._model, input=safe)
            vectors = getattr(response, "embeddings", None)
            if vectors is None and isinstance(response, dict):
                vectors = response.get("embeddings")
            return [list(v) for v in vectors]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Embedding call failed for {len(texts)} texts with model "
                f"'{self._model}': {exc}"
            ) from exc

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
