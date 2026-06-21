"""Thin, resilient wrapper around a single FalkorDB graph.

This module owns the one connection the whole pipeline shares, the schema
indexes (HNSW vector index on ``Chunk.embedding`` and a BM25 full-text index on
``Chunk.text``), and a handful of safe primitives — parameterised query
execution, idempotent index creation, a graph wipe, and a self-contained smoke
test. Everything is model- and domain-agnostic; only graph mechanics live here.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from falkordb import FalkorDB, Graph

from src.config import config

logger = logging.getLogger(__name__)

# Graph name is a mechanical detail (not a model/domain choice). Default is
# sensible and overridable via the optional FALKORDB_GRAPH env var.
_GRAPH_NAME = os.getenv("FALKORDB_GRAPH", "document_graph")

# Substrings FalkorDB uses when an index/attribute already exists. Creating an
# index twice is an error we treat as success (idempotency).
_ALREADY_EXISTS_MARKERS = ("already indexed", "already exists", "attribute is already")


class FalkorDBError(RuntimeError):
    """Raised when a FalkorDB operation fails irrecoverably."""


class FalkorDBClient:
    """Owns a connection to one FalkorDB graph and its schema indexes.

    Args:
        graph_name: Name of the graph to operate on. Defaults to the
            ``FALKORDB_GRAPH`` env var or ``"document_graph"``.
        host: FalkorDB host. Defaults to the configured value.
        port: FalkorDB port. Defaults to the configured value.
    """

    def __init__(
        self,
        graph_name: str = _GRAPH_NAME,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.graph_name = graph_name
        self.host = host or config.falkordb_host
        self.port = port or config.falkordb_port
        try:
            self._db: FalkorDB = FalkorDB(host=self.host, port=self.port)
            self._db.connection.ping()
        except Exception as exc:  # noqa: BLE001
            raise FalkorDBError(
                f"Could not connect to FalkorDB at {self.host}:{self.port} ({exc}). "
                f"Is the container running?"
            ) from exc
        self._graph: Graph = self._db.select_graph(graph_name)
        logger.info("Connected to FalkorDB graph '%s' at %s:%s", graph_name, self.host, self.port)

    # ------------------------------------------------------------------ #
    # Core query primitive
    # ------------------------------------------------------------------ #
    @property
    def graph(self) -> Graph:
        """The underlying FalkorDB :class:`~falkordb.Graph` handle."""
        return self._graph

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        read_only: bool = False,
    ) -> list[list[Any]]:
        """Execute a Cypher query and return its result rows.

        Args:
            cypher: The Cypher statement, using ``$param`` placeholders.
            params: Optional parameter map bound to the statement.
            read_only: If True, use FalkorDB's read-only execution path.

        Returns:
            The ``result_set`` as a list of rows (each row a list of values).

        Raises:
            FalkorDBError: If the query fails to execute.
        """
        try:
            if read_only:
                result = self._graph.ro_query(cypher, params or {})
            else:
                result = self._graph.query(cypher, params or {})
            return result.result_set or []
        except Exception as exc:  # noqa: BLE001
            raise FalkorDBError(
                f"Cypher query failed: {exc}\n  query: {cypher.strip()[:200]}"
            ) from exc

    def ping(self) -> bool:
        """Return True if the server responds to a ping."""
        try:
            return bool(self._db.connection.ping())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ #
    # Schema / indexes
    # ------------------------------------------------------------------ #
    def ensure_indexes(self) -> dict[str, str]:
        """Create the vector and full-text indexes if they do not yet exist.

        Idempotent: re-running against an already-indexed graph is a no-op that
        reports ``"exists"`` rather than failing.

        Returns:
            A mapping of index name to ``"created"`` or ``"exists"``.
        """
        return {
            "chunk_embedding_vector": self._create_vector_index(
                "Chunk", "embedding", dim=config.embedding_dim, similarity="cosine"
            ),
            "chunk_text_fulltext": self._create_fulltext_index("Chunk", "text"),
        }

    def _create_vector_index(
        self, label: str, prop: str, *, dim: int, similarity: str
    ) -> str:
        """Create an HNSW vector index, treating 'already exists' as success.

        Uses an explicit ``efRuntime`` so KNN queries return the full requested
        ``k`` (the default is small and under-returns on modest graphs).
        """
        try:
            self._graph.query(
                f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.{prop}) "
                f"OPTIONS {{dimension: {dim}, similarityFunction: '{similarity}', "
                f"M: 16, efConstruction: 200, efRuntime: 100}}"
            )
            logger.info(
                "Created vector index on :%s(%s) dim=%d sim=%s efRuntime=100",
                label, prop, dim, similarity,
            )
            return "created"
        except Exception as exc:  # noqa: BLE001
            if self._is_already_exists(exc):
                logger.debug("Vector index on :%s(%s) already exists", label, prop)
                return "exists"
            raise FalkorDBError(
                f"Failed to create vector index on :{label}({prop}): {exc}"
            ) from exc

    def _create_fulltext_index(self, label: str, prop: str) -> str:
        """Create a BM25 full-text index, treating 'already exists' as success."""
        try:
            self._graph.create_node_fulltext_index(label, prop)
            logger.info("Created full-text index on :%s(%s)", label, prop)
            return "created"
        except Exception as exc:  # noqa: BLE001
            if self._is_already_exists(exc):
                logger.debug("Full-text index on :%s(%s) already exists", label, prop)
                return "exists"
            raise FalkorDBError(
                f"Failed to create full-text index on :{label}({prop}): {exc}"
            ) from exc

    @staticmethod
    def _is_already_exists(exc: Exception) -> bool:
        """Return True if the exception indicates a pre-existing index/attribute."""
        msg = str(exc).lower()
        return any(marker in msg for marker in _ALREADY_EXISTS_MARKERS)

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #
    def clear_graph(self) -> None:
        """Delete every node and relationship in the graph (keeps indexes).

        Used by the Streamlit "Clear all documents" action.
        """
        self.query("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all nodes from graph '%s'", self.graph_name)

    def node_count(self) -> int:
        """Return the total number of nodes currently in the graph."""
        rows = self.query("MATCH (n) RETURN count(n)", read_only=True)
        return int(rows[0][0]) if rows else 0

    # ------------------------------------------------------------------ #
    # Smoke test
    # ------------------------------------------------------------------ #
    def smoke_test(self) -> bool:
        """Create a node, read it back, and delete it to prove the round-trip.

        Returns:
            True if the node was written, read back identically, and removed.

        Raises:
            FalkorDBError: If any step of the round-trip fails or mismatches.
        """
        marker = f"smoketest_{int(time.time() * 1000)}"
        self.query(
            "CREATE (n:__SmokeTest {marker: $marker, value: $value})",
            {"marker": marker, "value": 42},
        )
        rows = self.query(
            "MATCH (n:__SmokeTest {marker: $marker}) RETURN n.value",
            {"marker": marker},
            read_only=True,
        )
        if not rows or rows[0][0] != 42:
            raise FalkorDBError(f"Smoke test read-back failed; got {rows!r}")
        self.query("MATCH (n:__SmokeTest {marker: $marker}) DELETE n", {"marker": marker})
        remaining = self.query(
            "MATCH (n:__SmokeTest {marker: $marker}) RETURN count(n)",
            {"marker": marker},
            read_only=True,
        )
        if remaining and int(remaining[0][0]) != 0:
            raise FalkorDBError("Smoke test delete failed; node still present")
        logger.info("FalkorDB smoke test passed (write -> read -> delete round-trip).")
        return True


# Module-level shared client, created lazily so importing the module never fails
# merely because the database is momentarily down.
_client: FalkorDBClient | None = None


def get_client() -> FalkorDBClient:
    """Return the process-wide :class:`FalkorDBClient`, creating it on first use."""
    global _client
    if _client is None:
        _client = FalkorDBClient()
    return _client


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    client = get_client()
    print("Ensuring indexes...")
    for name, status in client.ensure_indexes().items():
        print(f"  {name}: {status}")
    print("Running smoke test...")
    client.smoke_test()
    print(f"Node count after smoke test: {client.node_count()}")
    print("\n✅ FalkorDB client is working.")
