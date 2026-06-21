"""Stage 7 — global entity resolution across the whole batch (Splink).

After every document is ingested, all entities from all documents are resolved
together with Splink:

  * **Blocking** on the first three characters of the name plus the entity type
    (cheap candidate generation).
  * **Pairwise scoring** combining Jaro-Winkler and Jaccard string similarity
    with embedding cosine similarity (entity names embedded with the configured
    model). Match probabilities use manually-set m/u weights, so no EM training
    is needed on small batches.
  * **Clustering** of pairwise predictions at ``config.SPLINK_MERGE_THRESHOLD``
    via connected components.

The result is a :class:`ResolutionResult` describing the canonical entities and
the cross-document edges (``SAME_AS``, ``RELATED_TO``, ``CORROBORATES``,
``CONTRADICTS``) the graph publisher then writes. The resolver itself performs no
graph writes — it returns data. Cross-document chunk pairs that share a merged
entity are classified as corroborating or contradicting by the configured LLM.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from itertools import combinations

from src.config import config
from src.extraction.ner import Entity

logger = logging.getLogger(__name__)
logging.getLogger("splink").setLevel(logging.WARNING)

# m/u weights per comparison level (manually set so no training is required).
# Each tuple is (m_probability, u_probability): m = P(level | match),
# u = P(level | non-match). Strong signals get high m and low u.
# Within a block (same name prefix + type) the match rate is high, so the prior
# is well above the unblocked base rate.
_PRIOR_MATCH = 0.1
_MAX_CONTRA_PAIRS = 24  # cap LLM corroborate/contradict calls per batch

_CONTRA_INSTRUCTION = (
    "Two statements from different documents mention the same entity. Decide "
    "whether they corroborate (agree / support each other), contradict (state "
    "conflicting facts), or are unrelated. "
    'Respond with JSON {"label": "corroborate" | "contradict" | "unrelated"}.'
)


@dataclass
class CanonicalEntity:
    """One resolved entity cluster."""

    cluster_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str]
    source_documents: list[str]
    member_ids: list[str]
    confidence: float
    size: int


@dataclass
class ResolutionResult:
    """Stage 7 output consumed by the UI and the graph publisher."""

    merged_count: int
    canonical_entities: list[CanonicalEntity] = field(default_factory=list)
    cross_doc_edges: list[dict] = field(default_factory=list)


class EntityResolver:
    """Resolves entities across documents with Splink + an LLM contradiction check."""

    def __init__(self, embedder=None, llm_client=None) -> None:
        self._embedder = embedder
        self._llm = llm_client
        self._threshold = config.splink_merge_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def resolve_all(
        self,
        all_entities: list[Entity],
        chunk_text_by_id: dict[str, str] | None = None,
        classify_pairs: bool = True,
    ) -> ResolutionResult:
        """Resolve every entity in the batch into canonical clusters.

        Args:
            all_entities: Entities from all documents in the batch.
            chunk_text_by_id: Optional chunk_id→text map; enables the
                corroborate/contradict classification of cross-document chunks.
            classify_pairs: When True (and chunk text is available), classify
                cross-document chunk pairs that share a merged entity.

        Returns:
            A :class:`ResolutionResult` with merge counts, canonical entities,
            and cross-document edges.
        """
        unique = self._dedupe(all_entities)
        if len(unique) < 2:
            return ResolutionResult(merged_count=0,
                                    canonical_entities=self._singletons(unique))

        clusters = self._cluster(unique)
        canonicals = self._build_canonicals(clusters)
        merged_count = len(unique) - len(canonicals)

        edges: list[dict] = []
        edges += self._same_as_edges(clusters)
        edges += self._related_doc_edges(canonicals)
        if classify_pairs and chunk_text_by_id:
            edges += self._corroboration_edges(clusters, chunk_text_by_id)

        logger.info(
            "Resolved %d entities -> %d canonical (%d merged), %d cross-doc edges",
            len(unique), len(canonicals), merged_count, len(edges),
        )
        return ResolutionResult(
            merged_count=merged_count, canonical_entities=canonicals, cross_doc_edges=edges
        )

    # ------------------------------------------------------------------ #
    # Splink clustering
    # ------------------------------------------------------------------ #
    def _cluster(self, entities: dict[str, Entity]) -> dict[str, list[Entity]]:
        """Run Splink and return clusters keyed by cluster id."""
        import pandas as pd
        from splink import DuckDBAPI, Linker, SettingsCreator, block_on

        names = [e.name for e in entities.values()]
        embeddings = self._embed(names)
        rows = []
        for (uid, e), emb in zip(entities.items(), embeddings):
            norm = self._norm(e.name)
            rows.append({
                "unique_id": uid, "name": e.name, "name_norm": norm,
                "first3": norm.replace(" ", "")[:3], "entity_type": e.entity_type,
                "embedding": emb,
            })
        df = pd.DataFrame(rows)

        settings = SettingsCreator(
            link_type="dedupe_only",
            unique_id_column_name="unique_id",
            probability_two_random_records_match=_PRIOR_MATCH,
            blocking_rules_to_generate_predictions=[block_on("first3", "entity_type")],
            comparisons=[
                self._name_jaro_comparison(),
                self._name_jaccard_comparison(),
                self._embedding_comparison(),
            ],
        )
        linker = Linker(df, settings, DuckDBAPI())
        predictions = linker.inference.predict()
        clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions, threshold_match_probability=self._threshold
        )
        cdf = clustered.as_pandas_dataframe()

        clusters: dict[str, list[Entity]] = {}
        for _, row in cdf.iterrows():
            clusters.setdefault(str(row["cluster_id"]), []).append(entities[row["unique_id"]])
        return clusters

    @staticmethod
    def _name_jaro_comparison():
        import splink.comparison_level_library as cll
        from splink.comparison_library import CustomComparison

        return CustomComparison(
            output_column_name="name_norm",
            comparison_description="Name Jaro-Winkler",
            comparison_levels=[
                cll.NullLevel("name_norm"),
                cll.ExactMatchLevel("name_norm").configure(m_probability=0.6, u_probability=0.001),
                cll.JaroWinklerLevel("name_norm", 0.92).configure(m_probability=0.3, u_probability=0.02),
                cll.JaroWinklerLevel("name_norm", 0.82).configure(m_probability=0.08, u_probability=0.1),
                cll.ElseLevel().configure(m_probability=0.02, u_probability=0.879),
            ],
        )

    @staticmethod
    def _name_jaccard_comparison():
        import splink.comparison_level_library as cll
        from splink.comparison_library import CustomComparison

        return CustomComparison(
            output_column_name="name_jaccard",
            comparison_description="Name token Jaccard",
            comparison_levels=[
                cll.NullLevel("name_norm"),
                cll.JaccardLevel("name_norm", 0.8).configure(m_probability=0.55, u_probability=0.02),
                cll.JaccardLevel("name_norm", 0.4).configure(m_probability=0.3, u_probability=0.2),
                # A weak Jaccard must not veto a strong embedding match (abbrevs).
                cll.ElseLevel().configure(m_probability=0.4, u_probability=0.5),
            ],
        )

    @staticmethod
    def _embedding_comparison():
        import splink.comparison_level_library as cll
        from splink.comparison_library import CustomComparison

        # DuckDB's array_cosine_similarity (used by CosineSimilarityLevel) needs a
        # fixed-size ARRAY, but a pandas list column registers as a variable LIST.
        # list_cosine_similarity handles LIST, so use it via CustomLevel.
        def _cos(threshold: float):
            return cll.CustomLevel(
                f'list_cosine_similarity("embedding_l", "embedding_r") >= {threshold}',
                label_for_charts=f"cosine >= {threshold}",
            )

        return CustomComparison(
            output_column_name="embedding",
            comparison_description="Name embedding cosine similarity",
            comparison_levels=[
                cll.NullLevel("embedding"),
                _cos(0.92).configure(m_probability=0.6, u_probability=0.004),
                _cos(0.82).configure(m_probability=0.35, u_probability=0.05),
                cll.ElseLevel().configure(m_probability=0.05, u_probability=0.83),
            ],
        )

    # ------------------------------------------------------------------ #
    # Canonicals & edges
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_canonicals(clusters: dict[str, list[Entity]]) -> list[CanonicalEntity]:
        """Summarise each cluster into a canonical entity."""
        canonicals: list[CanonicalEntity] = []
        for cluster_id, members in clusters.items():
            canonical = EntityResolver._pick_canonical_name(members)
            aliases = sorted({a for m in members for a in ([m.name] + m.aliases)} - {canonical})
            docs = sorted({m.document_id for m in members})
            confidence = max(m.confidence for m in members)
            canonicals.append(CanonicalEntity(
                cluster_id=cluster_id, canonical_name=canonical,
                entity_type=members[0].entity_type, aliases=aliases,
                source_documents=docs, member_ids=[EntityResolver._eid(m) for m in members],
                confidence=confidence, size=len(members),
            ))
        return canonicals

    @staticmethod
    def _pick_canonical_name(members: list[Entity]) -> str:
        """Choose a clean, representative canonical name for a cluster.

        Prefers the longest name that is free of clause punctuation and not
        excessively long (avoids noisy over-extended NER/GLiNER spans); falls
        back to the longest name overall.
        """
        clean = [
            m.name for m in members
            if "," not in m.name and ";" not in m.name and len(m.name.split()) <= 6
        ]
        pool = clean or [m.name for m in members]
        return max(pool, key=len)

    @staticmethod
    def _same_as_edges(clusters: dict[str, list[Entity]]) -> list[dict]:
        """SAME_AS edges from each non-representative member to the representative."""
        edges: list[dict] = []
        for members in clusters.values():
            if len(members) < 2:
                continue
            canonical_name = EntityResolver._pick_canonical_name(members)
            rep = next((m for m in members if m.name == canonical_name), members[0])
            rep_id = EntityResolver._eid(rep)
            for m in members:
                mid = EntityResolver._eid(m)
                if mid != rep_id:
                    edges.append({"type": "SAME_AS", "from_id": mid, "to_id": rep_id,
                                  "props": {"confidence": m.confidence}})
        return edges

    @staticmethod
    def _related_doc_edges(canonicals: list[CanonicalEntity]) -> list[dict]:
        """RELATED_TO edges between documents that share a merged entity."""
        seen: set[tuple[str, str]] = set()
        edges: list[dict] = []
        for c in canonicals:
            if len(c.source_documents) < 2:
                continue
            for a, b in combinations(sorted(c.source_documents), 2):
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                edges.append({"type": "RELATED_TO", "from_id": a, "to_id": b,
                              "props": {"via_entity": c.canonical_name}})
        return edges

    def _corroboration_edges(
        self, clusters: dict[str, list[Entity]], chunk_text_by_id: dict[str, str]
    ) -> list[dict]:
        """Classify cross-document chunk pairs sharing a merged entity."""
        candidates: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for members in clusters.values():
            if len({m.document_id for m in members}) < 2:
                continue
            # One representative mention chunk per document.
            chunk_by_doc: dict[str, str] = {}
            for m in members:
                if m.mentions:
                    chunk_by_doc.setdefault(m.document_id, m.mentions[0].chunk_id)
            chunk_ids = list(chunk_by_doc.values())
            for a, b in combinations(chunk_ids, 2):
                key = tuple(sorted((a, b)))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    candidates.append((a, b))

        edges: list[dict] = []
        for a, b in candidates[:_MAX_CONTRA_PAIRS]:
            text_a, text_b = chunk_text_by_id.get(a), chunk_text_by_id.get(b)
            if not text_a or not text_b:
                continue
            label = self._classify_pair(text_a, text_b)
            if label == "corroborate":
                edges.append({"type": "CORROBORATES", "from_id": a, "to_id": b, "props": {}})
            elif label == "contradict":
                edges.append({"type": "CONTRADICTS", "from_id": a, "to_id": b, "props": {}})
        return edges

    def _classify_pair(self, text_a: str, text_b: str) -> str | None:
        """Ask the LLM whether two statements corroborate/contradict/unrelated."""
        client = self._get_llm()
        prompt = f"{_CONTRA_INSTRUCTION}\n\nStatement A:\n{text_a}\n\nStatement B:\n{text_b}\n\nJSON:"
        try:
            response = client.generate(
                model=config.ollama_llm_model, prompt=prompt,
                format="json", options={"temperature": 0.0},
            )
            raw = (getattr(response, "response", None) or "").strip()
            label = str(json.loads(raw).get("label", "")).lower()
            return label if label in {"corroborate", "contradict"} else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pair classification failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dedupe(entities: list[Entity]) -> dict[str, Entity]:
        """Map unique entity id → entity, keeping the first of any duplicates."""
        out: dict[str, Entity] = {}
        for e in entities:
            out.setdefault(EntityResolver._eid(e), e)
        return out

    @staticmethod
    def _singletons(entities: dict[str, Entity]) -> list[CanonicalEntity]:
        """Wrap each entity as its own canonical cluster (no merges)."""
        return [
            CanonicalEntity(
                cluster_id=uid, canonical_name=e.name, entity_type=e.entity_type,
                aliases=e.aliases, source_documents=[e.document_id],
                member_ids=[uid], confidence=e.confidence, size=1,
            )
            for uid, e in entities.items()
        ]

    @staticmethod
    def _eid(entity: Entity) -> str:
        """Graph-aligned entity id (matches GraphPublisher._entity_id)."""
        return f"{entity.document_id}::{entity.entity_type}::{EntityResolver._norm(entity.name)}"

    @staticmethod
    def _norm(name: str) -> str:
        return " ".join(name.lower().split())

    def _embed(self, names: list[str]) -> list[list[float]]:
        """Embed entity names with the configured model (lazy embedder)."""
        if self._embedder is None:
            from src.retrieval.embedder import Embedder

            self._embedder = Embedder()
        return self._embedder.embed_texts(names)

    def _get_llm(self):
        if self._llm is None:
            import ollama

            self._llm = ollama.Client(host=config.ollama_base_url)
        return self._llm


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = sys.argv[1:] or [
        "data/uploads/acme_annual_report.pdf", "data/uploads/globex_q3_filing.pdf",
    ]
    from src.extraction.domain_detector import DomainDetector
    from src.extraction.gliner_extractor import GLiNERExtractor
    from src.extraction.ner import NERExtractor
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser

    parser, chunker = DocumentParser(), Chunker()
    dd, ner, gl = DomainDetector(), NERExtractor(), GLiNERExtractor()
    all_entities, chunk_text = [], {}
    for p in paths:
        pr = parser.parse(p)
        cr = chunker.chunk(pr)
        for c in cr.chunks:
            chunk_text[c.chunk_id] = c.text
        dom = dd.detect(cr)
        n = ner.extract(cr)
        er = gl.extract(cr, dom, prior_entities=n.entities)
        all_entities.extend(er.entities)

    result = EntityResolver().resolve_all(all_entities, chunk_text, classify_pairs=False)
    print(f"\nmerged_count={result.merged_count}")
    print("Merged clusters (size>1):")
    for c in result.canonical_entities:
        if c.size > 1:
            print(f"  [{c.entity_type}] {c.canonical_name!r} <- aliases={c.aliases} docs={c.source_documents}")
    print(f"cross_doc_edges: {[(e['type'], e.get('props')) for e in result.cross_doc_edges][:10]}")
