"""Tests for Stage 7 (global entity resolution) — src/resolution/entity_resolution.py.

A stub embedder drives fast, deterministic Splink clustering; pure tests cover
the canonical/edge builders; a monkeypatched classifier covers corroborate/
contradict edges; and one real run over the 2-document corpus confirms genuine
cross-document merges (and that distinct companies are not merged).
"""

from __future__ import annotations

import pytest

from src.extraction.ner import Entity
from src.resolution.entity_resolution import EntityResolver


class _StubEmbedder:
    """Returns a deterministic vector keyed by a keyword in the name."""

    def embed_texts(self, names: list[str]) -> list[list[float]]:
        out = []
        for n in names:
            low = n.lower()
            if "acme" in low:
                out.append([1.0, 0.0, 0.0])
            elif "globex" in low:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out


def _e(name: str, etype: str, doc: str, *chunk_ids: str, aliases=None) -> Entity:
    e = Entity(name=name, entity_type=etype, document_id=doc, document_name=f"{doc}.pdf",
               aliases=aliases or [])
    for cid in chunk_ids:
        e.add_mention(cid, 1)
    return e


@pytest.fixture(scope="module")
def resolver() -> EntityResolver:
    return EntityResolver(embedder=_StubEmbedder())


# --------------------------------------------------------------------------- #
# Clustering (stub embedder, real Splink)
# --------------------------------------------------------------------------- #
def test_merges_variants_and_keeps_distinct(resolver: EntityResolver) -> None:
    entities = [
        _e("Acme Corporation", "Organisation", "docA"),
        _e("Acme", "Organisation", "docA"),
        _e("Acme Corporation", "Organisation", "docB"),
        _e("Globex Industries", "Organisation", "docA"),
    ]
    res = resolver.resolve_all(entities, classify_pairs=False)
    clusters = {c.canonical_name: c for c in res.canonical_entities}
    acme = next(c for c in res.canonical_entities if "Acme Corporation" == c.canonical_name)
    assert acme.size == 3  # the three Acme variants
    assert "Acme" in acme.aliases
    assert "docA" in acme.source_documents and "docB" in acme.source_documents
    # Globex remains its own singleton cluster (not merged with Acme).
    globex = next(c for c in res.canonical_entities if "Globex" in c.canonical_name)
    assert globex.size == 1
    assert res.merged_count == 2  # 4 entities → 2 clusters


def test_same_as_and_related_doc_edges(resolver: EntityResolver) -> None:
    entities = [
        _e("Acme Corporation", "Organisation", "docA"),
        _e("Acme Corporation", "Organisation", "docB"),
    ]
    res = resolver.resolve_all(entities, classify_pairs=False)
    same_as = [e for e in res.cross_doc_edges if e["type"] == "SAME_AS"]
    related = [e for e in res.cross_doc_edges if e["type"] == "RELATED_TO"]
    assert len(same_as) == 1  # one non-representative member → representative
    assert len(related) == 1 and {related[0]["from_id"], related[0]["to_id"]} == {"docA", "docB"}


def test_fewer_than_two_entities(resolver: EntityResolver) -> None:
    res = resolver.resolve_all([_e("Acme", "Organisation", "docA")])
    assert res.merged_count == 0
    assert len(res.canonical_entities) == 1 and res.canonical_entities[0].size == 1


# --------------------------------------------------------------------------- #
# Corroborate / contradict edges (monkeypatched LLM)
# --------------------------------------------------------------------------- #
def test_contradiction_edges_from_classifier(resolver: EntityResolver, monkeypatch) -> None:
    entities = [
        _e("Acme Corporation", "Organisation", "docA", "docA_chunk_0001"),
        _e("Acme Corporation", "Organisation", "docB", "docB_chunk_0001"),
    ]
    chunk_text = {
        "docA_chunk_0001": "Acme revenue increased sharply.",
        "docB_chunk_0001": "Acme revenue fell sharply.",
    }
    monkeypatch.setattr(resolver, "_classify_pair", lambda a, b: "contradict")
    res = resolver.resolve_all(entities, chunk_text, classify_pairs=True)
    contradicts = [e for e in res.cross_doc_edges if e["type"] == "CONTRADICTS"]
    assert len(contradicts) == 1
    assert {contradicts[0]["from_id"], contradicts[0]["to_id"]} == set(chunk_text)


# --------------------------------------------------------------------------- #
# Pure builders
# --------------------------------------------------------------------------- #
def test_entity_id_matches_publisher_scheme() -> None:
    e = _e("Acme Corporation", "Organisation", "docA")
    assert EntityResolver._eid(e) == "docA::Organisation::acme corporation"


def test_canonical_picks_longest_name() -> None:
    members = [_e("Acme", "Organisation", "docA"), _e("Acme Corporation", "Organisation", "docB")]
    canon = EntityResolver._build_canonicals({"c1": members})[0]
    assert canon.canonical_name == "Acme Corporation"
    assert "Acme" in canon.aliases
    assert sorted(canon.source_documents) == ["docA", "docB"]


# --------------------------------------------------------------------------- #
# Real end-to-end over the corpus
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_real_cross_document_resolution(corpus_pdfs) -> None:
    from src.extraction.domain_detector import DomainDetector
    from src.extraction.gliner_extractor import GLiNERExtractor
    from src.extraction.ner import NERExtractor
    from src.ingestion.chunker import Chunker
    from src.ingestion.parser import DocumentParser

    parser, chunker = DocumentParser(), Chunker()
    dd, ner, gl = DomainDetector(), NERExtractor(), GLiNERExtractor()
    all_entities = []
    for p in corpus_pdfs:
        cr = chunker.chunk(parser.parse(p))
        dom = dd.detect(cr)
        n = ner.extract(cr)
        all_entities.extend(gl.extract(cr, dom, prior_entities=n.entities).entities)

    res = EntityResolver().resolve_all(all_entities, classify_pairs=False)
    merged = [c for c in res.canonical_entities if c.size > 1]
    # The deliberately shared entities should merge across the two documents.
    # (Canonical name may include GLiNER span noise, so match on substring.)
    acme = [c for c in merged if "Acme" in c.canonical_name and c.entity_type == "Organisation"]
    assert acme and len(acme[0].source_documents) == 2
    # Acme and Globex must be in separate canonical entities (no over-merge).
    acme_cluster = acme[0]
    assert not any("Globex" in a for a in [acme_cluster.canonical_name, *acme_cluster.aliases])
    assert any("Globex" in c.canonical_name for c in res.canonical_entities)
    assert res.merged_count >= 3
