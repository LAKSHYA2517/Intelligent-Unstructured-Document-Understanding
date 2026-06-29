"""Lightweight retrieval evaluation utilities for benchmarkable improvements."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Awaitable, Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class EvalCase:
    query: str
    relevant_ids: set[str] = field(default_factory=set)
    relevant_text: str = ""
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    query: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: bool


@dataclass(frozen=True, slots=True)
class EvalSummary:
    cases: int
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    hit_rate: float
    results: list[EvalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "hit_rate": self.hit_rate,
            "results": [
                {
                    "query": result.query,
                    "retrieved_ids": result.retrieved_ids,
                    "relevant_ids": sorted(result.relevant_ids),
                    "precision_at_k": result.precision_at_k,
                    "recall_at_k": result.recall_at_k,
                    "reciprocal_rank": result.reciprocal_rank,
                    "hit": result.hit,
                }
                for result in self.results
            ],
        }


RetrieverFn = Callable[[str, int], Awaitable[Sequence[Any]]]


async def evaluate_retrieval(cases: Sequence[EvalCase], retrieve: RetrieverFn, *, k: int = 6) -> EvalSummary:
    if k < 1:
        raise ValueError("k must be positive.")

    results: list[EvalResult] = []
    for case in cases:
        retrieved = list(await retrieve(case.query, k))
        retrieved_ids = [_result_id(item) for item in retrieved[:k]]
        results.append(score_case(case, retrieved_ids, k=k))

    if not results:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0, [])

    return EvalSummary(
        cases=len(results),
        precision_at_k=mean(result.precision_at_k for result in results),
        recall_at_k=mean(result.recall_at_k for result in results),
        mean_reciprocal_rank=mean(result.reciprocal_rank for result in results),
        hit_rate=mean(1.0 if result.hit else 0.0 for result in results),
        results=results,
    )


def score_case(case: EvalCase, retrieved_ids: Sequence[str], *, k: int) -> EvalResult:
    top_ids = list(retrieved_ids[:k])
    relevant = set(case.relevant_ids)
    if not relevant:
        return EvalResult(case.query, top_ids, relevant, 0.0, 0.0, 0.0, False)

    hits = [chunk_id for chunk_id in top_ids if chunk_id in relevant]
    precision = len(hits) / k
    recall = len(set(hits)) / len(relevant)
    reciprocal_rank = 0.0
    for index, chunk_id in enumerate(top_ids, start=1):
        if chunk_id in relevant:
            reciprocal_rank = 1.0 / index
            break
    return EvalResult(
        query=case.query,
        retrieved_ids=top_ids,
        relevant_ids=relevant,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        hit=bool(hits),
    )


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load JSONL cases with query plus relevant_ids or relevant_id."""

    cases: list[EvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            query = str(payload.get("query", "")).strip()
            if not query:
                raise ValueError(f"Missing query on line {line_number}.")
            relevant_ids = payload.get("relevant_ids", [])
            if "relevant_id" in payload:
                relevant_ids = [payload["relevant_id"], *list(relevant_ids)]
            cases.append(
                EvalCase(
                    query=query,
                    relevant_ids={str(item) for item in relevant_ids},
                    relevant_text=str(payload.get("relevant_text", "")),
                    source=str(payload["source"]) if payload.get("source") is not None else None,
                    metadata=dict(payload.get("metadata", {}) or {}),
                )
            )
    return cases


def write_eval_summary(summary: EvalSummary, path: str | Path) -> None:
    Path(path).write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def evaluate_retrieval_sync(cases: Sequence[EvalCase], retrieve: RetrieverFn, *, k: int = 6) -> EvalSummary:
    return asyncio.run(evaluate_retrieval(cases, retrieve, k=k))


def _result_id(item: Any) -> str:
    chunk = getattr(item, "chunk", None)
    if chunk is not None:
        return str(getattr(chunk, "chunk_id", ""))
    if isinstance(item, Mapping):
        if "chunk_id" in item:
            return str(item["chunk_id"])
        if "id" in item:
            return str(item["id"])
        if "chunk" in item and isinstance(item["chunk"], Mapping):
            return str(item["chunk"].get("chunk_id", ""))
    return str(getattr(item, "chunk_id", getattr(item, "id", "")))
