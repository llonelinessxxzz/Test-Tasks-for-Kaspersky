from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from eval.data import GenerationEvalCase
from support_rag.generation.prompt import response_matches_language


@dataclass(frozen=True)
class CaseResult:
    id: str
    question: str
    language: str

    expected_abstain: bool
    actual_abstain: bool

    expected_document_ids: list[str]
    retrieved_document_ids: list[str]
    cited_document_ids: list[str]

    abstention_correct: bool

    citation_present: bool
    citations_valid: bool
    expected_source_cited: bool | None
    expected_source_retrieved: bool | None

    language_correct: bool
    finished_normally: bool
    repaired: bool

    generation_attempts: int

    context_tokens: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    latency_seconds: float

    answer: str

    error: str | None = None


def expected_language_name(
    language: str,
) -> str:
    if language == "ru":
        return "Russian"

    if language == "en":
        return "English"

    raise ValueError(f"Unsupported language: {language}")


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        return 0.0

    if not 0.0 <= percentile_value <= 1.0:
        raise ValueError("percentile_value must be between 0 and 1")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = percentile_value * (len(ordered) - 1)

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    weight = position - lower_index

    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


async def evaluate_case(
    *,
    runtime: Any,
    case: GenerationEvalCase,
) -> CaseResult:
    started_at = time.perf_counter()

    try:
        response = await runtime.service.ask(case.question)

        latency_seconds = time.perf_counter() - started_at

        retrieved_document_ids = list(
            dict.fromkeys(source.document_id for source in response.sources)
        )

        cited_document_ids = list(
            dict.fromkeys(source.document_id for source in response.sources if source.cited)
        )

        expected_documents = set(case.expected_document_ids)

        retrieved_documents = set(retrieved_document_ids)

        cited_documents = set(cited_document_ids)

        abstention_correct = response.abstained == case.expected_abstain

        citation_present = bool(response.cited_source_numbers)

        citations_valid = not response.invalid_citation_numbers

        if case.expected_abstain:
            expected_source_cited = None
            expected_source_retrieved = None
        else:
            expected_source_cited = bool(expected_documents & cited_documents)

            expected_source_retrieved = bool(expected_documents & retrieved_documents)

        language_correct = response_matches_language(
            response.answer,
            expected_language_name(case.language),
        )

        finished_normally = response.finish_reason == "stop"

        repaired = response.generation_attempts > 1

        return CaseResult(
            id=case.id,
            question=case.question,
            language=case.language,
            expected_abstain=(case.expected_abstain),
            actual_abstain=(response.abstained),
            expected_document_ids=(case.expected_document_ids),
            retrieved_document_ids=(retrieved_document_ids),
            cited_document_ids=(cited_document_ids),
            abstention_correct=(abstention_correct),
            citation_present=(citation_present),
            citations_valid=(citations_valid),
            expected_source_cited=(expected_source_cited),
            expected_source_retrieved=(expected_source_retrieved),
            language_correct=(language_correct),
            finished_normally=(finished_normally),
            repaired=repaired,
            generation_attempts=(response.generation_attempts),
            context_tokens=(response.context_tokens),
            prompt_tokens=(response.prompt_tokens),
            completion_tokens=(response.completion_tokens),
            total_tokens=(response.total_tokens),
            latency_seconds=(latency_seconds),
            answer=response.answer,
        )

    except Exception as exc:
        latency_seconds = time.perf_counter() - started_at

        return CaseResult(
            id=case.id,
            question=case.question,
            language=case.language,
            expected_abstain=(case.expected_abstain),
            actual_abstain=False,
            expected_document_ids=(case.expected_document_ids),
            retrieved_document_ids=[],
            cited_document_ids=[],
            abstention_correct=False,
            citation_present=False,
            citations_valid=False,
            expected_source_cited=(None if case.expected_abstain else False),
            expected_source_retrieved=(None if case.expected_abstain else False),
            language_correct=False,
            finished_normally=False,
            repaired=False,
            generation_attempts=0,
            context_tokens=0,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_seconds=(latency_seconds),
            answer="",
            error=(f"{type(exc).__name__}: {exc}"),
        )


def summarize_generation(results: list[CaseResult]) -> dict[str, Any]:
    completed = [row for row in results if row.error is None]
    supported = [row for row in results if not row.expected_abstain]
    out_of_kb = [row for row in results if row.expected_abstain]
    answered = [row for row in completed if not row.expected_abstain and not row.actual_abstain]

    def success_rate(rows: list[CaseResult], predicate: Any) -> float | None:
        return mean(bool(row.error is None and predicate(row)) for row in rows) if rows else None

    latencies = [row.latency_seconds for row in completed]
    summary = {
        "cases": len(results),
        "successful_cases": len(completed),
        "failed_cases": len(results) - len(completed),
        "supported_cases": len(supported),
        "out_of_kb_cases": len(out_of_kb),
        "metrics": {
            "supported_answer_rate": success_rate(supported, lambda row: not row.actual_abstain),
            "out_of_kb_abstention_rate": success_rate(out_of_kb, lambda row: row.actual_abstain),
            "abstention_accuracy": success_rate(results, lambda row: row.abstention_correct),
            "expected_source_retrieval_accuracy": success_rate(
                supported, lambda row: row.expected_source_retrieved
            ),
            "expected_source_citation_accuracy": success_rate(
                supported, lambda row: row.expected_source_cited
            ),
            "citation_presence_supported": success_rate(answered, lambda row: row.citation_present),
            "citation_validity": success_rate(completed, lambda row: row.citations_valid),
            "language_accuracy": success_rate(results, lambda row: row.language_correct),
            "finish_rate": success_rate(results, lambda row: row.finished_normally),
            "repair_rate": success_rate(completed, lambda row: row.repaired),
        },
        "latency_seconds": {
            "mean": mean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50) if latencies else None,
            "p95": percentile(latencies, 0.95) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "breakdown": {},
        "denominator_note": (
            "Answer/abstention/coverage/language/finish rates include failed requests. "
            "Citation presence is conditional on answered supported cases. "
            "Citation validity and latency use completed requests; "
            "they do not measure groundedness."
        ),
    }
    for language in ("ru", "en"):
        rows = [row for row in results if row.language == language]
        summary["breakdown"][language] = {
            "cases": len(rows),
            "abstention_accuracy": success_rate(rows, lambda row: row.abstention_correct),
            "language_accuracy": success_rate(rows, lambda row: row.language_correct),
        }
    groups: dict[str, list[str]] = {
        "technical_errors": [],
        "retrieval_misses": [],
        "abstained_with_expected_context": [],
        "answered_with_expected_context": [],
        "out_of_kb_correct_abstentions": [],
        "out_of_kb_answered": [],
    }
    for row in results:
        if row.error:
            group = "technical_errors"
        elif row.expected_abstain:
            group = "out_of_kb_correct_abstentions" if row.actual_abstain else "out_of_kb_answered"
        elif not row.expected_source_retrieved:
            group = "retrieval_misses"
        elif row.actual_abstain:
            group = "abstained_with_expected_context"
        else:
            group = "answered_with_expected_context"
        groups[group].append(row.id)
    summary["case_groups"] = groups
    return summary


def recall_at_k(
    ranked_documents: Sequence[str],
    relevant_documents: set[str],
    k: int,
) -> float:
    retrieved = set(ranked_documents[:k])

    return len(retrieved & relevant_documents) / len(relevant_documents)


def hit_at_k(
    ranked_documents: Sequence[str],
    relevant_documents: set[str],
    k: int,
) -> float:
    retrieved = set(ranked_documents[:k])

    return float(bool(retrieved & relevant_documents))


def reciprocal_rank(
    ranked_documents: Sequence[str],
    relevant_documents: set[str],
) -> float:
    for rank, document_id in enumerate(
        ranked_documents,
        start=1,
    ):
        if document_id in relevant_documents:
            return 1.0 / rank

    return 0.0


def evaluate_ranking(
    ranked_documents: Sequence[str],
    relevant_documents: set[str],
) -> dict[str, float]:
    return {
        "recall@1": recall_at_k(
            ranked_documents,
            relevant_documents,
            1,
        ),
        "recall@3": recall_at_k(
            ranked_documents,
            relevant_documents,
            3,
        ),
        "recall@5": recall_at_k(
            ranked_documents,
            relevant_documents,
            5,
        ),
        "hit@1": hit_at_k(
            ranked_documents,
            relevant_documents,
            1,
        ),
        "hit@3": hit_at_k(
            ranked_documents,
            relevant_documents,
            3,
        ),
        "hit@5": hit_at_k(
            ranked_documents,
            relevant_documents,
            5,
        ),
        "mrr": reciprocal_rank(
            ranked_documents,
            relevant_documents,
        ),
    }


def print_case_result(
    index: int,
    total: int,
    result: CaseResult,
) -> None:
    if result.error is not None:
        print(f"[{index:>2}/{total}] {result.id} ERROR {result.error}")
        return

    expected_status = "ABSTAIN" if result.expected_abstain else "ANSWER"

    actual_status = "ABSTAIN" if result.actual_abstain else "ANSWER"

    source_status = ""

    if result.expected_source_cited is not None:
        source_status = f" source={'OK' if result.expected_source_cited else 'MISS'}"

    print(
        f"[{index:>2}/{total}] "
        f"{result.id} "
        f"expected={expected_status} "
        f"actual={actual_status} "
        f"lang="
        f"{'OK' if result.language_correct else 'FAIL'} "
        f"citation="
        f"{'YES' if result.citation_present else 'NO'}"
        f"{source_status} "
        f"attempts="
        f"{result.generation_attempts} "
        f"latency="
        f"{result.latency_seconds:.2f}s"
    )


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Cases: {summary['cases']}; backend errors: {summary['failed_cases']}")
    for name, value in summary["metrics"].items():
        formatted = "n/a" if value is None else f"{value:.1%}"
        print(f"  {name}: {formatted}")
    latency = summary["latency_seconds"]
    if latency["mean"] is not None:
        print(f"Latency: mean={latency['mean']:.2f}s, p95={latency['p95']:.2f}s")
