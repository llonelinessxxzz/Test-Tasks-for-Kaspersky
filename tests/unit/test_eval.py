from __future__ import annotations

from types import SimpleNamespace

import pytest

from eval.data import GenerationEvalCase
from eval.metrics import evaluate_case, summarize_generation
from eval.run import reserve_output
from support_rag.core.schemas import RAGResponse


async def test_backend_errors_are_not_removed_from_rate_denominators() -> None:
    class Service:
        async def ask(self, question):
            if question == "fail":
                raise RuntimeError("Backend unavailable")
            return RAGResponse(
                answer="A supported answer",
                model="test",
                context_tokens=0,
                abstained=False,
                finish_reason="stop",
            )

    runtime = SimpleNamespace(service=Service())
    results = []
    for index, (question, expected_abstain) in enumerate(
        [("ok", False), ("fail", False), ("fail", True)]
    ):
        case = GenerationEvalCase(
            id=str(index),
            question=question,
            language="en",
            expected_abstain=expected_abstain,
            expected_document_ids=[] if expected_abstain else ["doc"],
        )
        results.append(await evaluate_case(runtime=runtime, case=case))
    summary = summarize_generation(results)
    assert summary["failed_cases"] == 2
    assert summary["supported_cases"] == 2
    assert summary["out_of_kb_cases"] == 1
    assert summary["metrics"]["supported_answer_rate"] == 0.5
    assert summary["metrics"]["out_of_kb_abstention_rate"] == 0.0
    assert summary["case_groups"]["technical_errors"] == ["1", "2"]


def test_existing_results_cannot_be_overwritten(tmp_path) -> None:
    previous = tmp_path / "manual_review.jsonl"
    previous.write_text("human review", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        reserve_output(tmp_path)
    assert previous.read_text(encoding="utf-8") == "human review"
    assert not (tmp_path / "run_manifest.json").exists()


def test_second_runner_cannot_claim_the_same_directory(tmp_path) -> None:
    reserve_output(tmp_path)
    manifest = (tmp_path / "run_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        reserve_output(tmp_path)
    assert (tmp_path / "run_manifest.json").read_bytes() == manifest
