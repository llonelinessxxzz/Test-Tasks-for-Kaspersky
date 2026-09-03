"""Evaluate the deployed hybrid retriever and the UI orchestration code inside Docker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from eval.data import GenerationEvalCase, RetrievalEvalCase, load_cases
from eval.manual_review import write_review
from eval.metrics import evaluate_case, evaluate_ranking, print_case_result, summarize_generation
from support_rag.core.config import Settings
from support_rag.web.config import WebSettings
from support_rag.web.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def save(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


async def run(output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "transport": "Docker service DNS / HTTP",
        "split": "curated-dev",
        "note": "Selected development scenarios, not an independent holdout or correctness score.",
    }
    manifest_path = output / "run_manifest.json"
    save(manifest_path, manifest)
    orchestrator = Orchestrator(Settings(), WebSettings())
    try:
        manifest["health"] = await orchestrator.health()
        if not all(manifest["health"].values()):
            raise RuntimeError("Docker services are not ready")
        generation_file = ROOT / "eval/datasets/generation_eval.jsonl"
        retrieval_file = ROOT / "eval/datasets/retrieval_eval.jsonl"
        files = [
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("eval/*.py"),
            generation_file,
            retrieval_file,
        ]
        hashes = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
        manifest["sha256"] = hashes
        rows = []
        for case in load_cases(retrieval_file, RetrievalEvalCase):
            result = await orchestrator.retrieve(case.question)
            expected = set(case.relevant_document_ids)
            rows.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "language": case.language,
                    "expected_document_ids": case.relevant_document_ids,
                    "retrieved_document_ids": result.retrieved_document_ids,
                    "context_document_ids": result.context_document_ids,
                    "context_chunk_ids": result.context_chunk_ids,
                    "context_tokens": result.token_count,
                    "metrics": evaluate_ranking(result.retrieved_document_ids, expected),
                    "expected_article_in_context": bool(
                        expected.intersection(result.context_document_ids)
                    ),
                }
            )
        retrieval = {
            "retriever": "deployed dense-first hybrid",
            "cases_count": len(rows),
            "context_coverage": mean(row["expected_article_in_context"] for row in rows),
            "overall": {
                key: mean(row["metrics"][key] for row in rows) for key in rows[0]["metrics"]
            },
            "cases": rows,
        }
        save(output / "retrieval_current.json", retrieval)
        print(f"Retrieval context coverage: {retrieval['context_coverage']:.1%}", flush=True)

        class Service:
            async def ask(self, question):
                return (await orchestrator.ask(question)).response

        runtime = SimpleNamespace(service=Service())
        cases = load_cases(generation_file, GenerationEvalCase)
        results = []
        for index, case in enumerate(cases, 1):
            result = await evaluate_case(runtime=runtime, case=case)
            results.append(result)
            print_case_result(index, len(cases), result)
        summary = summarize_generation(results)
        save(
            output / "generation_eval.json",
            {
                "summary": summary,
                "cases": [asdict(row) for row in results],
                "note": manifest["note"],
            },
        )
        write_review([asdict(row) for row in results], output)
        assert hashes == {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }, "Evaluator changed during the run"
        manifest["status"] = "completed" if summary["failed_cases"] == 0 else "failed"
        manifest["failed_cases"] = summary["failed_cases"]
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if summary["failed_cases"] == 0 else 1
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await orchestrator.close()
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path("/app/state/eval") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output-dir", type=Path, default=default)
    args = parser.parse_args()
    print(f"Report: {args.output_dir}", flush=True)
    raise SystemExit(asyncio.run(run(args.output_dir)))


if __name__ == "__main__":
    main()
