"""Evaluate the application runtime and save an isolated, reproducible report."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Any

from eval.data import GenerationEvalCase, RetrievalEvalCase, load_cases
from eval.manual_review import write_review
from eval.metrics import (
    evaluate_case,
    evaluate_ranking,
    print_case_result,
    print_summary,
    summarize_generation,
)
from support_rag.core.config import get_settings
from support_rag.core.logging import configure_logging
from support_rag.ingestion.loader import load_chunks_jsonl
from support_rag.runtime import build_runtime

ROOT = Path(__file__).resolve().parents[1]
NOTE = (
    "Automated contract evaluation, not factual answer correctness. "
    "Source IDs/metadata do not prove that a cited source supports every claim. "
    "Correctness, groundedness and citation support require independent manual review. "
    "The bundled generation dataset is a curated development smoke set selected "
    "using known outcomes; it is not an unbiased benchmark or an untouched holdout."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hashes(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in sorted(set(paths)):
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = digest.hexdigest()
    return result


def reserve_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(item.name != ".gitkeep" for item in path.iterdir()):
        raise FileExistsError(
            f"{path} is not empty. Choose a new --output-dir; existing runs are preserved."
        )
    with (path / "run_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"status": "starting", "started_at": utc_now()}, stream)


def evaluate_current_retrieval(runtime: Any, cases: list[Any]) -> dict[str, Any]:
    rows = []
    for case in cases:
        hits = runtime.service._retriever.search(case.question)
        context = runtime.service._context_builder.build(hits)
        documents = list(dict.fromkeys(hit.chunk.document_id for hit in hits))
        context_documents = list(dict.fromkeys(source.document_id for source in context.sources))
        expected = set(case.relevant_document_ids)
        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "language": case.language,
                "expected_document_ids": case.relevant_document_ids,
                "retrieved_document_ids": documents,
                "context_document_ids": context_documents,
                "context_chunk_ids": [source.chunk_id for source in context.sources],
                "context_tokens": context.token_count,
                "metrics": evaluate_ranking(documents, expected),
                "expected_article_in_context": bool(expected.intersection(context_documents)),
            }
        )
    return {
        "retriever": "runtime dense-first hybrid with top-document chunk expansion",
        "cases_count": len(rows),
        "overall": {key: mean(row["metrics"][key] for row in rows) for key in rows[0]["metrics"]},
        "context_coverage": mean(row["expected_article_in_context"] for row in rows),
        "note": "Ranking metrics use the exact retriever and context builder used for generation.",
        "cases": rows,
    }


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    generation_cases = load_cases(args.generation_dataset, GenerationEvalCase)
    retrieval_cases = load_cases(args.retrieval_dataset, RetrievalEvalCase)
    chunks = load_chunks_jsonl(settings.processed_data_dir / "chunks.jsonl")
    known_documents = {chunk.document_id for chunk in chunks}
    for case in [*generation_cases, *retrieval_cases]:
        expected = (
            case.expected_document_ids
            if hasattr(case, "expected_document_ids")
            else case.relevant_document_ids
        )
        if missing := set(expected) - known_documents:
            raise ValueError(f"{case.id}: documents absent from corpus: {sorted(missing)}")

    paths = [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("eval/*.py"),
        ROOT / "pyproject.toml",
        args.generation_dataset.resolve(),
        args.retrieval_dataset.resolve(),
        (settings.processed_data_dir / "chunks.jsonl").resolve(),
        (settings.index_dir / "embeddings.npy").resolve(),
        (settings.index_dir / "manifest.json").resolve(),
    ]
    if (ROOT / ".env").exists():
        paths.append(ROOT / ".env")
    hashes = file_hashes(paths)
    config = settings.model_dump(mode="json", exclude={"llm_base_url", "model_cache_dir"})
    manifest = {
        "status": "running",
        "started_at": utc_now(),
        "split": args.split,
        "python": platform.python_version(),
        "config": config,
        "sha256": hashes,
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "numpy",
                "pydantic",
                "httpx",
            )
        },
        "generation_dataset": str(args.generation_dataset),
        "retrieval_dataset": str(args.retrieval_dataset),
        "generation_cases": len(generation_cases),
        "retrieval_cases": len(retrieval_cases),
        "corpus_documents": len(known_documents),
        "corpus_chunks": len(chunks),
        "generation_seed": None,
        "note": NOTE,
    }
    reserve_output(args.output_dir)
    manifest_path = args.output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    runtime = None
    try:
        print(f"Report: {args.output_dir}", flush=True)
        print("Loading RAG runtime...", flush=True)
        runtime = build_runtime(settings)
        if not await runtime.llm_client.is_ready():
            raise RuntimeError("Configured vLLM model is not ready")
        manifest["served_models"] = await runtime.llm_client.list_models()
        manifest["top_document_chunks"] = runtime.service._retriever._top_document_chunks
        print(f"Runtime ready. Retrieval: {len(retrieval_cases)} cases.", flush=True)
        retrieval = evaluate_current_retrieval(runtime, retrieval_cases)
        write_json(args.output_dir / "retrieval_current.json", retrieval)
        print(
            f"Context coverage on retrieval dataset: {retrieval['context_coverage']:.1%}",
            flush=True,
        )

        results = []
        with (args.output_dir / "generation_cases.jsonl").open("x", encoding="utf-8") as stream:
            for index, case in enumerate(generation_cases, 1):
                result = await evaluate_case(runtime=runtime, case=case)
                results.append(result)
                stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                stream.flush()
                print_case_result(index, len(generation_cases), result)
        summary = summarize_generation(results)
        write_json(
            args.output_dir / "generation_eval.json",
            {
                "dataset": str(args.generation_dataset),
                "split": args.split,
                "summary": summary,
                "cases": [asdict(row) for row in results],
                "manual_evaluation_note": NOTE,
            },
        )
        write_review([asdict(row) for row in results], args.output_dir)
        print_summary(summary)
        print(json.dumps(summary["case_groups"], ensure_ascii=False, indent=2), flush=True)
        if file_hashes(paths) != hashes:
            raise RuntimeError("Frozen source, settings, data or evaluator changed during the run")
        manifest["frozen_inputs_unchanged"] = True
        manifest["status"] = "completed" if not summary["failed_cases"] else "failed"
        manifest["failed_cases"] = summary["failed_cases"]
        return 0 if not summary["failed_cases"] else 1
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if runtime is not None:
                await runtime.close()
        finally:
            manifest["finished_at"] = utc_now()
            write_json(manifest_path, manifest)


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New empty report directory")
    parser.add_argument(
        "--generation-dataset", type=Path, default=Path("eval/datasets/generation_eval.jsonl")
    )
    parser.add_argument(
        "--retrieval-dataset", type=Path, default=Path("eval/datasets/retrieval_eval.jsonl")
    )
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = Path("eval/results") / stamp
    if args.split == "holdout" and (
        args.generation_dataset.resolve() == ROOT / "eval/datasets/generation_eval.jsonl"
        or args.retrieval_dataset.resolve() == ROOT / "eval/datasets/retrieval_eval.jsonl"
    ):
        parser.error("Holdout requires separate untouched datasets, not the development files")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
