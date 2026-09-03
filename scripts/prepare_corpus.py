from __future__ import annotations

import argparse
from pathlib import Path

import structlog
from transformers import AutoTokenizer

from support_rag.core.config import get_settings
from support_rag.core.logging import configure_logging
from support_rag.ingestion.chunking import (
    TokenWindowChunker,
)
from support_rag.ingestion.loader import (
    load_jsonl,
    write_chunks_jsonl,
)

log = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Validate and chunk the raw Customer Support corpus.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = get_settings()

    configure_logging(settings.log_level)

    output_path = (
        args.output if args.output is not None else (settings.processed_data_dir / "chunks.jsonl")
    )

    documents = load_jsonl(args.input)

    log.info(
        "corpus_loaded",
        path=str(args.input),
        documents=len(documents),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        settings.embedding_model,
        use_fast=True,
        cache_dir=settings.model_cache_dir,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(f"{settings.embedding_model} did not provide a fast tokenizer")

    chunker = TokenWindowChunker(
        tokenizer,
        chunk_size=(settings.chunk_size_tokens),
        overlap=(settings.chunk_overlap_tokens),
        model_max_length=(settings.embedding_max_length),
        document_prefix=(settings.embedding_document_prefix),
    )

    chunks = chunker.split_many(documents)

    if not chunks:
        raise RuntimeError("Chunking produced no chunks")

    write_chunks_jsonl(
        chunks,
        output_path,
    )

    log.info(
        "corpus_prepared",
        input=str(args.input),
        output=str(output_path),
        documents=len(documents),
        chunks=len(chunks),
        effective_chunk_size=(chunker.chunk_size),
        overlap=chunker.overlap,
        tokenizer=(type(tokenizer).__name__),
    )


if __name__ == "__main__":
    main()
