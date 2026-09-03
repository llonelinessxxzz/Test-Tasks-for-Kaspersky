from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from support_rag.core.schemas import (
    DocumentChunk,
    SourceDocument,
)


class CorpusFormatError(ValueError):
    pass


def _parse_json_line(
    line: str,
    *,
    source_path: Path,
    line_number: int,
) -> SourceDocument:
    try:
        payload: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        raise CorpusFormatError(
            f"Invalid JSON in {source_path} at line {line_number}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise CorpusFormatError(f"Expected JSON object in {source_path} at line {line_number}")

    try:
        return SourceDocument.model_validate(payload)
    except ValidationError as exc:
        raise CorpusFormatError(
            f"Invalid document schema in {source_path} at line {line_number}: {exc}"
        ) from exc


def load_jsonl(
    path: Path,
) -> list[SourceDocument]:
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise CorpusFormatError(f"Corpus path is not a file: {path}")

    documents: list[SourceDocument] = []
    seen_document_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            document = _parse_json_line(
                line,
                source_path=path,
                line_number=line_number,
            )

            if document.document_id in seen_document_ids:
                raise CorpusFormatError(
                    f"Duplicate document_id "
                    f"{document.document_id!r} "
                    f"in {path} at line {line_number}"
                )

            seen_document_ids.add(document.document_id)

            documents.append(document)

    if not documents:
        raise CorpusFormatError(f"Corpus is empty: {path}")

    return documents


def write_jsonl(
    documents: Iterable[SourceDocument],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            record = document.model_dump(mode="json")

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def load_chunks_jsonl(
    path: Path,
) -> list[DocumentChunk]:
    if not path.exists():
        raise FileNotFoundError(path)

    chunks: list[DocumentChunk] = []
    seen_chunk_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)

                chunk = DocumentChunk.model_validate(payload)

            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                raise CorpusFormatError(
                    f"Invalid chunk in {path} at line {line_number}: {exc}"
                ) from exc

            if chunk.chunk_id in seen_chunk_ids:
                raise CorpusFormatError(
                    f"Duplicate chunk_id {chunk.chunk_id!r} in {path} at line {line_number}"
                )

            seen_chunk_ids.add(chunk.chunk_id)

            chunks.append(chunk)

    if not chunks:
        raise CorpusFormatError(f"Chunk corpus is empty: {path}")

    return chunks


def write_chunks_jsonl(
    chunks: Iterable[DocumentChunk],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk.model_dump(mode="json"),
                    ensure_ascii=False,
                )
            )

            file.write("\n")
