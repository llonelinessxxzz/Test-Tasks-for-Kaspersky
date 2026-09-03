from __future__ import annotations

import json
from pathlib import Path

import pytest

from support_rag.ingestion.loader import (
    CorpusFormatError,
    load_jsonl,
)


def test_load_jsonl_reads_valid_documents(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"

    rows = [
        {
            "document_id": "doc-1",
            "title": "First",
            "source_url": "https://example.com/1",
            "text": "First document text.",
            "language": "en",
            "metadata": {},
        },
        {
            "document_id": "doc-2",
            "title": "Second",
            "source_url": "https://example.com/2",
            "text": "Second document text.",
            "language": "en",
            "metadata": {},
        },
    ]

    corpus.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    documents = load_jsonl(corpus)

    assert len(documents) == 2
    assert documents[0].document_id == "doc-1"
    assert documents[1].document_id == "doc-2"


def test_load_jsonl_rejects_duplicate_document_ids(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"

    row = {
        "document_id": "duplicate",
        "title": "Article",
        "source_url": "https://example.com/article",
        "text": "Content",
    }

    corpus.write_text(
        "\n".join(
            [
                json.dumps(row),
                json.dumps(row),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        CorpusFormatError,
        match="Duplicate document_id",
    ):
        load_jsonl(corpus)
