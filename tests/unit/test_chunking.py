from __future__ import annotations

from typing import Any

from support_rag.core.schemas import SourceDocument
from support_rag.ingestion.chunking import TokenWindowChunker


class WhitespaceTokenizer:
    is_fast = True

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        del add_special_tokens

        return list(range(len(text.split())))

    def num_special_tokens_to_add(
        self,
        *,
        pair: bool,
    ) -> int:
        del pair

        return 2

    def __call__(
        self,
        text: str,
        **_: Any,
    ) -> dict[str, list[tuple[int, int]]]:
        offsets: list[tuple[int, int]] = []

        position = 0

        for word in text.split():
            start = text.index(
                word,
                position,
            )

            end = start + len(word)

            offsets.append((start, end))

            position = end

        return {
            "offset_mapping": offsets,
        }


def test_chunker_preserves_overlap() -> None:
    tokenizer = WhitespaceTokenizer()

    chunker = TokenWindowChunker(
        tokenizer,
        chunk_size=4,
        overlap=1,
        model_max_length=32,
        document_prefix="search_document:",
    )

    document = SourceDocument(
        document_id="doc-1",
        title="Example",
        source_url="https://example.com",
        text=("one two three four five six seven eight"),
    )

    chunks = chunker.split_document(document)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight",
    ]

    assert [chunk.chunk_index for chunk in chunks] == [
        0,
        1,
        2,
    ]


def test_chunk_ids_are_deterministic() -> None:
    tokenizer = WhitespaceTokenizer()

    chunker = TokenWindowChunker(
        tokenizer,
        chunk_size=4,
        overlap=1,
        model_max_length=32,
        document_prefix="search_document:",
    )

    document = SourceDocument(
        document_id="doc-1",
        title="Example",
        source_url="https://example.com",
        text="one two three four five",
    )

    first = chunker.split_document(document)

    second = chunker.split_document(document)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
