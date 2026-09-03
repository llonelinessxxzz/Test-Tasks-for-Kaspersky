from __future__ import annotations

from typing import Literal

from support_rag.core.schemas import (
    DocumentChunk,
    RankedChunk,
)
from support_rag.retrieval.hybrid import (
    dense_first_merge,
)


def make_chunk(
    document_id: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{document_id}-chunk",
        document_id=document_id,
        title=document_id,
        source_url=(f"https://example.com/{document_id}"),
        text=f"Content for {document_id}",
        chunk_index=0,
        token_count=3,
    )


def make_hit(
    document_id: str,
    *,
    retriever: Literal["bm25", "dense"],
    rank: int,
    score: float,
) -> RankedChunk:
    return RankedChunk(
        chunk=make_chunk(document_id),
        retriever=retriever,
        rank=rank,
        score=score,
    )


def test_dense_first_preserves_dense_top_one() -> None:
    dense_hits = [
        make_hit(
            "dense-a",
            retriever="dense",
            rank=1,
            score=0.95,
        ),
        make_hit(
            "dense-b",
            retriever="dense",
            rank=2,
            score=0.90,
        ),
        make_hit(
            "dense-c",
            retriever="dense",
            rank=3,
            score=0.85,
        ),
    ]

    bm25_hits = [
        make_hit(
            "bm25-x",
            retriever="bm25",
            rank=1,
            score=8.0,
        ),
    ]

    results = dense_first_merge(
        dense_hits,
        bm25_hits,
        top_k=4,
        bm25_insert_position=2,
        bm25_max_rank=1,
        bm25_slots=1,
    )

    assert results[0].chunk.document_id == "dense-a"

    assert results[1].chunk.document_id == "bm25-x"


def test_dense_first_can_promote_existing_dense_document() -> None:
    dense_hits = [
        make_hit(
            "a",
            retriever="dense",
            rank=1,
            score=0.95,
        ),
        make_hit(
            "b",
            retriever="dense",
            rank=2,
            score=0.90,
        ),
        make_hit(
            "c",
            retriever="dense",
            rank=3,
            score=0.85,
        ),
        make_hit(
            "target",
            retriever="dense",
            rank=4,
            score=0.80,
        ),
    ]

    bm25_hits = [
        make_hit(
            "target",
            retriever="bm25",
            rank=1,
            score=10.0,
        ),
    ]

    results = dense_first_merge(
        dense_hits,
        bm25_hits,
        top_k=4,
        bm25_insert_position=2,
        bm25_max_rank=1,
        bm25_slots=1,
    )

    assert [result.chunk.document_id for result in results] == [
        "a",
        "target",
        "b",
        "c",
    ]

    assert results[1].source == "bm25_promoted"


def test_dense_first_does_not_duplicate_documents() -> None:
    dense_hits = [
        make_hit(
            "a",
            retriever="dense",
            rank=1,
            score=0.95,
        ),
        make_hit(
            "b",
            retriever="dense",
            rank=2,
            score=0.90,
        ),
    ]

    bm25_hits = [
        make_hit(
            "b",
            retriever="bm25",
            rank=1,
            score=9.0,
        ),
    ]

    results = dense_first_merge(
        dense_hits,
        bm25_hits,
        top_k=5,
        bm25_insert_position=2,
        bm25_max_rank=1,
        bm25_slots=1,
    )

    document_ids = [result.chunk.document_id for result in results]

    assert len(document_ids) == len(set(document_ids))
