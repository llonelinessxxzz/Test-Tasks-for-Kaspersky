from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from support_rag.core.schemas import DocumentChunk
from support_rag.retrieval.dense import DenseRetriever


class FakeQueryEncoder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = np.asarray(vector, dtype=np.float32)

    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return np.repeat(
            self._vector[None, :],
            repeats=len(texts),
            axis=0,
        )


def make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        title=chunk_id,
        source_url=f"https://example.com/{chunk_id}",
        text=text,
        chunk_index=0,
        token_count=len(text.split()),
    )


def test_dense_retriever_ranks_by_cosine_similarity() -> None:
    chunks = [
        make_chunk("password", "Password recovery"),
        make_chunk("network", "Network troubleshooting"),
        make_chunk("update", "Application update"),
    ]

    document_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )

    retriever = DenseRetriever(
        chunks=chunks,
        document_embeddings=document_embeddings,
        query_encoder=FakeQueryEncoder([1.0, 0.0]),
    )

    hits = retriever.search(
        "How do I recover my password?",
        top_k=3,
    )

    assert hits[0].chunk.chunk_id == "password"
    assert hits[0].retriever == "dense"
    assert hits[0].rank == 1
    assert hits[0].score == pytest.approx(1.0)


def test_dense_retriever_rejects_embedding_count_mismatch() -> None:
    chunks = [
        make_chunk("a", "First"),
        make_chunk("b", "Second"),
    ]

    with pytest.raises(
        ValueError,
        match="Chunk count does not match embedding count",
    ):
        DenseRetriever(
            chunks=chunks,
            document_embeddings=np.asarray(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
            query_encoder=FakeQueryEncoder([1.0, 0.0]),
        )


def test_dense_retriever_returns_empty_result_for_blank_query() -> None:
    chunk = make_chunk("a", "First")

    retriever = DenseRetriever(
        chunks=[chunk],
        document_embeddings=np.asarray(
            [[1.0, 0.0]],
            dtype=np.float32,
        ),
        query_encoder=FakeQueryEncoder([1.0, 0.0]),
    )

    assert retriever.search("   ", top_k=1) == []
