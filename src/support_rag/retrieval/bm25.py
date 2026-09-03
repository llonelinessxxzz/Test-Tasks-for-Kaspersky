from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from support_rag.core.schemas import DocumentChunk, RankedChunk
from support_rag.retrieval.text import chunk_retrieval_text

_TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


class BM25Retriever:
    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("BM25 index cannot be built from an empty corpus")

        self._chunks = list(chunks)

        self._tokenized_corpus = [
            tokenize_for_bm25(chunk_retrieval_text(chunk)) for chunk in self._chunks
        ]

        self._vocabulary = {
            token for document_tokens in self._tokenized_corpus for token in document_tokens
        }

        self._index = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[RankedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = tokenize_for_bm25(query)

        if not query_tokens:
            return []

        if not any(token in self._vocabulary for token in query_tokens):
            return []

        scores = np.asarray(
            self._index.get_scores(query_tokens),
            dtype=np.float64,
        )

        limit = min(top_k, len(self._chunks))

        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )[:limit]

        return [
            RankedChunk(
                chunk=self._chunks[int(chunk_index)],
                retriever="bm25",
                rank=rank,
                score=float(scores[chunk_index]),
            )
            for rank, chunk_index in enumerate(
                ranked_indices,
                start=1,
            )
        ]
