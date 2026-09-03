from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from support_rag.core.config import Settings
from support_rag.core.schemas import DocumentChunk, RankedChunk
from support_rag.retrieval.bm25 import BM25Retriever
from support_rag.retrieval.dense import DenseRetriever, QueryEncoder

RetrievalSource = Literal[
    "dense",
    "bm25_promoted",
    "bm25_backfill",
]


@dataclass(frozen=True)
class HybridResult:
    chunk: DocumentChunk
    rank: int

    dense_rank: int | None
    bm25_rank: int | None

    dense_score: float | None
    bm25_score: float | None

    source: RetrievalSource


@dataclass(frozen=True)
class _DocumentHit:
    chunk: DocumentChunk
    rank: int
    score: float


def _best_hit_per_document(
    hits: Sequence[RankedChunk],
) -> dict[str, _DocumentHit]:
    """
    Collapse chunk-level retrieval results into document-level hits.

    The first chunk encountered for a document is its representative
    because input hits are already ordered by the retriever.
    """
    documents: dict[str, _DocumentHit] = {}

    for hit in hits:
        document_id = hit.chunk.document_id

        if document_id in documents:
            continue

        documents[document_id] = _DocumentHit(
            chunk=hit.chunk,
            rank=len(documents) + 1,
            score=hit.score,
        )

    return documents


def dense_first_merge(
    dense_hits: Sequence[RankedChunk],
    bm25_hits: Sequence[RankedChunk],
    *,
    top_k: int,
    bm25_insert_position: int = 3,
    bm25_max_rank: int = 2,
    bm25_slots: int = 1,
) -> list[HybridResult]:
    """
    Build document-level retrieval results.

    Dense retrieval remains the primary ranking signal. BM25 may
    conservatively promote a small number of highly ranked lexical
    candidates without replacing the dense top-1 document.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if bm25_insert_position < 2:
        raise ValueError("bm25_insert_position must be >= 2 to preserve the dense top-1 result")

    if bm25_max_rank <= 0:
        raise ValueError("bm25_max_rank must be positive")

    if bm25_slots < 0:
        raise ValueError("bm25_slots cannot be negative")

    dense_documents = _best_hit_per_document(dense_hits)

    bm25_documents = _best_hit_per_document(bm25_hits)

    dense_order = list(dense_documents)

    final_order = dense_order.copy()

    dense_top_document = dense_order[0] if dense_order else None

    promotion_candidates = [
        document_id
        for document_id, hit in bm25_documents.items()
        if (hit.rank <= bm25_max_rank and document_id != dense_top_document)
    ]

    selected_candidates = promotion_candidates[:bm25_slots]

    insertion_index = min(
        bm25_insert_position - 1,
        len(final_order),
    )

    for offset, document_id in enumerate(selected_candidates):
        if document_id in final_order:
            final_order.remove(document_id)

        position = min(
            insertion_index + offset,
            len(final_order),
        )

        final_order.insert(
            position,
            document_id,
        )

    for document_id in bm25_documents:
        if document_id not in final_order:
            final_order.append(document_id)

    results: list[HybridResult] = []

    for rank, document_id in enumerate(
        final_order[:top_k],
        start=1,
    ):
        dense = dense_documents.get(document_id)

        bm25 = bm25_documents.get(document_id)

        if document_id in selected_candidates:
            source: RetrievalSource = "bm25_promoted"

        elif dense is not None:
            source = "dense"

        else:
            source = "bm25_backfill"

        if dense is not None:
            representative = dense.chunk

        elif bm25 is not None:
            representative = bm25.chunk

        else:
            continue

        results.append(
            HybridResult(
                chunk=representative,
                rank=rank,
                dense_rank=(dense.rank if dense is not None else None),
                bm25_rank=(bm25.rank if bm25 is not None else None),
                dense_score=(dense.score if dense is not None else None),
                bm25_score=(bm25.score if bm25 is not None else None),
                source=source,
            )
        )

    return results


def _rerank_results(
    results: Sequence[HybridResult],
) -> list[HybridResult]:
    """
    Reassign final context ranks while preserving original retrieval
    metadata such as dense/BM25 ranks and scores.
    """
    return [
        HybridResult(
            chunk=result.chunk,
            rank=rank,
            dense_rank=result.dense_rank,
            bm25_rank=result.bm25_rank,
            dense_score=result.dense_score,
            bm25_score=result.bm25_score,
            source=result.source,
        )
        for rank, result in enumerate(
            results,
            start=1,
        )
    ]


def expand_top_document_chunks(
    base_results: Sequence[HybridResult],
    dense_hits: Sequence[RankedChunk],
    bm25_hits: Sequence[RankedChunk],
    *,
    top_k: int,
    top_document_chunks: int = 3,
) -> list[HybridResult]:
    """
    Expand the strongest retrieved document into multiple chunks.

    Document retrieval answers "which article is relevant?". For long
    Knowledge Base articles that contain several independent procedures,
    a single representative chunk may not contain the section needed to
    answer the question.

    The top document is therefore allowed to contribute several strong
    chunks. Chunks from the same article are presented to the LLM in
    article order rather than similarity-score order.

    Other documents are used only when the top document does not provide
    enough candidate chunks to fill the final retrieval budget.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if top_document_chunks <= 0:
        raise ValueError("top_document_chunks must be positive")

    if not base_results:
        return []

    top_result = base_results[0]

    top_document_id = top_result.chunk.document_id

    dense_by_chunk_id = {hit.chunk.chunk_id: hit for hit in dense_hits}

    bm25_by_chunk_id = {hit.chunk.chunk_id: hit for hit in bm25_hits}

    same_document_candidates: list[HybridResult] = []

    seen_chunk_ids: set[str] = set()

    for dense_hit in dense_hits:
        if len(same_document_candidates) >= top_document_chunks:
            break

        chunk = dense_hit.chunk

        if chunk.document_id != top_document_id:
            continue

        if chunk.chunk_id in seen_chunk_ids:
            continue

        bm25_hit = bm25_by_chunk_id.get(chunk.chunk_id)

        same_document_candidates.append(
            HybridResult(
                chunk=chunk,
                rank=0,
                dense_rank=dense_hit.rank,
                bm25_rank=(bm25_hit.rank if bm25_hit is not None else None),
                dense_score=dense_hit.score,
                bm25_score=(bm25_hit.score if bm25_hit is not None else None),
                source="dense",
            )
        )

        seen_chunk_ids.add(chunk.chunk_id)

    if len(same_document_candidates) < top_document_chunks:
        for bm25_hit in bm25_hits:
            if len(same_document_candidates) >= top_document_chunks:
                break

            chunk = bm25_hit.chunk

            if chunk.document_id != top_document_id:
                continue

            if chunk.chunk_id in seen_chunk_ids:
                continue

            dense_hit = dense_by_chunk_id.get(chunk.chunk_id)

            same_document_candidates.append(
                HybridResult(
                    chunk=chunk,
                    rank=0,
                    dense_rank=(dense_hit.rank if dense_hit is not None else None),
                    bm25_rank=bm25_hit.rank,
                    dense_score=(dense_hit.score if dense_hit is not None else None),
                    bm25_score=bm25_hit.score,
                    source=("dense" if dense_hit is not None else "bm25_backfill"),
                )
            )

            seen_chunk_ids.add(chunk.chunk_id)

    same_document_candidates.sort(key=lambda result: result.chunk.chunk_index)

    selected = list(same_document_candidates[:top_k])

    if len(selected) < top_k:
        for result in base_results[1:]:
            if len(selected) >= top_k:
                break

            if result.chunk.chunk_id in seen_chunk_ids:
                continue

            selected.append(result)

            seen_chunk_ids.add(result.chunk.chunk_id)

    return _rerank_results(selected[:top_k])


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        document_embeddings: np.ndarray,
        query_encoder: QueryEncoder,
        settings: Settings,
        *,
        top_document_chunks: int = 3,
    ) -> None:
        if top_document_chunks <= 0:
            raise ValueError("top_document_chunks must be positive")
        self._settings = settings
        self._dense_top_k = settings.dense_top_k
        self._bm25_top_k = settings.bm25_top_k
        self._bm25_insert_position = settings.hybrid_bm25_insert_position
        self._bm25_max_rank = settings.hybrid_bm25_max_rank
        self._bm25_slots = settings.hybrid_bm25_slots
        self._top_document_chunks = top_document_chunks

        self._bm25 = BM25Retriever(chunks)

        self._dense = DenseRetriever(
            chunks=chunks,
            document_embeddings=document_embeddings,
            query_encoder=query_encoder,
        )

    def search(
        self,
        query: str,
    ) -> list[HybridResult]:
        query = query.strip()

        if not query:
            return []

        dense_hits = self._dense.search(
            query,
            top_k=self._dense_top_k,
        )

        bm25_hits = self._bm25.search(
            query,
            top_k=self._bm25_top_k,
        )

        document_results = dense_first_merge(
            dense_hits,
            bm25_hits,
            top_k=(self._settings.retrieval_top_k),
            bm25_insert_position=(self._bm25_insert_position),
            bm25_max_rank=(self._bm25_max_rank),
            bm25_slots=(self._bm25_slots),
        )

        return expand_top_document_chunks(
            document_results,
            dense_hits,
            bm25_hits,
            top_k=(self._settings.retrieval_top_k),
            top_document_chunks=(self._top_document_chunks),
        )
