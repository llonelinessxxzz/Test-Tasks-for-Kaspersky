from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from support_rag.core.config import Settings
from support_rag.core.schemas import (
    DocumentChunk,
    RankedChunk,
)
from support_rag.retrieval.text import (
    chunk_retrieval_text,
)


class QueryEncoder(Protocol):
    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> np.ndarray: ...


def _prefix_text(
    prefix: str,
    text: str,
) -> str:
    return f"{prefix.rstrip()} {text.strip()}"


def _normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={matrix.shape}")

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("Embedding matrix contains zero-length vectors")

    return matrix / norms


class RoSBERTaEncoder:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        cache_dir = settings.model_cache_dir.resolve()

        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        os.environ["HF_HOME"] = str(cache_dir)

        os.environ["HF_HUB_CACHE"] = str(cache_dir)

        os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")

        from sentence_transformers import (
            SentenceTransformer,
        )

        self._settings = settings

        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
            cache_folder=str(cache_dir),
        )

        self._model.max_seq_length = settings.embedding_max_length

    @property
    def embedding_dimension(
        self,
    ) -> int:
        dimension = self._model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError("Embedding model did not expose its output dimension")

        return dimension

    def _encode(
        self,
        texts: Sequence[str],
        *,
        prefix: str,
    ) -> np.ndarray:
        if not texts:
            return np.empty(
                (
                    0,
                    self.embedding_dimension,
                ),
                dtype=np.float32,
            )

        prepared = [
            _prefix_text(
                prefix,
                text,
            )
            for text in texts
        ]

        embeddings = self._model.encode(
            prepared,
            batch_size=(self._settings.embedding_batch_size),
            show_progress_bar=(len(prepared) > self._settings.embedding_batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return self._encode(
            texts,
            prefix=(self._settings.embedding_document_prefix),
        )

    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return self._encode(
            texts,
            prefix=(self._settings.embedding_query_prefix),
        )


class DenseRetriever:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        document_embeddings: np.ndarray,
        query_encoder: QueryEncoder,
    ) -> None:
        if not chunks:
            raise ValueError("Dense index cannot be built from an empty corpus")

        embeddings = _normalize_rows(document_embeddings)

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Chunk count does not match "
                "embedding count: "
                f"{len(chunks)} != "
                f"{embeddings.shape[0]}"
            )

        self._chunks = list(chunks)
        self._embeddings = embeddings
        self._query_encoder = query_encoder

    @classmethod
    def build(
        cls,
        chunks: Sequence[DocumentChunk],
        encoder: RoSBERTaEncoder,
    ) -> DenseRetriever:
        if not chunks:
            raise ValueError("Dense index cannot be built from an empty corpus")

        retrieval_texts = [chunk_retrieval_text(chunk) for chunk in chunks]

        embeddings = encoder.encode_documents(retrieval_texts)

        return cls(
            chunks=chunks,
            document_embeddings=embeddings,
            query_encoder=encoder,
        )

    @property
    def embeddings(
        self,
    ) -> np.ndarray:
        return self._embeddings

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[RankedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = query.strip()

        if not query:
            return []

        query_embedding = self._query_encoder.encode_queries([query])

        expected_shape = (
            1,
            self._embeddings.shape[1],
        )

        if query_embedding.shape != expected_shape:
            raise ValueError(
                "Query embedding dimension "
                "does not match document "
                "embeddings: "
                f"{query_embedding.shape} != "
                f"{expected_shape}"
            )

        query_vector = _normalize_rows(query_embedding)[0]

        scores = self._embeddings @ query_vector

        limit = min(
            top_k,
            len(self._chunks),
        )

        ranked_indices = np.argsort(
            -scores,
            kind="stable",
        )[:limit]

        return [
            RankedChunk(
                chunk=self._chunks[int(chunk_index)],
                retriever="dense",
                rank=rank,
                score=float(scores[chunk_index]),
            )
            for rank, chunk_index in enumerate(
                ranked_indices,
                start=1,
            )
        ]
