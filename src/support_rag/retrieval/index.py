from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from support_rag.core.schemas import DocumentChunk


@dataclass(frozen=True)
class IndexManifest:
    schema_version: int

    embedding_model: str
    embedding_dimension: int
    embedding_max_length: int
    document_prefix: str

    retrieval_text_version: int

    chunk_count: int
    chunk_ids_sha256: str


def chunk_ids_fingerprint(
    chunks: list[DocumentChunk],
) -> str:
    hasher = hashlib.sha256()

    for chunk in chunks:
        hasher.update(chunk.chunk_id.encode())
        hasher.update(b"\0")

    return hasher.hexdigest()


def save_dense_index(
    embeddings: np.ndarray,
    manifest: IndexManifest,
    index_dir: Path,
) -> None:
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape={embeddings.shape}")

    if embeddings.shape[0] != manifest.chunk_count:
        raise ValueError("Manifest chunk count does not match embedding matrix")

    if embeddings.shape[1] != manifest.embedding_dimension:
        raise ValueError("Manifest embedding dimension does not match embedding matrix")

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        index_dir / "embeddings.npy",
        embeddings,
        allow_pickle=False,
    )

    manifest_path = index_dir / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_dense_index(
    chunks: list[DocumentChunk],
    index_dir: Path,
    *,
    expected_model: str,
    expected_max_length: int,
    expected_document_prefix: str,
) -> np.ndarray:
    embeddings_path = index_dir / "embeddings.npy"

    manifest_path = index_dir / "manifest.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest = IndexManifest(**manifest_data)

    if manifest.schema_version != 1:
        raise RuntimeError(f"Unsupported dense index schema version: {manifest.schema_version}")

    if manifest.retrieval_text_version != 1:
        raise RuntimeError("Dense index was built with another retrieval text representation")

    if manifest.embedding_model != expected_model:
        raise RuntimeError(f"Dense index uses another model: {manifest.embedding_model!r}")

    if manifest.embedding_max_length != expected_max_length:
        raise RuntimeError("Dense index was built with another max sequence length")

    if manifest.document_prefix != expected_document_prefix:
        raise RuntimeError("Dense index was built with another document prefix")

    if manifest.chunk_count != len(chunks):
        raise RuntimeError(
            f"Dense index is stale: {manifest.chunk_count} indexed chunks vs {len(chunks)} current"
        )

    expected_fingerprint = chunk_ids_fingerprint(chunks)

    if expected_fingerprint != manifest.chunk_ids_sha256:
        raise RuntimeError("Dense index is stale: chunk IDs or their order changed")

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    if embeddings.ndim != 2:
        raise RuntimeError(f"Invalid embedding matrix: {embeddings.shape}")

    if embeddings.shape[0] != len(chunks):
        raise RuntimeError("Embedding row count does not match chunk count")

    if embeddings.shape[1] != manifest.embedding_dimension:
        raise RuntimeError("Embedding dimension does not match manifest")

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )
