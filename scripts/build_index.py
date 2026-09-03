from __future__ import annotations

from time import perf_counter

from support_rag.core.config import get_settings
from support_rag.ingestion.loader import load_chunks_jsonl
from support_rag.retrieval.dense import RoSBERTaEncoder
from support_rag.retrieval.index import (
    IndexManifest,
    chunk_ids_fingerprint,
    save_dense_index,
)
from support_rag.retrieval.text import chunk_retrieval_text


def main() -> None:
    settings = get_settings()

    chunks_path = settings.processed_data_dir / "chunks.jsonl"

    chunks = load_chunks_jsonl(chunks_path)

    print(f"Chunks: {len(chunks)}")

    print(f"Model: {settings.embedding_model}")

    print(f"Device: {settings.embedding_device}")

    encoder = RoSBERTaEncoder(settings)

    retrieval_texts = [chunk_retrieval_text(chunk) for chunk in chunks]

    started = perf_counter()

    embeddings = encoder.encode_documents(retrieval_texts)

    elapsed = perf_counter() - started

    manifest = IndexManifest(
        schema_version=1,
        embedding_model=(settings.embedding_model),
        embedding_dimension=(embeddings.shape[1]),
        embedding_max_length=(settings.embedding_max_length),
        document_prefix=(settings.embedding_document_prefix),
        retrieval_text_version=1,
        chunk_count=len(chunks),
        chunk_ids_sha256=(chunk_ids_fingerprint(chunks)),
    )

    save_dense_index(
        embeddings,
        manifest,
        settings.index_dir,
    )

    print(f"Encoded {len(chunks)} chunks in {elapsed:.2f}s")

    print(f"Matrix: {embeddings.shape}")

    print(f"Saved: {settings.index_dir}")


if __name__ == "__main__":
    main()
