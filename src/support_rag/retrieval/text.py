from __future__ import annotations

from support_rag.core.schemas import DocumentChunk


def chunk_retrieval_text(chunk: DocumentChunk) -> str:
    title = chunk.title.strip()
    text = chunk.text.strip()

    if not title:
        return text

    return f"{title}\n\n{text}"
