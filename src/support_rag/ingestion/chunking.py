from __future__ import annotations

import hashlib
from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from support_rag.core.schemas import DocumentChunk, SourceDocument


class ChunkingError(RuntimeError):
    pass


def _stable_chunk_id(
    document_id: str,
    chunk_index: int,
    text: str,
) -> str:
    payload = f"{document_id}\0{chunk_index}\0{text}".encode()

    digest = hashlib.sha256(payload).hexdigest()[:16]

    return f"{document_id}:{chunk_index}:{digest}"


class TokenWindowChunker:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        chunk_size: int,
        overlap: int,
        model_max_length: int,
        document_prefix: str,
    ) -> None:
        if not tokenizer.is_fast:
            raise ChunkingError(
                "A fast tokenizer is required because chunk boundaries "
                "are reconstructed from token offsets."
            )

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self._tokenizer = tokenizer
        self._overlap = overlap

        prefix_token_count = len(
            tokenizer.encode(
                document_prefix,
                add_special_tokens=False,
            )
        )

        special_token_count = tokenizer.num_special_tokens_to_add(pair=False)

        payload_capacity = model_max_length - prefix_token_count - special_token_count

        if payload_capacity <= 0:
            raise ChunkingError(
                "Embedding prefix and special tokens consume the entire model input budget."
            )

        self._chunk_size = min(
            chunk_size,
            payload_capacity,
        )

        if self._overlap >= self._chunk_size:
            raise ChunkingError("Effective chunk size became smaller than configured overlap.")

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    def split_document(
        self,
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        text = document.text.strip()

        if not text:
            return []

        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            truncation=False,
        )

        offsets = encoded.get("offset_mapping")

        if offsets is None:
            raise ChunkingError("Tokenizer did not return offset_mapping.")

        token_offsets: Sequence[tuple[int, int]] = offsets

        if not token_offsets:
            return []

        step = self._chunk_size - self._overlap

        chunks: list[DocumentChunk] = []

        chunk_index = 0

        for token_start in range(
            0,
            len(token_offsets),
            step,
        ):
            token_end = min(
                token_start + self._chunk_size,
                len(token_offsets),
            )

            window = token_offsets[token_start:token_end]

            if not window:
                break

            char_start = window[0][0]
            char_end = window[-1][1]

            chunk_text = text[char_start:char_end].strip()

            if chunk_text:
                token_count = token_end - token_start

                chunks.append(
                    DocumentChunk(
                        chunk_id=_stable_chunk_id(
                            document.document_id,
                            chunk_index,
                            chunk_text,
                        ),
                        document_id=document.document_id,
                        title=document.title,
                        source_url=document.source_url,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        token_count=token_count,
                        metadata=document.metadata.copy(),
                    )
                )

                chunk_index += 1

            if token_end >= len(token_offsets):
                break

        return chunks

    def split_many(
        self,
        documents: Sequence[SourceDocument],
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks
