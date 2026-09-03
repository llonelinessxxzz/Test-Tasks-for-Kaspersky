from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from support_rag.retrieval.hybrid import HybridResult


class TokenizerLike(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...


@dataclass(frozen=True)
class ContextSource:
    source_number: int
    document_id: str
    chunk_id: str
    title: str
    source_url: str
    text: str

    dense_rank: int | None
    bm25_rank: int | None
    retrieval_source: str


@dataclass(frozen=True)
class BuiltContext:
    text: str
    sources: tuple[ContextSource, ...]
    token_count: int


def _source_header(
    source_number: int,
    *,
    title: str,
    source_url: str,
) -> str:
    del source_url

    return f"[SOURCE {source_number}]\nTitle: {title}\nContent:\n"


def _count_tokens(
    tokenizer: TokenizerLike,
    text: str,
) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def _truncate_to_tokens(
    tokenizer: TokenizerLike,
    text: str,
    *,
    max_tokens: int,
) -> str:
    if max_tokens <= 0:
        return ""

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(token_ids) <= max_tokens:
        return text.strip()

    return tokenizer.decode(
        token_ids[:max_tokens],
        skip_special_tokens=True,
    ).strip()


class ContextBuilder:
    def __init__(
        self,
        tokenizer: TokenizerLike,
        *,
        max_tokens: int,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    def build(
        self,
        hits: Sequence[HybridResult],
    ) -> BuiltContext:
        if not hits:
            return BuiltContext(
                text="",
                sources=(),
                token_count=0,
            )

        blocks: list[str] = []
        sources: list[ContextSource] = []

        used_tokens = 0

        for source_number, hit in enumerate(
            hits,
            start=1,
        ):
            chunk = hit.chunk

            header = _source_header(
                source_number,
                title=chunk.title,
                source_url=chunk.source_url,
            )

            separator = "\n\n" if blocks else ""

            fixed_text = separator + header

            fixed_tokens = _count_tokens(
                self._tokenizer,
                fixed_text,
            )

            remaining_tokens = self._max_tokens - used_tokens - fixed_tokens

            if remaining_tokens <= 0:
                break

            body = _truncate_to_tokens(
                self._tokenizer,
                chunk.text,
                max_tokens=remaining_tokens,
            )

            if not body:
                continue

            block = fixed_text + body

            block_tokens = _count_tokens(
                self._tokenizer,
                block,
            )

            if used_tokens + block_tokens > self._max_tokens:
                remaining_tokens = self._max_tokens - used_tokens - fixed_tokens

                body = _truncate_to_tokens(
                    self._tokenizer,
                    chunk.text,
                    max_tokens=max(
                        remaining_tokens - 1,
                        0,
                    ),
                )

                if not body:
                    break

                block = fixed_text + body

                block_tokens = _count_tokens(
                    self._tokenizer,
                    block,
                )

            blocks.append(block)

            sources.append(
                ContextSource(
                    source_number=source_number,
                    document_id=(chunk.document_id),
                    chunk_id=(chunk.chunk_id),
                    title=chunk.title,
                    source_url=(chunk.source_url),
                    text=body,
                    dense_rank=(hit.dense_rank),
                    bm25_rank=(hit.bm25_rank),
                    retrieval_source=(hit.source),
                )
            )

            used_tokens += block_tokens

            if used_tokens >= self._max_tokens:
                break

        context_text = "".join(blocks).strip()

        actual_tokens = _count_tokens(
            self._tokenizer,
            context_text,
        )

        if actual_tokens > self._max_tokens:
            raise RuntimeError(
                "Context builder exceeded "
                "the configured token budget: "
                f"{actual_tokens} > "
                f"{self._max_tokens}"
            )

        return BuiltContext(
            text=context_text,
            sources=tuple(sources),
            token_count=actual_tokens,
        )
