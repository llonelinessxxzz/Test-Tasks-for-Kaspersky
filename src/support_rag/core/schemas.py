from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AbstentionReason = Literal["insufficient_context", "generation_contract"]


class SourceDocument(BaseModel):
    document_id: str
    title: str
    source_url: str
    text: str = Field(min_length=1)

    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str

    title: str
    source_url: str

    text: str = Field(min_length=1)

    chunk_index: int = Field(ge=0)

    token_count: int = Field(ge=1)

    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedChunk(BaseModel):
    chunk: DocumentChunk

    retriever: Literal[
        "bm25",
        "dense",
    ]

    rank: int = Field(ge=1)

    score: float


class ChatMessage(BaseModel):
    role: Literal[
        "system",
        "user",
        "assistant",
    ]

    content: str


class GenerationResult(BaseModel):
    text: str
    model: str

    finish_reason: str | None = None

    prompt_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    completion_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    total_tokens: int | None = Field(
        default=None,
        ge=0,
    )


class RAGSource(BaseModel):
    source_number: int = Field(ge=1)

    document_id: str
    chunk_id: str

    title: str
    source_url: str

    dense_rank: int | None = Field(
        default=None,
        ge=1,
    )

    bm25_rank: int | None = Field(
        default=None,
        ge=1,
    )

    retrieval_source: Literal[
        "dense",
        "bm25_promoted",
        "bm25_backfill",
    ]

    cited: bool = False


class RAGResponse(BaseModel):
    answer: str

    sources: list[RAGSource] = Field(default_factory=list)

    cited_source_numbers: list[int] = Field(default_factory=list)

    invalid_citation_numbers: list[int] = Field(default_factory=list)

    model: str

    context_tokens: int = Field(ge=0)

    prompt_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    completion_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    total_tokens: int | None = Field(
        default=None,
        ge=0,
    )

    finish_reason: str | None = None

    abstained: bool = False
    abstention_reason: AbstentionReason | None = None

    generation_attempts: int = Field(
        default=1,
        ge=0,
    )
