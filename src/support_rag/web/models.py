from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from support_rag.core.schemas import RAGResponse
from support_rag.generation.context import BuiltContext, ContextSource


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value.encode("utf-8")) > 1600:
            raise ValueError("Question is blank or too long; use at most 1600 UTF-8 bytes")
        return value


class ContextPayload(BaseModel):
    text: str
    sources: list[ContextSource]
    token_count: int = Field(ge=0)
    retrieved_document_ids: list[str]
    context_document_ids: list[str]
    context_chunk_ids: list[str]
    prompt_tokens: int = Field(ge=0)

    @classmethod
    def from_context(cls, context: BuiltContext, documents: list[str], prompt_tokens: int):
        return cls(
            text=context.text,
            sources=[asdict(source) for source in context.sources],
            token_count=context.token_count,
            retrieved_document_ids=documents,
            context_document_ids=list(dict.fromkeys(s.document_id for s in context.sources)),
            context_chunk_ids=[s.chunk_id for s in context.sources],
            prompt_tokens=prompt_tokens,
        )

    def built_context(self) -> BuiltContext:
        return BuiltContext(self.text, tuple(self.sources), self.token_count)


class Confidence(BaseModel):
    level: Literal["supported", "review", "insufficient"]
    label: str
    reason: str


class Draft(BaseModel):
    response: RAGResponse
    confidence: Confidence
    is_draft: bool = True
    retrieval_query: str
    history_used: bool = False
    latency_seconds: float = Field(ge=0)
