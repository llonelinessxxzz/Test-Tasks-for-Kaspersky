from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GenerationEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: Literal["ru", "en"]
    expected_abstain: bool
    expected_document_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sources(self) -> GenerationEvalCase:
        if self.expected_abstain == bool(self.expected_document_ids):
            raise ValueError("Only supported cases must specify expected document IDs")
        return self


class RetrievalEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(min_length=1)
    language: Literal["ru", "en", "mixed"]
    query_type: Literal["lexical", "semantic", "noisy", "multi_intent"]
    difficulty: Literal["easy", "medium", "hard"]


Case = TypeVar("Case", GenerationEvalCase, RetrievalEvalCase)


def load_cases(path: Path, schema: type[Case]) -> list[Case]:
    cases = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                case = schema.model_validate(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid evaluation case at {path}:{line_number}") from exc
            if case.id in seen:
                raise ValueError(f"Duplicate evaluation ID: {case.id}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"Empty evaluation dataset: {path}")
    return cases
