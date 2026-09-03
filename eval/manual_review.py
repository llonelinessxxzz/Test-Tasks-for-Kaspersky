from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ReviewItem(BaseModel):
    id: str
    question: str
    language: str
    expected_abstain: bool
    expected_document_ids: list[str]
    retrieved_document_ids: list[str]
    cited_document_ids: list[str]
    answer: str
    error: str | None = None
    correctness: int | None = Field(default=None, ge=0, le=2)
    groundedness: int | None = Field(default=None, ge=0, le=2)
    completeness: int | None = Field(default=None, ge=0, le=2)
    citation_support: int | None = Field(default=None, ge=0, le=2)
    notes: str = ""


RUBRIC = """# Manual Generation Review Rubric

Each answer receives four scores from 0 to 2.

## Correctness

**2 — Correct**

The answer contains no material factual or procedural errors relative to the
expected Knowledge Base source.

**1 — Partially correct**

The core answer is correct, but there are minor inaccuracies, ambiguous
statements, unnecessary claims, or small procedural issues.

**0 — Incorrect**

The answer contains a material factual error, misleading instruction, or
answers the wrong question.


## Groundedness

**2 — Fully grounded**

All material factual claims and troubleshooting steps are supported by the
retrieved Knowledge Base content.

**1 — Partially grounded**

The main answer is grounded, but one or more secondary claims appear
unsupported, overgeneralized, or cannot be clearly traced to the source.

**0 — Ungrounded**

The answer substantially relies on unsupported claims or invented
instructions.


## Completeness

**2 — Sufficient**

The answer contains the information required to solve the user's question
without unnecessary omissions.

**1 — Partially complete**

The answer is useful, but misses one or more relevant steps, conditions,
warnings, or alternatives contained in the source.

**0 — Incomplete**

The answer omits essential information required to address the question.


## Citation support

**2 — Correctly supported**

The cited source or sources directly support the claims they are attached to.

**1 — Partially supported**

The main citation is relevant, but some cited claims are weakly supported,
the citation is too broad, or an unnecessary source is also cited.

**0 — Unsupported**

The citation does not support the answer or the model cites an irrelevant
document as evidence.


## Abstention cases

For out-of-KB questions:

- Correctness = 2 if the assistant correctly refuses to answer.
- Groundedness = 2 if it introduces no unsupported domain answer.
- Completeness = 2 if the refusal clearly explains that the available
  Kaspersky Knowledge Base is insufficient.
- Citation support = 2 if no irrelevant source is cited.

Do not judge style, wording, or verbosity unless it affects factual quality.
"""


def write_review(cases: list[dict[str, Any]], output_dir: Path) -> None:
    """Write blank review forms; never invent scores or overwrite a human review."""
    with (output_dir / "manual_review.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            fields = {
                key: case[key]
                for key in (
                    "id",
                    "question",
                    "language",
                    "expected_abstain",
                    "expected_document_ids",
                    "retrieved_document_ids",
                    "cited_document_ids",
                    "answer",
                    "error",
                )
                if key in case
            }
            stream.write(ReviewItem(**fields).model_dump_json() + "\n")
    with (output_dir / "manual_review_rubric.md").open("x", encoding="utf-8") as stream:
        stream.write(RUBRIC)
