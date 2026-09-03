from pathlib import Path

import pytest
from pydantic import ValidationError

from support_rag.core.config import Settings


def test_example_matches_defaults_without_duplicate_settings():
    path = Path(__file__).resolve().parents[2] / ".env.example"
    keys = [
        line.split("=", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(keys) == len(set(keys))
    assert {key.lower() for key in keys} == set(Settings.model_fields)
    assert Settings(_env_file=path).model_dump() == Settings(_env_file=None).model_dump()


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_overlap_tokens": 192},
        {"chunk_size_tokens": 512},
        {"retrieval_top_k": 17},
        {"hybrid_bm25_max_rank": 9},
        {"llm_max_tokens": 800},
    ],
)
def test_invalid_runtime_budgets_are_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)
