from __future__ import annotations

from support_rag.core.schemas import DocumentChunk
from support_rag.generation.context import ContextBuilder
from support_rag.retrieval.hybrid import HybridResult


class FakeTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        del add_special_tokens

        return list(range(len(text.split())))

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens

        return " ".join(f"token-{token_id}" for token_id in token_ids)


def make_result(
    document_id: str,
    *,
    text: str,
    rank: int,
) -> HybridResult:
    chunk = DocumentChunk(
        chunk_id=f"{document_id}-chunk",
        document_id=document_id,
        title=f"Article {document_id}",
        source_url=(f"https://example.com/{document_id}"),
        text=text,
        chunk_index=0,
        token_count=max(
            len(text.split()),
            1,
        ),
    )

    return HybridResult(
        chunk=chunk,
        rank=rank,
        dense_rank=rank,
        bm25_rank=None,
        dense_score=0.9,
        bm25_score=None,
        source="dense",
    )


def test_context_builder_includes_sources() -> None:
    builder = ContextBuilder(
        FakeTokenizer(),
        max_tokens=100,
    )

    result = builder.build(
        [
            make_result(
                "doc-a",
                text="first article body",
                rank=1,
            ),
            make_result(
                "doc-b",
                text="second article body",
                rank=2,
            ),
        ]
    )

    assert "[SOURCE 1]" in result.text
    assert "[SOURCE 2]" in result.text

    assert len(result.sources) == 2

    assert result.sources[0].document_id == "doc-a"

    assert result.sources[1].document_id == "doc-b"


def test_context_builder_respects_token_budget() -> None:
    builder = ContextBuilder(
        FakeTokenizer(),
        max_tokens=20,
    )

    result = builder.build(
        [
            make_result(
                "doc-a",
                text=" ".join(["word"] * 100),
                rank=1,
            )
        ]
    )

    assert result.token_count <= 20


def test_context_builder_handles_empty_results() -> None:
    builder = ContextBuilder(
        FakeTokenizer(),
        max_tokens=100,
    )

    result = builder.build([])

    assert result.text == ""
    assert result.sources == ()
    assert result.token_count == 0
