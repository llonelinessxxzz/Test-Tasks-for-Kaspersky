"""Exercise context, prompts, HTTP client and response contract without a live model."""

from __future__ import annotations

import json

import httpx
import pytest

from support_rag.core.config import Settings
from support_rag.core.schemas import DocumentChunk, GenerationResult
from support_rag.generation.client import VLLMClient
from support_rag.generation.context import ContextBuilder
from support_rag.generation.prompt import (
    RESPONSE_CONTRACT_REGEX,
    generation_failure_message,
    insufficient_message,
)
from support_rag.retrieval.hybrid import HybridResult
from support_rag.services.rag import RAGService


class ByteTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return bytes(token_ids).decode("utf-8", errors="ignore")


class StubRetriever:
    def search(self, query: str) -> list[HybridResult]:
        return [
            HybridResult(
                chunk=DocumentChunk(
                    chunk_id="doc-0",
                    document_id="doc",
                    title="Подписка",
                    source_url="https://example.com/subscription",
                    chunk_index=0,
                    token_count=8,
                    text="Откройте настройки подписки и отключите автопродление.",
                ),
                rank=1,
                dense_rank=1,
                bm25_rank=None,
                dense_score=0.9,
                bm25_score=None,
                source="dense",
            )
        ]


@pytest.mark.parametrize(
    ("outputs", "abstained", "attempts", "citations"),
    [
        ([("[STATUS:SUPPORTED]\nОткройте настройки подписки. [SOURCE 1]", "stop")], False, 1, [1]),
        ([("[STATUS:SUPPORTED]\nОткройте настройки подписки.", "stop")], False, 1, [1]),
        ([("[STATUS:INSUFFICIENT]\nНет сведений.", "stop")], True, 1, []),
        ([("[STATUS:INSUFFICIENT]\nНет сведений. [SOURCE 1]", "stop")], True, 1, []),
        (
            [
                ("[STATUS:SUPPORTED]\nОткройте настройки. [SOURCE 99]", "stop"),
                ("[STATUS:SUPPORTED]\nОткройте настройки подписки. [SOURCE 1]", "stop"),
            ],
            False,
            2,
            [1],
        ),
        (
            [
                ("[STATUS:SUPPORTED]\nОткройте настройки. [SOURCE 1]", "length"),
                ("[STATUS:INSUFFICIENT]\nНедостаточно информации.", "stop"),
            ],
            True,
            2,
            [],
        ),
        (
            [("Ответ без статусного маркера", "stop"), ("Ещё один ответ без маркера", "stop")],
            True,
            2,
            [],
        ),
        (
            [
                ("[sOURCE 1]\nОткройте настройки подписки.", "stop"),
                ("[STATUS:SUPPORTED]", "stop"),
            ],
            True,
            2,
            [],
        ),
        (
            [
                ("[SOURCE 1]\nОткройте настройки подписки.", "stop"),
                ("[STATUS:SUPPORTED]\nОткройте настройки подписки.", "stop"),
            ],
            False,
            2,
            [1],
        ),
        (
            [
                ("Here is an ungrounded answer with no status. [SOURCE 1]", "stop"),
                ("[STATUS:INSUFFICIENT]\nНет сведений.", "stop"),
            ],
            True,
            2,
            [],
        ),
        (
            [
                ("[STATUS:SUPPORTED]\nОткройте настройки. [SOURCE 99]", "stop"),
                ("[STATUS:SUPPORTED]\nОткройте настройки. [SOURCE 99]", "stop"),
            ],
            True,
            2,
            [],
        ),
    ],
)
async def test_rag_response_contract(outputs, abstained, attempts, citations) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        text, finish_reason = outputs[len(requests)]
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        service = RAGService(
            retriever=StubRetriever(),
            context_builder=ContextBuilder(ByteTokenizer(), max_tokens=1000),
            generator=VLLMClient(Settings(_env_file=None), http_client=http),
        )
        response = await service.ask("Как отключить автопродление?")

    assert response.abstained is abstained
    assert response.generation_attempts == attempts == len(requests)
    assert response.cited_source_numbers == citations
    assert [source.source_number for source in response.sources if source.cited] == citations
    assert "[STATUS:" not in response.answer
    if abstained:
        if outputs[-1][0].startswith("[STATUS:INSUFFICIENT]"):
            assert response.abstention_reason == "insufficient_context"
            assert response.answer == insufficient_message("Russian")
        else:
            assert response.abstention_reason == "generation_contract"
            assert response.answer == generation_failure_message("Russian")
    else:
        assert response.abstention_reason is None
    assert "structured_outputs" not in requests[0]
    if attempts == 2:
        assert requests[1]["structured_outputs"] == {"regex": RESPONSE_CONTRACT_REGEX}
    assert "[SOURCE 1]" in requests[0]["messages"][1]["content"]
    assert "Как отключить автопродление?" in requests[0]["messages"][1]["content"]


@pytest.mark.parametrize(
    ("text", "finish_reason"),
    [
        ("[STATUS:SUPPORTED]", "stop"),
        ("[STATUS:SUPPORTED]\n \n", "stop"),
        ("[STATUS:SUPPORTED]\n[SOURCE 1]", "stop"),
        ("[STATUS:SUPPORTED]\n[SOURCE invalid]", "stop"),
        ("[STATUS:SUPPORTED]\nOpen subscription settings. [SOURCE 1]", "length"),
        ("[STATUS:SUPPORTED]\nOpen subscription settings.", "content_filter"),
        ("[STATUS:SUPPORTED]\nOpen subscription settings. [SOURCE 1]", None),
        ("[STATUS:SUPPORTED]\nОткройте настройки. [SOURCE 1]", "stop"),
    ],
)
async def test_invalid_supported_output_cannot_be_rescued_with_citations(text, finish_reason):
    class Generator:
        calls = 0

        async def generate(self, messages, *, response_regex=None):
            self.calls += 1
            assert response_regex == (RESPONSE_CONTRACT_REGEX if self.calls > 1 else None)
            return GenerationResult(text=text, model="test", finish_reason=finish_reason)

    service = RAGService(
        retriever=StubRetriever(),
        context_builder=ContextBuilder(ByteTokenizer(), max_tokens=1000),
        generator=Generator(),
    )
    result = await service.ask("How can I disable automatic renewal?")
    assert result.abstained is True
    assert result.abstention_reason == "generation_contract"
    assert result.generation_attempts == 2
    assert result.answer == generation_failure_message("English")
    assert not result.cited_source_numbers
    assert not any(source.cited for source in result.sources)


async def test_blank_question_does_not_call_dependencies() -> None:
    class MustNotCall:
        def __getattr__(self, name):
            raise AssertionError(f"Unexpected dependency access: {name}")

    service = RAGService(
        retriever=MustNotCall(), context_builder=MustNotCall(), generator=MustNotCall()
    )
    with pytest.raises(ValueError, match="Question cannot be blank"):
        await service.ask("   ")
