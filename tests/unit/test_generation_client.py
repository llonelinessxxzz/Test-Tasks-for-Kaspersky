from __future__ import annotations

import json

import httpx
import pytest

from support_rag.core.config import Settings
from support_rag.core.schemas import ChatMessage
from support_rag.generation.client import VLLMClient
from support_rag.generation.prompt import RESPONSE_CONTRACT_REGEX


@pytest.mark.parametrize("response_regex", [None, RESPONSE_CONTRACT_REGEX])
async def test_decoding_constraint_is_explicit_and_not_applied_to_plain_calls(response_regex):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test",
                "choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = VLLMClient(Settings(_env_file=None), http_client=http)
        await client.generate(
            [ChatMessage(role="user", content="Question")], response_regex=response_regex
        )

    if response_regex is None:
        assert "structured_outputs" not in requests[0]
    else:
        assert requests[0]["structured_outputs"] == {"regex": response_regex}
