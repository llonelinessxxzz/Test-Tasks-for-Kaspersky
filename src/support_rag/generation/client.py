from __future__ import annotations

from collections.abc import Sequence

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from support_rag.core.config import Settings
from support_rag.core.schemas import ChatMessage, GenerationResult

log = structlog.get_logger(__name__)


class GenerationBackendError(RuntimeError):
    pass


class VLLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._owns_http_client = http_client is None

        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=settings.llm_connect_timeout_seconds,
                read=settings.llm_request_timeout_seconds,
                write=30.0,
                pool=10.0,
            )
        )

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    async def list_models(self) -> list[str]:
        try:
            response = await self._http.get(
                self._settings.llm_models_url,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationBackendError(f"Unable to reach vLLM model endpoint: {exc}") from exc

        payload = response.json()

        return [
            item["id"]
            for item in payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]

    async def is_ready(self) -> bool:
        try:
            models = await self.list_models()
        except GenerationBackendError:
            return False

        return self._settings.llm_model in models

    @retry(
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(
            initial=0.5,
            max=4.0,
        ),
        reraise=True,
    )
    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_regex: str | None = None,
    ) -> GenerationResult:
        payload = {
            "model": self._settings.llm_model,
            "messages": [message.model_dump() for message in messages],
            "max_tokens": max_tokens or self._settings.llm_max_tokens,
            "temperature": (self._settings.llm_temperature if temperature is None else temperature),
        }

        if response_regex is not None:
            payload["structured_outputs"] = {"regex": response_regex}

        try:
            response = await self._http.post(
                self._settings.llm_chat_completions_url,
                json=payload,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            response_body = exc.response.text[:1000]

            log.error(
                "vllm_request_rejected",
                status_code=exc.response.status_code,
                response_body=response_body,
            )

            raise GenerationBackendError(f"vLLM returned HTTP {exc.response.status_code}") from exc

        except httpx.HTTPError:
            log.exception("vllm_request_failed")
            raise

        data = response.json()

        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationBackendError("vLLM returned an unexpected response schema") from exc

        usage = data.get("usage") or {}

        result = GenerationResult(
            text=content,
            model=data.get("model", self._settings.llm_model),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

        log.info(
            "generation_completed",
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
        )

        return result
