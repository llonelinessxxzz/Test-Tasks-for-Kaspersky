from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import structlog

from support_rag.core.schemas import (
    AbstentionReason,
    ChatMessage,
    GenerationResult,
    RAGResponse,
    RAGSource,
)
from support_rag.generation.context import (
    BuiltContext,
    ContextBuilder,
    ContextSource,
)
from support_rag.generation.prompt import (
    RESPONSE_CONTRACT_REGEX,
    build_generation_messages,
    build_repair_messages,
    detect_response_language,
    generation_failure_message,
    insufficient_message,
    response_matches_language,
)

if TYPE_CHECKING:
    from support_rag.retrieval.hybrid import HybridResult

logger = structlog.get_logger(__name__)


ResponseStatus = Literal[
    "supported",
    "insufficient",
]


_CITATION_GROUP_PATTERN = re.compile(
    r"\[\s*SOURCE\s+([^\]]+)\]",
    flags=re.IGNORECASE,
)

_CITATION_NUMBER_PATTERN = re.compile(r"\d+")

_STATUS_LINE_PATTERN = re.compile(
    r"^\s*\[STATUS:"
    r"(SUPPORTED|INSUFFICIENT)"
    r"\]\s*$",
    flags=re.IGNORECASE,
)

_STATUS_TAG_PATTERN = re.compile(
    r"\[\s*STATUS:[^\]\r\n]+\]",
    flags=re.IGNORECASE,
)


class Retriever(Protocol):
    def search(
        self,
        query: str,
    ) -> list[HybridResult]: ...


class Generator(Protocol):
    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_regex: str | None = None,
    ) -> GenerationResult: ...


@dataclass(frozen=True)
class _ParsedGeneration:
    status: ResponseStatus | None
    answer: str

    cited_source_numbers: list[int]
    valid_citation_numbers: list[int]
    invalid_citation_numbers: list[int]


@dataclass(frozen=True)
class _ContractValidation:
    valid: bool
    status: ResponseStatus | None

    valid_citation_numbers: list[int]
    invalid_citation_numbers: list[int]


def _parse_status(
    text: str,
) -> ResponseStatus | None:
    """
    Parse the response status from the first non-empty line only.
    """
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        match = _STATUS_LINE_PATTERN.fullmatch(stripped)

        if match is None:
            return None

        status = match.group(1).lower()

        if status == "supported":
            return "supported"

        if status == "insufficient":
            return "insufficient"

        return None

    return None


def _strip_status_markers(
    text: str,
) -> str:
    """
    Remove internal STATUS markers from the user-visible answer.

    The valid first-line marker is removed. Any leaked or malformed
    STATUS marker later in the response is also sanitized.
    """
    lines = text.splitlines()

    cleaned_lines: list[str] = []

    first_non_empty_seen = False

    for line in lines:
        stripped = line.strip()

        if not first_non_empty_seen:
            if not stripped:
                continue

            first_non_empty_seen = True

            if _STATUS_LINE_PATTERN.fullmatch(stripped) is not None:
                continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    cleaned = _STATUS_TAG_PATTERN.sub(
        "",
        cleaned,
    )

    return cleaned.strip()


def _extract_citation_numbers(
    text: str,
) -> list[int]:
    """
    Extract citation numbers from the model output.

    Preferred format:
        [SOURCE 1] [SOURCE 2]

    Tolerated formats:
        [SOURCE 1; SOURCE 2]
        [SOURCE 1, SOURCE 2]
    """
    numbers: list[int] = []

    for match in _CITATION_GROUP_PATTERN.finditer(text):
        payload = match.group(1)

        raw_numbers = _CITATION_NUMBER_PATTERN.findall(payload)

        for raw_number in raw_numbers:
            number = int(raw_number)

            if number in numbers:
                continue

            numbers.append(number)

    return numbers


def _partition_citations(
    citation_numbers: Sequence[int],
    *,
    source_count: int,
) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []

    for number in citation_numbers:
        if 1 <= number <= source_count:
            valid.append(number)
        else:
            invalid.append(number)

    return valid, invalid


def _parse_generation(
    generation: GenerationResult,
    *,
    source_count: int,
) -> _ParsedGeneration:
    status = _parse_status(generation.text)

    answer = _strip_status_markers(generation.text)

    cited_source_numbers = _extract_citation_numbers(answer)

    (
        valid_citation_numbers,
        invalid_citation_numbers,
    ) = _partition_citations(
        cited_source_numbers,
        source_count=source_count,
    )

    return _ParsedGeneration(
        status=status,
        answer=answer,
        cited_source_numbers=(cited_source_numbers),
        valid_citation_numbers=(valid_citation_numbers),
        invalid_citation_numbers=(invalid_citation_numbers),
    )


def _validate_contract(
    parsed: _ParsedGeneration,
    generation: GenerationResult,
    *,
    expected_language: str,
) -> _ContractValidation:
    """
    Validate deterministic response properties.

    Semantic factual correctness is evaluated separately.
    """
    valid = True

    if parsed.status is None:
        valid = False

    if generation.finish_reason != "stop":
        valid = False

    if parsed.invalid_citation_numbers:
        valid = False

    if parsed.status == "supported":
        body = _CITATION_GROUP_PATTERN.sub("", parsed.answer).strip()
        if not body:
            valid = False

        if not parsed.valid_citation_numbers:
            valid = False

        if not response_matches_language(
            body,
            expected_language,
        ):
            valid = False

    elif parsed.status == "insufficient":
        if parsed.cited_source_numbers:
            valid = False

    return _ContractValidation(
        valid=valid,
        status=parsed.status,
        valid_citation_numbers=(parsed.valid_citation_numbers),
        invalid_citation_numbers=(parsed.invalid_citation_numbers),
    )


def _can_attach_retrieval_citations(
    parsed: _ParsedGeneration,
    generation: GenerationResult,
    *,
    expected_language: str,
) -> bool:
    """
    Accept a citation-free supported response when everything except
    citation syntax is valid.

    Verified citations are then attached deterministically from retrieval
    metadata rather than asking the model to rewrite an otherwise good
    answer.
    """
    if parsed.status != "supported":
        return False

    body = _CITATION_GROUP_PATTERN.sub("", parsed.answer).strip()
    if not body:
        return False

    if generation.finish_reason != "stop":
        return False

    if parsed.invalid_citation_numbers:
        return False

    if parsed.valid_citation_numbers:
        return False

    return response_matches_language(
        body,
        expected_language,
    )


def _fallback_citation_numbers(
    context: BuiltContext,
) -> list[int]:
    """
    Return verified source numbers belonging to the strongest retrieved
    document.

    Source identity comes from retrieval metadata, not model-generated
    text.
    """
    if not context.sources:
        return []

    top_document_id = context.sources[0].document_id

    return [
        source.source_number
        for source in context.sources
        if (source.document_id == top_document_id)
    ]


def _build_rag_sources(
    context_sources: Sequence[ContextSource],
    *,
    cited_source_numbers: Sequence[int],
) -> list[RAGSource]:
    """
    Convert ContextSource metadata into API-facing sources.
    """
    cited_numbers = set(cited_source_numbers)

    sources: list[RAGSource] = []

    for source in context_sources:
        sources.append(
            RAGSource(
                source_number=(source.source_number),
                document_id=(source.document_id),
                chunk_id=(source.chunk_id),
                title=(source.title),
                source_url=(source.source_url),
                dense_rank=(source.dense_rank),
                bm25_rank=(source.bm25_rank),
                retrieval_source=(source.retrieval_source),
                cited=(source.source_number in cited_numbers),
            )
        )

    return sources


def _build_response(
    *,
    answer: str,
    context: BuiltContext,
    generation: GenerationResult,
    cited_source_numbers: Sequence[int],
    invalid_citation_numbers: Sequence[int],
    abstained: bool,
    generation_attempts: int,
    abstention_reason: AbstentionReason | None = None,
) -> RAGResponse:
    cited_numbers = list(cited_source_numbers)

    invalid_numbers = list(invalid_citation_numbers)

    return RAGResponse(
        answer=answer,
        sources=_build_rag_sources(
            context.sources,
            cited_source_numbers=(cited_numbers),
        ),
        cited_source_numbers=(cited_numbers),
        invalid_citation_numbers=(invalid_numbers),
        model=generation.model,
        context_tokens=(context.token_count),
        prompt_tokens=(generation.prompt_tokens),
        completion_tokens=(generation.completion_tokens),
        total_tokens=(generation.total_tokens),
        finish_reason=(generation.finish_reason),
        abstained=abstained,
        abstention_reason=abstention_reason,
        generation_attempts=(generation_attempts),
    )


def _log_completed_request(
    *,
    response: RAGResponse,
    retrieved_hits: Sequence[HybridResult],
) -> None:
    logger.info(
        "rag_request_completed",
        abstained=response.abstained,
        abstention_reason=response.abstention_reason,
        cited_sources=(response.cited_source_numbers),
        completion_tokens=(response.completion_tokens),
        context_sources=len(response.sources),
        context_tokens=(response.context_tokens),
        finish_reason=(response.finish_reason),
        generation_attempts=(response.generation_attempts),
        invalid_citations=(response.invalid_citation_numbers),
        prompt_tokens=(response.prompt_tokens),
        retrieved_hits=len(retrieved_hits),
    )


class RAGService:
    """
    End-to-end retrieval-augmented generation service.

    Flow:

        question
            ↓
        hybrid retrieval
            ↓
        bounded KB context
            ↓
        generation
            ↓
        deterministic contract validation
            ↓
        generated citations OR verified retrieval citations
            ↓
        one repair attempt when necessary
            ↓
        safe abstention on unrecoverable failure
    """

    def __init__(
        self,
        *,
        retriever: Retriever | None,
        context_builder: ContextBuilder | None,
        generator: Generator,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._generator = generator

    async def ask(
        self,
        question: str,
    ) -> RAGResponse:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be blank")

        if self._retriever is None or self._context_builder is None:
            raise RuntimeError("Local retrieval is not configured")
        retrieved_hits = self._retriever.search(question)
        context = self._context_builder.build(retrieved_hits)
        return await self.answer_context(question, context, retrieved_hits=retrieved_hits)

    async def answer_context(
        self,
        question: str,
        context: BuiltContext,
        *,
        retrieved_hits: Sequence[HybridResult] = (),
    ) -> RAGResponse:
        """Apply the same generation/repair contract to locally or remotely built context."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be blank")
        response_language = detect_response_language(question)

        generation_attempts = 0

        last_generation: GenerationResult | None = None

        last_parsed: _ParsedGeneration | None = None

        for attempt in range(1, 3):
            generation_attempts = attempt

            if attempt == 1:
                messages = build_generation_messages(
                    question=question,
                    context=context.text,
                    response_language=(response_language),
                )
            else:
                messages = build_repair_messages(
                    question=question,
                    context=context.text,
                    response_language=(response_language),
                )

            generation = await self._generator.generate(
                messages, response_regex=RESPONSE_CONTRACT_REGEX if attempt > 1 else None
            )

            last_generation = generation

            parsed = _parse_generation(
                generation,
                source_count=len(context.sources),
            )

            last_parsed = parsed

            if parsed.status == "insufficient":
                response = _build_response(
                    answer=(insufficient_message(response_language)),
                    context=context,
                    generation=generation,
                    cited_source_numbers=[],
                    invalid_citation_numbers=[],
                    abstained=True,
                    abstention_reason="insufficient_context",
                    generation_attempts=(generation_attempts),
                )

                _log_completed_request(
                    response=response,
                    retrieved_hits=(retrieved_hits),
                )

                return response

            validation = _validate_contract(
                parsed,
                generation,
                expected_language=(response_language),
            )

            if validation.valid:
                response = _build_response(
                    answer=(parsed.answer),
                    context=context,
                    generation=generation,
                    cited_source_numbers=(validation.valid_citation_numbers),
                    invalid_citation_numbers=(validation.invalid_citation_numbers),
                    abstained=False,
                    generation_attempts=(generation_attempts),
                )

                _log_completed_request(
                    response=response,
                    retrieved_hits=(retrieved_hits),
                )

                return response

            if _can_attach_retrieval_citations(
                parsed,
                generation,
                expected_language=(response_language),
            ):
                fallback_citations = _fallback_citation_numbers(context)

                if fallback_citations:
                    logger.info(
                        "citations_attached_from_retrieval",
                        source_numbers=(fallback_citations),
                    )

                    response = _build_response(
                        answer=(parsed.answer),
                        context=context,
                        generation=generation,
                        cited_source_numbers=(fallback_citations),
                        invalid_citation_numbers=[],
                        abstained=False,
                        generation_attempts=(generation_attempts),
                    )

                    _log_completed_request(
                        response=response,
                        retrieved_hits=(retrieved_hits),
                    )

                    return response

            logger.warning(
                "generation_contract_violation",
                attempt=attempt,
                expected_language=(response_language),
                finish_reason=(generation.finish_reason),
                invalid_citations=(validation.invalid_citation_numbers),
                status=(parsed.status),
                valid_citations=(validation.valid_citation_numbers),
            )

        if last_generation is None or last_parsed is None:
            raise RuntimeError("Generation loop completed without a model response")

        logger.error(
            "generation_contract_fallback",
            attempts=(generation_attempts),
            expected_language=(response_language),
        )

        response = _build_response(
            answer=generation_failure_message(response_language),
            context=context,
            generation=last_generation,
            cited_source_numbers=[],
            invalid_citation_numbers=(last_parsed.invalid_citation_numbers),
            abstained=True,
            abstention_reason="generation_contract",
            generation_attempts=(generation_attempts),
        )

        _log_completed_request(
            response=response,
            retrieved_hits=(retrieved_hits),
        )

        return response
