from __future__ import annotations

import re

import pytest

from support_rag.generation.context import (
    BuiltContext,
    ContextSource,
)
from support_rag.generation.prompt import (
    RESPONSE_CONTRACT_REGEX,
    SYSTEM_PROMPT,
    build_generation_messages,
    detect_response_language,
    generation_failure_message,
    response_matches_language,
)


def make_context() -> BuiltContext:
    source = ContextSource(
        source_number=1,
        document_id="subscription-auto-renew",
        chunk_id="subscription-auto-renew-0",
        title="Отключение автопродления",
        source_url="https://example.com/article",
        text="Откройте настройки подписки.",
        dense_rank=1,
        bm25_rank=1,
        retrieval_source="dense",
    )

    return BuiltContext(
        text=(
            "[SOURCE 1]\nTitle: Отключение автопродления\nContent:\nОткройте настройки подписки."
        ),
        sources=(source,),
        token_count=12,
    )


def test_build_generation_messages_contains_context_and_question() -> None:
    messages = build_generation_messages(
        question="Как отключить автопродление?",
        context=make_context().text,
    )

    assert len(messages) == 2

    assert messages[0].role == "system"
    assert messages[1].role == "user"

    assert "[STATUS:SUPPORTED]" in SYSTEM_PROMPT
    assert "[STATUS:INSUFFICIENT]" in SYSTEM_PROMPT
    assert "KNOWLEDGE BASE CONTEXT" in SYSTEM_PROMPT

    assert "[SOURCE 1]" in messages[1].content

    assert "Как отключить автопродление?" in messages[1].content

    assert "RESPONSE LANGUAGE" in messages[1].content

    assert "Russian" in messages[1].content


def test_build_generation_messages_handles_empty_context() -> None:
    messages = build_generation_messages(
        question="Как решить проблему?",
        context="",
    )

    assert "No relevant support material was retrieved." in messages[1].content


def test_build_generation_messages_rejects_blank_question() -> None:
    with pytest.raises(
        ValueError,
        match="question cannot be empty",
    ):
        build_generation_messages(
            question="   ",
            context=make_context().text,
        )


def test_detect_response_language_for_russian() -> None:
    assert detect_response_language("Как восстановить пароль?") == "Russian"


def test_detect_response_language_for_english() -> None:
    assert detect_response_language("How do I remove Kaspersky?") == "English"


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("[STATUS:SUPPORTED]\nОткройте настройки. [SOURCE 1]", True),
        ("[STATUS:INSUFFICIENT]\nNot enough information.", True),
        ("[STATUS:SUPPORTED]\nFirst step.\nSecond step.", True),
        ("[STATUS:SUPPORTED]", False),
        ("[STATUS:SUPPORTED]\n", False),
        ("[sOURCE 1] [SOURCE 2]\nЧтобы восстановить пароль...", False),
        ("[STATUS:UNKNOWN]\nAnswer.", False),
        ("Preface\n[STATUS:SUPPORTED]\nAnswer.", False),
    ],
)
def test_response_constraint_requires_status_and_body_without_forcing_support(text, accepted):
    assert bool(re.fullmatch(RESPONSE_CONTRACT_REGEX, text)) is accepted


@pytest.mark.parametrize("language", ["Russian", "English"])
def test_generation_failure_message_is_localized(language):
    assert response_matches_language(generation_failure_message(language), language)
