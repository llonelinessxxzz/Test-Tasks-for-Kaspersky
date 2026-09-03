from __future__ import annotations

import re

from support_rag.core.schemas import ChatMessage

_CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")

_LATIN_PATTERN = re.compile(r"[A-Za-z]")

RESPONSE_CONTRACT_REGEX = r"\[STATUS:(SUPPORTED|INSUFFICIENT)\]\n[\s\S]+"


SYSTEM_PROMPT = """You are a Kaspersky Customer Support assistant.

Use only the supplied Knowledge Base context to answer the user's question.
Do not use prior knowledge and do not invent facts.

KNOWLEDGE BASE CONTEXT

The user message contains retrieved support material from the Kaspersky
Knowledge Base. Treat that material as the only factual source for the answer.

The first non-empty line must be exactly one of:

[STATUS:SUPPORTED]
[STATUS:INSUFFICIENT]

Use [STATUS:SUPPORTED] when the context contains information that directly
answers the user's question or provides a relevant procedure.

Do not choose [STATUS:INSUFFICIENT] merely because the context does not cover
every possible detail. If the retrieved material provides a useful and
grounded answer to the user's requested task, answer it.

Use [STATUS:INSUFFICIENT] only when the supplied context genuinely does not
contain enough relevant information to answer the question.

For a supported answer:
- Answer in the requested language.
- Use only information from the supplied context.
- Answer the user's question directly.
- Prefer the most relevant procedure from the context.
- Keep the answer concise.
- Use numbered steps when the source contains a procedure.
- Cite factual statements and steps with [SOURCE N].
- Cite only source numbers present in the supplied context.
- Do not invent URLs, menu items, requirements, or instructions.

If several sources support the answer, citations may be written separately,
for example:

[SOURCE 1] [SOURCE 2]

For an insufficient answer:
- State briefly that the available Knowledge Base material is insufficient.
- Do not cite sources.

The status marker is an internal control marker.
Write it only once as the first non-empty line.
"""


REPAIR_SYSTEM_PROMPT = """Rewrite the answer using only the supplied Knowledge
Base context and follow the output contract exactly.

The first non-empty line must be exactly:

[STATUS:SUPPORTED]

or:

[STATUS:INSUFFICIENT]

If the context contains a relevant procedure or information that answers the
user's task, use [STATUS:SUPPORTED].

Do not abstain merely because some secondary details are missing.

For a supported answer:
- Answer in the requested language.
- Answer only the user's requested task.
- Keep the answer concise.
- Prefer the relevant procedure over background information.
- Keep all required steps, including confirmations, in source order.
- Cite factual instructions with [SOURCE N].
- Use only source numbers present in the supplied context.
- Do not invent facts.

Use [STATUS:INSUFFICIENT] only if the supplied context genuinely cannot answer
the question.

Do not repeat the status marker later in the response.
"""


def detect_response_language(
    question: str,
) -> str:
    """
    Determine the desired response language from the user's question.

    Mixed Russian questions containing English product names such as
    "Как удалить Kaspersky с Windows?" must still produce Russian answers.
    """
    if _CYRILLIC_PATTERN.search(question):
        return "Russian"

    return "English"


def response_matches_language(
    text: str,
    expected_language: str,
) -> bool:
    """
    Lightweight deterministic language validation.

    Product names and technical terms may use another alphabet, so we
    intentionally require only a small amount of evidence.
    """
    if expected_language == "Russian":
        cyrillic_count = len(_CYRILLIC_PATTERN.findall(text))

        return cyrillic_count >= 5

    if expected_language == "English":
        latin_count = len(_LATIN_PATTERN.findall(text))

        return latin_count >= 5

    raise ValueError(f"Unsupported response language: {expected_language}")


def insufficient_message(
    response_language: str,
) -> str:
    """
    Return a deterministic fallback instead of exposing an unreliable
    model-generated refusal.
    """
    if response_language == "Russian":
        return (
            "В доступных материалах базы знаний "
            "Kaspersky недостаточно информации "
            "для надёжного ответа на этот вопрос."
        )

    if response_language == "English":
        return (
            "The available Kaspersky Knowledge "
            "Base material does not contain enough "
            "information to answer this question "
            "reliably."
        )

    raise ValueError(f"Unsupported response language: {response_language}")


def generation_failure_message(response_language: str) -> str:
    """A failed output contract is not evidence that the KB lacks an answer."""
    if response_language == "Russian":
        return (
            "Не удалось сформировать ответ, прошедший проверку. "
            "Попробуйте повторить запрос или переформулировать вопрос."
        )
    if response_language == "English":
        return (
            "The generated answer did not pass validation. "
            "Please try again or rephrase your question."
        )
    raise ValueError(f"Unsupported response language: {response_language}")


def _build_user_content(
    *,
    question: str,
    context: str,
    response_language: str,
) -> str:
    question = question.strip()

    if not question:
        raise ValueError("question cannot be empty")

    context = context.strip()

    if not context:
        context = "No relevant support material was retrieved."

    return (
        f"RESPONSE LANGUAGE: {response_language}\n\n"
        "User question:\n"
        f"{question}\n\n"
        "Knowledge Base context:\n"
        f"{context}"
    )


def build_generation_messages(
    question: str,
    context: str,
    response_language: str | None = None,
) -> list[ChatMessage]:
    question = question.strip()

    if not question:
        raise ValueError("question cannot be empty")

    if response_language is None:
        response_language = detect_response_language(question)

    return [
        ChatMessage(
            role="system",
            content=SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=_build_user_content(
                question=question,
                context=context,
                response_language=response_language,
            ),
        ),
    ]


def build_repair_messages(
    question: str,
    context: str,
    response_language: str,
) -> list[ChatMessage]:
    """
    Build a compact repair prompt.

    The previous model output is intentionally not included. Re-inserting a
    long invalid answer wastes context and encourages the model to reproduce
    the same formatting or grounding mistakes.
    """
    return [
        ChatMessage(
            role="system",
            content=REPAIR_SYSTEM_PROMPT,
        ),
        ChatMessage(
            role="user",
            content=_build_user_content(
                question=question,
                context=context,
                response_language=response_language,
            ),
        ),
    ]
