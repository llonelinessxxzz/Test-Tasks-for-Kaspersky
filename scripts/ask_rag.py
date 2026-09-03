from __future__ import annotations

import argparse
import asyncio

from support_rag.core.config import get_settings
from support_rag.core.logging import configure_logging
from support_rag.core.schemas import RAGResponse
from support_rag.runtime import build_runtime


def render_answer(response: RAGResponse) -> str:
    lines = [response.answer]
    cited = [source for source in response.sources if source.cited]
    if cited:
        lines.extend(["", "Sources:"])
        for source in cited:
            lines.append(f"[SOURCE {source.source_number}] {source.title}")
            lines.append(source.source_url)
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local Kaspersky support assistant.")
    parser.add_argument("question", help="Customer support question in Russian or English")
    parser.add_argument("--debug", action="store_true", help="Print the full response as JSON")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question cannot be empty")

    settings = get_settings()
    configure_logging(settings.log_level)
    runtime = build_runtime(settings)
    try:
        if not await runtime.llm_client.is_ready():
            raise RuntimeError("vLLM is not ready or the configured model is unavailable")
        response = await runtime.service.ask(args.question)
        print(response.model_dump_json(indent=2) if args.debug else render_answer(response))
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
