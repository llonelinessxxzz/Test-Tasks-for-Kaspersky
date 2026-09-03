from __future__ import annotations

import asyncio

from support_rag.core.config import get_settings
from support_rag.generation.client import VLLMClient


async def main() -> None:
    settings = get_settings()
    client = VLLMClient(settings)
    try:
        models = await client.list_models()
        if settings.llm_model not in models:
            raise RuntimeError(f"Configured model {settings.llm_model!r} is not served by vLLM")
        print(f"Ready: {settings.llm_model}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
