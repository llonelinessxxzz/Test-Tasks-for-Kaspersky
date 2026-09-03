from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from support_rag.core.config import get_settings
from support_rag.core.logging import configure_logging
from support_rag.core.schemas import SourceDocument
from support_rag.ingestion.loader import write_jsonl

log = structlog.get_logger(__name__)


CONTENT_ROOT_SELECTORS = (
    "[itemprop='articleBody']",
    ".article-content",
    ".content-article",
    ".help-content",
    "article",
    "main",
)

REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "svg",
    ".breadcrumbs",
    ".breadcrumb",
    ".sidebar",
    ".navigation",
    ".toc",
    ".table-of-contents",
    ".feedback",
    ".article-feedback",
    ".related-articles",
    "[aria-hidden='true']",
)

BLOCK_TAGS = (
    "h1",
    "h2",
    "h3",
    "h4",
    "p",
    "li",
    "pre",
)

STOP_PHRASES = (
    "вам помогла эта статья",
    "вам помогла эта информация",
    "была ли эта статья полезной",
    "did you find this article helpful",
    "was this information helpful",
    "what can we do better",
    "что нам нужно улучшить",
)


class SourceSpec(BaseModel):
    document_id: str = Field(min_length=1)

    url: HttpUrl

    product: str = Field(min_length=1)

    topic: str = Field(min_length=1)

    language: Literal[
        "ru",
        "en",
    ]


class KBFetchError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Fetch curated public Kaspersky support articles.")
    )

    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/raw/kaspersky_sources.jsonl"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/kb.jsonl"),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=("Write successfully fetched documents even if some sources fail."),
    )

    return parser.parse_args()


def load_sources(
    path: Path,
) -> list[SourceSpec]:
    if not path.exists():
        raise FileNotFoundError(path)

    sources: list[SourceSpec] = []
    seen_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                payload: Any = json.loads(line)

                source = SourceSpec.model_validate(payload)

            except (
                json.JSONDecodeError,
                ValidationError,
            ) as exc:
                raise ValueError(
                    f"Invalid source definition at {path}:{line_number}: {exc}"
                ) from exc

            if source.document_id in seen_ids:
                raise ValueError(f"Duplicate document_id {source.document_id!r}")

            seen_ids.add(source.document_id)

            sources.append(source)

    if not sources:
        raise ValueError(f"No sources found in {path}")

    return sources


def normalize_text(
    text: str,
) -> str:
    text = text.replace(
        "\xa0",
        " ",
    )

    text = text.replace(
        "\u200b",
        "",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def extract_title(
    soup: BeautifulSoup,
) -> str:
    h1 = soup.find("h1")

    if isinstance(h1, Tag):
        title = normalize_text(
            h1.get_text(
                " ",
                strip=True,
            )
        )

        if title:
            return title

    if soup.title is not None:
        title = normalize_text(
            soup.title.get_text(
                " ",
                strip=True,
            )
        )

        if title:
            return title

    raise KBFetchError("Could not extract article title")


def find_content_root(
    soup: BeautifulSoup,
) -> Tag:
    for selector in CONTENT_ROOT_SELECTORS:
        candidate = soup.select_one(selector)

        if isinstance(candidate, Tag):
            return candidate

    if isinstance(soup.body, Tag):
        return soup.body

    raise KBFetchError("Could not find article content")


def clean_root(
    root: Tag,
) -> None:
    for selector in REMOVE_SELECTORS:
        for element in root.select(selector):
            element.decompose()


def extract_article_text(
    soup: BeautifulSoup,
) -> str:
    root = find_content_root(soup)

    clean_root(root)

    blocks: list[str] = []

    h1 = root.find("h1")

    started = not isinstance(h1, Tag)

    for element in root.find_all(BLOCK_TAGS):
        if not isinstance(
            element,
            Tag,
        ):
            continue

        if element.name == "h1":
            started = True
            continue

        if not started:
            continue

        text = normalize_text(
            element.get_text(
                " ",
                strip=True,
            )
        )

        if not text:
            continue

        lower = text.casefold()

        if any(phrase in lower for phrase in STOP_PHRASES):
            break

        if blocks and blocks[-1] == text:
            continue

        if element.name in {
            "h2",
            "h3",
            "h4",
        }:
            blocks.append(f"\n{text}\n")
        else:
            blocks.append(text)

    article = normalize_text("\n\n".join(blocks))

    if len(article) < 150:
        raise KBFetchError(f"Extracted article is suspiciously short: {len(article)} characters")

    return article


def fetch_html(
    client: httpx.Client,
    url: str,
    *,
    attempts: int = 3,
) -> httpx.Response:
    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            response = client.get(url)

            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    "Temporary upstream error",
                    request=response.request,
                    response=response,
                )

            response.raise_for_status()

            return response

        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc

            if attempt == attempts:
                break

            time.sleep(0.75 * attempt)

    raise KBFetchError(f"Failed to fetch {url}: {last_error}")


def fetch_document(
    client: httpx.Client,
    source: SourceSpec,
) -> SourceDocument:
    requested_url = str(source.url)

    log.info(
        "fetching_article",
        document_id=source.document_id,
        language=source.language,
        url=requested_url,
    )

    response = fetch_html(
        client,
        requested_url,
    )

    soup = BeautifulSoup(
        response.text,
        "lxml",
    )

    title = extract_title(soup)

    text = extract_article_text(soup)

    document = SourceDocument(
        document_id=source.document_id,
        title=title,
        source_url=str(response.url),
        text=text,
        language=source.language,
        metadata={
            "company": "Kaspersky",
            "product": source.product,
            "topic": source.topic,
            "language": source.language,
            "source_type": "public_support_kb",
            "requested_url": requested_url,
        },
    )

    log.info(
        "article_fetched",
        document_id=source.document_id,
        language=source.language,
        title=title,
        chars=len(text),
        final_url=str(response.url),
    )

    return document


def main() -> None:
    args = parse_args()

    settings = get_settings()

    configure_logging(settings.log_level)

    sources = load_sources(args.sources)

    declared_languages = Counter(source.language for source in sources)

    log.info(
        "source_list_loaded",
        sources=len(sources),
        languages=dict(declared_languages),
    )

    documents: list[SourceDocument] = []

    failures: list[tuple[str, str]] = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0 Safari/537.36"
        ),
        "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
        "Accept-Language": ("ru-RU,ru;q=0.9,en;q=0.8"),
    }

    timeout = httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=10.0,
        pool=10.0,
    )

    with httpx.Client(
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        for index, source in enumerate(
            sources,
            start=1,
        ):
            try:
                document = fetch_document(
                    client,
                    source,
                )

            except KBFetchError as exc:
                failures.append(
                    (
                        source.document_id,
                        str(exc),
                    )
                )

                log.error(
                    "article_fetch_failed",
                    document_id=(source.document_id),
                    error=str(exc),
                )

            else:
                documents.append(document)

            if index < len(sources):
                time.sleep(
                    max(
                        args.delay,
                        0.0,
                    )
                )

    if failures:
        print()
        print("Failed sources:")

        for (
            document_id,
            error,
        ) in failures:
            print(f"  {document_id}: {error}")

        if not args.allow_partial:
            raise SystemExit(
                "Corpus was not written because "
                "one or more sources failed. "
                "Use --allow-partial only for debugging."
            )

    if not documents:
        raise RuntimeError("No articles were fetched")

    write_jsonl(
        documents,
        args.output,
    )

    fetched_languages = Counter(document.language for document in documents)

    print()
    print(f"Fetched: {len(documents)}/{len(sources)}")

    print(f"Languages: {dict(fetched_languages)}")

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
