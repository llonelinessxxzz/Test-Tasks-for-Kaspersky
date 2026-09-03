from __future__ import annotations

from support_rag.core.schemas import DocumentChunk
from support_rag.retrieval.bm25 import BM25Retriever


def make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        title=chunk_id,
        source_url=f"https://example.com/{chunk_id}",
        text=text,
        chunk_index=0,
        token_count=len(text.split()),
    )


def test_bm25_ranks_lexically_relevant_chunk_first() -> None:
    retriever = BM25Retriever(
        [
            make_chunk(
                "password",
                "Чтобы изменить пароль, откройте настройки учетной записи.",
            ),
            make_chunk(
                "network",
                "Проверьте подключение устройства к сети интернет.",
            ),
            make_chunk(
                "update",
                "Обновление приложения устанавливается автоматически.",
            ),
        ]
    )

    hits = retriever.search(
        "Как изменить пароль учетной записи?",
        top_k=3,
    )

    assert hits[0].chunk.chunk_id == "password"
    assert hits[0].rank == 1
    assert hits[0].retriever == "bm25"


def test_bm25_is_case_insensitive() -> None:
    retriever = BM25Retriever(
        [
            make_chunk(
                "vpn",
                "VPN connection troubleshooting guide",
            ),
            make_chunk(
                "account",
                "Account settings and profile management",
            ),
        ]
    )

    hits = retriever.search(
        "vpn CONNECTION",
        top_k=1,
    )

    assert hits[0].chunk.chunk_id == "vpn"


def test_bm25_returns_empty_result_for_unknown_vocabulary() -> None:
    retriever = BM25Retriever(
        [
            make_chunk(
                "password",
                "Изменение пароля пользователя",
            ),
            make_chunk(
                "update",
                "Обновление приложения",
            ),
        ]
    )

    hits = retriever.search(
        "квантовый синхрофазотрон",
        top_k=2,
    )

    assert hits == []
