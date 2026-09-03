from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer

from support_rag.core.config import Settings, get_settings
from support_rag.generation.client import VLLMClient
from support_rag.generation.context import ContextBuilder
from support_rag.ingestion.loader import load_chunks_jsonl
from support_rag.retrieval.dense import RoSBERTaEncoder
from support_rag.retrieval.hybrid import HybridRetriever
from support_rag.retrieval.index import load_dense_index
from support_rag.services.rag import RAGService


@dataclass
class RAGRuntime:
    service: RAGService
    llm_client: VLLMClient

    async def close(self) -> None:
        await self.llm_client.close()


def build_runtime(
    settings: Settings | None = None,
) -> RAGRuntime:
    settings = settings or get_settings()

    chunks_path = settings.processed_data_dir / "chunks.jsonl"

    chunks = load_chunks_jsonl(chunks_path)

    query_encoder = RoSBERTaEncoder(settings)

    document_embeddings = load_dense_index(
        chunks,
        settings.index_dir,
        expected_model=(settings.embedding_model),
        expected_max_length=(settings.embedding_max_length),
        expected_document_prefix=(settings.embedding_document_prefix),
    )

    retriever = HybridRetriever(
        chunks=chunks,
        document_embeddings=document_embeddings,
        query_encoder=query_encoder,
        settings=settings,
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(
        settings.llm_model,
        cache_dir=settings.model_cache_dir,
        use_fast=True,
    )

    context_builder = ContextBuilder(
        llm_tokenizer,
        max_tokens=(settings.rag_context_max_tokens),
    )

    llm_client = VLLMClient(settings)

    service = RAGService(
        retriever=retriever,
        context_builder=context_builder,
        generator=llm_client,
    )

    return RAGRuntime(
        service=service,
        llm_client=llm_client,
    )
