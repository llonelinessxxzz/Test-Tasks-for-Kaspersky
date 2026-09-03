from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer

from support_rag.core.config import Settings
from support_rag.core.logging import configure_logging
from support_rag.generation.context import ContextBuilder
from support_rag.generation.prompt import build_generation_messages
from support_rag.ingestion.loader import load_chunks_jsonl
from support_rag.retrieval.dense import RoSBERTaEncoder
from support_rag.retrieval.hybrid import HybridRetriever
from support_rag.retrieval.index import load_dense_index
from support_rag.web.models import ContextPayload, Question


class RetrievalRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        chunks = load_chunks_jsonl(settings.processed_data_dir / "chunks.jsonl")
        encoder = RoSBERTaEncoder(settings)
        embeddings = load_dense_index(
            chunks,
            settings.index_dir,
            expected_model=settings.embedding_model,
            expected_max_length=settings.embedding_max_length,
            expected_document_prefix=settings.embedding_document_prefix,
        )
        self.retriever = HybridRetriever(chunks, embeddings, encoder, settings)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model,
            cache_dir=settings.model_cache_dir,
            use_fast=True,
        )
        self.builder = ContextBuilder(self.tokenizer, max_tokens=settings.rag_context_max_tokens)
        self.documents = len({c.document_id for c in chunks})
        self.chunks = len(chunks)

    def retrieve(self, question: str) -> ContextPayload:
        hits = self.retriever.search(question)
        context = self.builder.build(hits)
        messages = [m.model_dump() for m in build_generation_messages(question, context.text)]
        tokens = len(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        if tokens + self.settings.llm_max_tokens > self.settings.llm_context_window:
            raise ValueError("Question is too long for the model context; shorten it")
        return ContextPayload.from_context(
            context,
            list(dict.fromkeys(h.chunk.document_id for h in hits)),
            tokens,
        )


def create_app(runtime=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        settings = Settings()
        configure_logging(settings.log_level)
        app.state.runtime = runtime or RetrievalRuntime(settings)
        yield

    app = FastAPI(
        title="Internal hybrid retrieval", lifespan=lifespan, docs_url=None, redoc_url=None
    )

    @app.get("/healthz")
    def health():
        return {
            "status": "ok",
            "documents": app.state.runtime.documents,
            "chunks": app.state.runtime.chunks,
            "retriever": "dense-first-hybrid",
        }

    @app.post("/retrieve", response_model=ContextPayload)
    def retrieve(body: Question):
        try:
            return app.state.runtime.retrieve(body.question)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
