from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    log_level: str = "INFO"

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct-AWQ"
    llm_context_window: int = Field(default=2048, gt=0)
    llm_max_tokens: int = Field(default=500, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_connect_timeout_seconds: float = Field(default=10.0, gt=0.0)
    llm_request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    embedding_model: str = "ai-forever/ru-en-RoSBERTa"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=8, gt=0)
    embedding_max_length: int = Field(default=512, gt=0)
    embedding_query_prefix: str = "search_query:"
    embedding_document_prefix: str = "search_document:"
    chunk_size_tokens: int = Field(default=192, gt=0)
    chunk_overlap_tokens: int = Field(default=32, ge=0)

    bm25_top_k: int = Field(default=8, gt=0)
    dense_top_k: int = Field(default=8, gt=0)
    retrieval_top_k: int = Field(default=3, gt=0)
    hybrid_bm25_insert_position: int = Field(default=2, ge=2)
    hybrid_bm25_max_rank: int = Field(default=1, gt=0)
    hybrid_bm25_slots: int = Field(default=1, ge=0)

    rag_context_max_tokens: int = Field(default=900, gt=0)
    prompt_overhead_reserve_tokens: int = Field(default=550, ge=0)

    model_cache_dir: Path = Path(".cache/huggingface")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/index")

    @model_validator(mode="after")
    def validate_chunking(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS")
        if self.chunk_size_tokens >= self.embedding_max_length:
            raise ValueError("CHUNK_SIZE_TOKENS must be smaller than EMBEDDING_MAX_LENGTH")
        return self

    @model_validator(mode="after")
    def validate_retrieval(self) -> Settings:
        if self.retrieval_top_k > self.bm25_top_k + self.dense_top_k:
            raise ValueError("RETRIEVAL_TOP_K cannot exceed BM25_TOP_K + DENSE_TOP_K")
        if self.hybrid_bm25_max_rank > self.bm25_top_k:
            raise ValueError("HYBRID_BM25_MAX_RANK cannot exceed BM25_TOP_K")
        if self.hybrid_bm25_slots > self.retrieval_top_k:
            raise ValueError("HYBRID_BM25_SLOTS cannot exceed RETRIEVAL_TOP_K")
        return self

    @model_validator(mode="after")
    def validate_llm_token_budget(self) -> Settings:
        required = (
            self.rag_context_max_tokens + self.prompt_overhead_reserve_tokens + self.llm_max_tokens
        )
        if required > self.llm_context_window:
            raise ValueError(
                f"RAG token budget exceeds the LLM context window: {required} > "
                f"LLM_CONTEXT_WINDOW={self.llm_context_window}"
            )
        return self

    @property
    def llm_models_url(self) -> str:
        return f"{self.llm_base_url.rstrip('/')}/models"

    @property
    def llm_chat_completions_url(self) -> str:
        return f"{self.llm_base_url.rstrip('/')}/chat/completions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
