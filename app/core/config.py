import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings come from environment variables or `.env` (see `.env.example`)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service ---------------------------------------------------------------
    service_token: str = "change-me"
    database_url: str = "postgresql+asyncpg://hrd_ai:hrd_ai@localhost:5433/hrd_ai"
    upload_dir: Path = Path("data/uploads")
    max_upload_mb: int = 50

    # --- LLM (OpenAI-compatible: Ollama in dev, vLLM in prod) ------------------
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen3.5:4b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_context_window: int = 6144  # must match the server (OLLAMA_CONTEXT_LENGTH / vLLM --max-model-len)
    llm_timeout_seconds: float = 300  # first call may wait for the model to load
    # Extra JSON body for the chat API. Qwen3.5 "thinks" by default: on Ollama the hidden reasoning used the
    # whole token budget and left the answer empty. Turn it off:
    #   Ollama: {"reasoning_effort": "none"}   vLLM: {"chat_template_kwargs": {"enable_thinking": false}}
    llm_extra_body: dict[str, Any] = {"reasoning_effort": "none"}

    # --- Embeddings (OpenAI-compatible /v1/embeddings) --------------------------
    embed_base_url: str = "http://localhost:11434/v1"
    embed_api_key: str = "ollama"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    embed_batch_size: int = 32

    # --- Reranker (TEI or vLLM /rerank) -----------------------------------------
    rerank_enabled: bool = True
    rerank_base_url: str = "http://localhost:8081"
    rerank_api: Literal["tei", "vllm"] = "tei"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_min_score: float = 0.05
    rerank_timeout_seconds: float = 60

    # --- Chunking ----------------------------------------------------------------
    tokenizer_name: str = "BAAI/bge-m3"
    chunk_max_tokens: int = 450
    chunk_overlap_tokens: int = 60
    slide_merge_tokens: int = 300
    table_rows_per_chunk: int = 25

    # --- OCR ---------------------------------------------------------------------
    # auto = PaddleOCR first, Surya when PaddleOCR's text looks wrong (low confidence / not English, e.g. Khmer)
    ocr_engine: Literal["auto", "paddle", "surya", "tesseract", "none"] = "auto"
    ocr_structure: bool = False  # PP-StructureV3 layout parsing (slow on CPU)
    ocr_paddle_mkldnn: bool = False  # oneDNN acceleration; breaks on Windows CPU with Paddle 3.x
    ocr_lang: str = "en"
    ocr_min_page_chars: int = 50
    ocr_dpi: int = 200
    ocr_fallback_min_confidence: float = 0.85
    ocr_fallback_min_english_ratio: float = 0.04
    # Surya runs in its own virtualenv (PyTorch) as a worker process, see app/ingestion/surya_worker.py
    surya_python: str = ".venv-surya/Scripts/python.exe" if os.name == "nt" else ".venv-surya/bin/python"
    surya_device: str = "cpu"  # keep the small GPU for the LLM
    # Surya uses ~5 GB RAM on the CPU. False = stop its worker after each indexing job (16 GB laptops).
    surya_keep_alive: bool = False

    # --- Ingestion worker ----------------------------------------------------------
    worker_enabled: bool = True
    worker_concurrency: int = 1
    worker_poll_seconds: float = 2.0
    worker_max_attempts: int = 3
    worker_lock_timeout_minutes: int = 30
    ingest_callback_url: str | None = None

    # --- Retrieval ----------------------------------------------------------------
    retrieval_dense_k: int = 30
    retrieval_lexical_k: int = 30
    rrf_k: int = 60
    rerank_candidates: int = 20
    context_top_k: int = 5
    hnsw_ef_search: int = 100

    # --- Chat ---------------------------------------------------------------------
    query_rewrite: bool = True
    history_turns: int = 6
    history_max_tokens: int = 1500
    max_sessions_per_user: int = 10

    # --- Developer console (web UI at /console) ----------------------------------------
    # Lets the browser call the API with a session cookie instead of the service token. Dev only.
    console_enabled: bool = False
    console_user_id: str = "admin-1"

    # --- Service discovery ----------------------------------------------------------
    eureka_enabled: bool = False
    eureka_server: str = "http://localhost:8761/eureka"
    eureka_app_name: str = "ai-service"
    service_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Settings loaded once from the environment / .env and cached."""
    return Settings()
