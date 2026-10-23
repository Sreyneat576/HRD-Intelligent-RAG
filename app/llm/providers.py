"""Factories for the LangChain model objects. Everything speaks OpenAI-compatible HTTP, so the same
code runs against Ollama (dev laptop) and vLLM (GPU server): only the settings change."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Settings
from app.llm.reranker import RemoteReranker


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Chat LLM (Qwen3 on Ollama or vLLM) through the OpenAI-compatible API, streaming enabled."""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
        streaming=True,
        extra_body=settings.llm_extra_body or None,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """bge-m3 embeddings through the OpenAI-compatible /v1/embeddings API."""
    return OpenAIEmbeddings(
        model=settings.embed_model,
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
        chunk_size=settings.embed_batch_size,
        # Send raw strings: Ollama / vLLM / TEI do not accept tiktoken token arrays.
        check_embedding_ctx_length=False,
        max_retries=3,
    )


def build_reranker(settings: Settings) -> RemoteReranker | None:
    """bge-reranker-v2-m3 client, or None when reranking is disabled."""
    if not settings.rerank_enabled:
        return None
    return RemoteReranker(
        base_url=settings.rerank_base_url,
        api=settings.rerank_api,
        model=settings.rerank_model,
        top_n=settings.context_top_k,
        min_score=settings.rerank_min_score,
        timeout=settings.rerank_timeout_seconds,
    )
