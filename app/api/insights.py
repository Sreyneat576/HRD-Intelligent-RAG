"""Insights API - model stack, KPIs and pipeline traces (used by the console and for monitoring)."""

from pathlib import Path
from statistics import mean, median
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_container, get_session
from app.container import Container
from app.core.security import require_service_token
from app.db.models import Document

router = APIRouter(prefix="/api/v1", tags=["insights"], dependencies=[Depends(require_service_token)])

CHAT_STAGES = ["prepare_ms", "rewrite_ms", "embed_ms", "dense_ms", "lexical_ms", "fuse_ms", "rerank_ms",
               "llm_ttft_ms", "generate_ms", "first_token_ms", "total_ms", "tokens_per_s"]
INGEST_STAGES = ["queue_ms", "parse_ms", "extract_ms", "chunk_ms", "embed_ms", "store_ms", "total_ms"]

_RECENT_CHAT = text(
    """
    SELECT m.id, m.created_at, m.content AS answer, m.trace, m.citations, s.title, s.user_id, s.scope,
           (SELECT u.content FROM chat_messages u
             WHERE u.session_id = m.session_id AND u.id < m.id AND u.role = 'user'
             ORDER BY u.id DESC LIMIT 1) AS question
    FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id
    WHERE m.role = 'assistant' AND m.trace ? 'total_ms'
    ORDER BY m.id DESC LIMIT :limit
    """
)


def _averages(traces: list[dict], keys: list[str]) -> dict[str, dict[str, float | None]]:
    """Mean and median of each numeric stage over the given traces."""
    out: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [t[key] for t in traces if isinstance(t.get(key), (int, float))]
        out[key] = {"avg": round(mean(values), 1) if values else None,
                    "p50": round(median(values), 1) if values else None, "n": len(values)}
    return out


def _ocr_ms(trace: dict) -> int:
    """OCR time of one ingestion trace (0 when no page needed OCR)."""
    return (trace.get("ocr") or {}).get("ms", 0)


@router.get("/info")
async def info(container: Container = Depends(get_container)) -> dict[str, Any]:
    """The model stack and the main settings (no secrets)."""
    s = container.settings
    return {
        "llm": {"model": s.llm_model, "base_url": s.llm_base_url, "context_window": s.llm_context_window,
                "max_tokens": s.llm_max_tokens, "thinking": "off" if s.llm_extra_body else "default"},
        "embeddings": {"model": s.embed_model, "dim": s.embed_dim, "base_url": s.embed_base_url},
        "reranker": {"enabled": s.rerank_enabled, "model": s.rerank_model, "api": s.rerank_api,
                     "candidates": s.rerank_candidates, "min_score": s.rerank_min_score},
        "ocr": {"mode": s.ocr_engine, "surya_installed": Path(s.surya_python).exists(),
                "fallback_min_confidence": s.ocr_fallback_min_confidence},
        "chunking": {"max_tokens": s.chunk_max_tokens, "overlap_tokens": s.chunk_overlap_tokens,
                     "tokenizer": s.tokenizer_name},
        "retrieval": {"dense_k": s.retrieval_dense_k, "lexical_k": s.retrieval_lexical_k, "rrf_k": s.rrf_k,
                      "context_top_k": s.context_top_k},
        "vector_db": "PostgreSQL + pgvector (HNSW halfvec + GIN full-text)",
        "console_user_id": s.console_user_id,
    }


@router.get("/stats")
async def stats(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    """Document counts plus average / median time of every pipeline stage (last 200 runs each)."""
    rows = (await session.execute(
        select(Document.status, func.count(), func.coalesce(func.sum(Document.chunk_count), 0)).group_by(Document.status)
    )).all()
    by_status = {status: {"count": count, "chunks": int(chunks)} for status, count, chunks in rows}
    chat = [r.trace for r in (await session.execute(_RECENT_CHAT, {"limit": 200})).all()]
    ingest = [
        {**t, "ocr_ms": _ocr_ms(t)}
        for t in (await session.scalars(
            select(Document.trace).where(Document.trace.has_key("total_ms")).order_by(Document.indexed_at.desc()).limit(200)
        )).all()
    ]
    return {
        "documents": {
            "total": sum(v["count"] for v in by_status.values()),
            "chunks": sum(v["chunks"] for v in by_status.values()),
            "by_status": by_status,
        },
        "chat": {"answers": len(chat), "stages": _averages(chat, CHAT_STAGES),
                 "not_found": sum(1 for t in chat if t.get("llm_skipped"))},
        "ingestion": {"runs": len(ingest), "stages": _averages(ingest, INGEST_STAGES + ["ocr_ms"])},
    }


@router.get("/traces")
async def traces(
    limit: int = Query(default=50, le=200), session: AsyncSession = Depends(get_session)
) -> dict[str, list[dict[str, Any]]]:
    """Recent chat answers and indexing runs with their full traces (the console's pipeline log)."""
    chat = [
        {"id": r.id, "created_at": r.created_at, "question": r.question, "answer": r.answer[:240],
         "title": r.title, "user_id": r.user_id, "scope": r.scope, "citations": len(r.citations or []),
         "trace": r.trace}
        for r in (await session.execute(_RECENT_CHAT, {"limit": limit})).all()
    ]
    docs = (await session.scalars(
        select(Document).where(Document.trace.has_key("total_ms")).order_by(Document.indexed_at.desc()).limit(limit)
    )).all()
    ingestion = [
        {"id": d.id, "filename": d.filename, "extension": d.extension, "folder_id": d.folder_id,
         "indexed_at": d.indexed_at, "chunk_count": d.chunk_count, "page_count": d.page_count, "trace": d.trace}
        for d in docs
    ]
    return {"chat": chat, "ingestion": ingestion}
