"""Liveness / readiness probes (no auth, no sensitive data)."""

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.deps import get_container
from app.container import Container

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok"}


async def _probe(url: str) -> str:
    """'ok' when the URL answers without a server error, else the error text."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(url)
        return "ok" if response.status_code < 500 else f"HTTP {response.status_code}"
    except Exception as exc:
        return f"unreachable ({exc.__class__.__name__})"


@router.get("/ready")
async def ready(container: Container = Depends(get_container)) -> JSONResponse:
    """Readiness: database, LLM, embedding and reranker endpoints. 503 if the database is down."""
    s = container.settings
    checks: dict[str, str] = {}
    try:
        async with container.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error ({exc.__class__.__name__})"
    checks["llm"] = await _probe(s.llm_base_url.rstrip("/") + "/models")
    checks["embeddings"] = await _probe(s.embed_base_url.rstrip("/") + "/models")
    checks["reranker"] = await _probe(s.rerank_base_url.rstrip("/") + "/health") if s.rerank_enabled else "disabled"
    status_code = 200 if checks["database"] == "ok" else 503
    return JSONResponse({"status": "ok" if status_code == 200 else "degraded", "checks": checks}, status_code=status_code)
