"""LangGraph ingestion pipeline:

    START -> parse -> split -> embed -> store -> END

parse  : file -> structural blocks (LangChain loaders, OCR for scans/images)
split  : blocks -> retrieval chunks with context headers (structure-aware splitter)
embed  : chunks -> bge-m3 vectors (batched, OpenAI-compatible endpoint)
store  : replace the document's chunks and mark it `ready`, in one transaction

Every node adds its timing to `trace` (merged by a reducer); the store node saves the trace on the
document so the console can show where indexing time went. Nodes raise `IngestionError` on failure;
the worker decides whether to retry.
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.timing import merge_trace, ms_since
from app.db.models import DocStatus
from app.db.models import Document as DocumentRow
from app.ingestion.errors import IngestionError
from app.ingestion.loaders import load_blocks
from app.ingestion.ocr import OcrEngine, OcrRecorder
from app.ingestion.splitter import StructureAwareSplitter
from app.ingestion.store import replace_chunks


class IngestionState(TypedDict, total=False):
    # input
    document_id: str
    filename: str
    extension: str
    path: str
    sha256: str
    started: float  # time.perf_counter() when the worker picked the job
    queue_ms: int
    attempt: int
    # produced by the nodes
    blocks: list[Document]
    page_count: int | None
    chunks: list[Document]
    vectors: list[list[float]]
    stored: bool
    trace: Annotated[dict, merge_trace]


def build_ingestion_graph(
    *,
    settings: Settings,
    ocr: OcrEngine,
    splitter: StructureAwareSplitter,
    embeddings: Embeddings,
    sessionmaker: async_sessionmaker[AsyncSession],
):
    """Wire the ingestion nodes into a compiled LangGraph StateGraph."""

    async def parse(state: IngestionState) -> IngestionState:
        """Read the file into structural blocks (runs in a thread: parsing / OCR is CPU work)."""
        started = time.perf_counter()
        recorder = OcrRecorder(ocr)
        blocks, page_count = await asyncio.to_thread(
            load_blocks, state["path"], state["extension"], recorder, settings
        )
        if not any(b.page_content.strip() for b in blocks):
            raise IngestionError("No text could be extracted from this file")
        ocr_summary = recorder.summary()
        parse_ms = ms_since(started)
        return {
            "blocks": blocks,
            "page_count": page_count,
            "trace": {
                "parse_ms": parse_ms,
                "extract_ms": parse_ms - (ocr_summary["ms"] if ocr_summary else 0),
                "ocr": ocr_summary,
                "blocks": len(blocks),
                "pages": page_count,
            },
        }

    async def split(state: IngestionState) -> IngestionState:
        """Cut the blocks into retrieval chunks with context headers."""
        started = time.perf_counter()
        chunks = await asyncio.to_thread(splitter.split, state["blocks"], state["filename"])
        if not chunks:
            raise IngestionError("The file produced no chunks")
        return {
            "chunks": chunks,
            "trace": {
                "chunk_ms": ms_since(started),
                "chunks": len(chunks),
                "tokens": sum(c.metadata["token_count"] for c in chunks),
            },
        }

    async def embed(state: IngestionState) -> IngestionState:
        """Embed every chunk with bge-m3 (batched) and check the vector size."""
        started = time.perf_counter()
        try:
            vectors = await embeddings.aembed_documents([c.page_content for c in state["chunks"]])
        except Exception as exc:
            raise IngestionError(f"Embedding service error: {exc}", retryable=True) from exc
        if vectors and len(vectors[0]) != settings.embed_dim:
            raise IngestionError(
                f"Embedding model returned {len(vectors[0])} dimensions, expected EMBED_DIM={settings.embed_dim}"
            )
        embed_ms = ms_since(started)
        return {
            "vectors": vectors,
            "trace": {
                "embed_ms": embed_ms,
                "embed_model": settings.embed_model,
                "embed_chunks_per_s": round(len(vectors) / max(embed_ms / 1000, 1e-3), 1),
            },
        }

    async def store(state: IngestionState) -> IngestionState:
        """Save chunks + vectors and mark the document ready, unless it changed meanwhile."""
        started = time.perf_counter()
        async with sessionmaker() as session, session.begin():
            row = await session.get(DocumentRow, state["document_id"], with_for_update=True)
            # Deleted or re-uploaded while we were working: drop this result.
            if row is None or row.sha256 != state["sha256"]:
                return {"stored": False}
            await replace_chunks(session, row.id, state["chunks"], state["vectors"])
            store_ms = ms_since(started)
            trace = {
                **(state.get("trace") or {}),
                "queue_ms": state.get("queue_ms"),
                "store_ms": store_ms,
                "total_ms": ms_since(state["started"]) if state.get("started") else None,
                "attempt": state.get("attempt"),
            }
            row.status = DocStatus.READY
            row.error = None
            row.locked_at = None
            row.storage_path = None
            row.page_count = state.get("page_count")
            row.chunk_count = len(state["chunks"])
            row.embed_model = settings.embed_model
            row.indexed_at = datetime.now(UTC)
            row.trace = trace
        return {"stored": True, "trace": {"store_ms": store_ms, "total_ms": trace["total_ms"]}}

    graph = StateGraph(IngestionState)
    graph.add_node("parse", parse)
    graph.add_node("split", split)
    graph.add_node("embed", embed)
    graph.add_node("store", store)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "split")
    graph.add_edge("split", "embed")
    graph.add_edge("embed", "store")
    graph.add_edge("store", END)
    return graph.compile(name="ingestion")
