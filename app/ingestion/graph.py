"""LangGraph ingestion pipeline:

    START -> parse -> split -> embed -> store -> END

parse  : file -> structural blocks (LangChain loaders, OCR for scans/images)
split  : blocks -> retrieval chunks with context headers (structure-aware splitter)
embed  : chunks -> bge-m3 vectors (batched, OpenAI-compatible endpoint)
store  : replace the document's chunks and mark it `ready`, in one transaction

Nodes raise `IngestionError` on failure; the worker decides whether to retry.
"""

import asyncio
from datetime import UTC, datetime
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import DocStatus
from app.db.models import Document as DocumentRow
from app.ingestion.errors import IngestionError
from app.ingestion.loaders import load_blocks
from app.ingestion.ocr import OcrEngine
from app.ingestion.splitter import StructureAwareSplitter
from app.ingestion.store import replace_chunks


class IngestionState(TypedDict, total=False):
    # input
    document_id: str
    filename: str
    extension: str
    path: str
    sha256: str
    # produced by the nodes
    blocks: list[Document]
    page_count: int | None
    chunks: list[Document]
    vectors: list[list[float]]
    stored: bool


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
        blocks, page_count = await asyncio.to_thread(
            load_blocks, state["path"], state["extension"], ocr, settings
        )
        if not any(b.page_content.strip() for b in blocks):
            raise IngestionError("No text could be extracted from this file")
        return {"blocks": blocks, "page_count": page_count}

    async def split(state: IngestionState) -> IngestionState:
        """Cut the blocks into retrieval chunks with context headers."""
        chunks = await asyncio.to_thread(splitter.split, state["blocks"], state["filename"])
        if not chunks:
            raise IngestionError("The file produced no chunks")
        return {"chunks": chunks}

    async def embed(state: IngestionState) -> IngestionState:
        """Embed every chunk with bge-m3 (batched) and check the vector size."""
        try:
            vectors = await embeddings.aembed_documents([c.page_content for c in state["chunks"]])
        except Exception as exc:
            raise IngestionError(f"Embedding service error: {exc}", retryable=True) from exc
        if vectors and len(vectors[0]) != settings.embed_dim:
            raise IngestionError(
                f"Embedding model returned {len(vectors[0])} dimensions, expected EMBED_DIM={settings.embed_dim}"
            )
        return {"vectors": vectors}

    async def store(state: IngestionState) -> IngestionState:
        """Save chunks + vectors and mark the document ready, unless it changed meanwhile."""
        async with sessionmaker() as session, session.begin():
            row = await session.get(DocumentRow, state["document_id"], with_for_update=True)
            # Deleted or re-uploaded while we were working: drop this result.
            if row is None or row.sha256 != state["sha256"]:
                return {"stored": False}
            await replace_chunks(session, row.id, state["chunks"], state["vectors"])
            row.status = DocStatus.READY
            row.error = None
            row.locked_at = None
            row.storage_path = None
            row.page_count = state.get("page_count")
            row.chunk_count = len(state["chunks"])
            row.embed_model = settings.embed_model
            row.indexed_at = datetime.now(UTC)
        return {"stored": True}

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
