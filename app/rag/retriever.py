"""Hybrid retriever over pgvector: dense (HNSW, cosine) + lexical (Postgres full-text), fused with
Reciprocal Rank Fusion. Results are always restricted to the document ids the caller may access."""

import time
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.types import String

from app.core.timing import ms_since
from app.db.session import to_pgvector

SearchMode = Literal["dense", "lexical", "hybrid"]

_CHUNK_COLUMNS = "c.id, c.document_id, c.content, c.page, c.section_path, c.metadata, d.filename"

_DENSE_SQL = text(
    f"""
    SELECT {_CHUNK_COLUMNS}, 1 - (c.embedding <=> CAST(CAST(:qvec AS text) AS halfvec)) AS score
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.document_id = ANY(:ids)
    ORDER BY c.embedding <=> CAST(CAST(:qvec AS text) AS halfvec)
    LIMIT :k
    """
).bindparams(bindparam("ids", type_=ARRAY(String)))

# The question's words are OR-ed (not AND-ed like websearch_to_tsquery), so a chunk matching more
# of the question's terms ranks higher instead of a natural-language question matching nothing.
_LEXICAL_SQL = text(
    f"""
    WITH q AS (
        SELECT CAST((SELECT string_agg(quote_literal(lexeme), ' | ')
                     FROM unnest(tsvector_to_array(to_tsvector('english', :query))) AS lexeme) AS tsquery) AS query
    )
    SELECT {_CHUNK_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
    FROM chunks c JOIN documents d ON d.id = c.document_id, q
    WHERE q.query IS NOT NULL AND c.document_id = ANY(:ids) AND c.tsv @@ q.query
    ORDER BY score DESC
    LIMIT :k
    """
).bindparams(bindparam("ids", type_=ARRAY(String)))


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """score(d) = sum over lists of 1 / (k + rank), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


class HybridPgRetriever(BaseRetriever):
    """LangChain retriever. Build one per request with `for_documents(ids)`."""

    sessionmaker: async_sessionmaker[AsyncSession]
    embeddings: Embeddings
    allowed_document_ids: list[str] = []
    mode: SearchMode = "hybrid"
    dense_k: int = 30
    lexical_k: int = 30
    rrf_k: int = 60
    top_k: int = 20
    hnsw_ef_search: int = 100

    model_config = {"arbitrary_types_allowed": True}

    def for_documents(self, document_ids: list[str], mode: SearchMode | None = None) -> "HybridPgRetriever":
        """Copy of this retriever restricted to `document_ids` (and optionally another search mode)."""
        return self.model_copy(update={"allowed_document_ids": list(document_ids), "mode": mode or self.mode})

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Sync API is not supported: the database driver is async."""
        raise NotImplementedError("HybridPgRetriever is async-only; use ainvoke()")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        """LangChain retriever API: the documents of `asearch` without the timings."""
        documents, _ = await self.asearch(query)
        return documents

    async def asearch(self, query: str) -> tuple[list[Document], dict]:
        """Run dense and/or lexical search on the allowed documents and fuse the rankings with RRF.
        Returns (documents, trace) where trace holds the time of each step in milliseconds."""
        trace: dict[str, Any] = {"embed_ms": 0, "dense_ms": 0, "lexical_ms": 0, "dense_hits": 0, "lexical_hits": 0}
        if not self.allowed_document_ids or not query.strip():
            return [], {**trace, "candidates": 0}

        rows: dict[int, Any] = {}
        dense_ids: list[int] = []
        lexical_ids: list[int] = []
        dense_scores: dict[int, float] = {}
        lexical_scores: dict[int, float] = {}

        qvec = None
        if self.mode in ("dense", "hybrid"):
            started = time.perf_counter()
            qvec = to_pgvector(await self.embeddings.aembed_query(query))
            trace["embed_ms"] = ms_since(started)

        async with self.sessionmaker() as session:
            if qvec is not None:
                started = time.perf_counter()
                # Keep HNSW recall high even when the permission filter removes most neighbours.
                await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.hnsw_ef_search)}"))
                await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
                result = await session.execute(
                    _DENSE_SQL, {"qvec": qvec, "ids": self.allowed_document_ids, "k": self.dense_k}
                )
                for row in result.mappings():
                    rows[row["id"]] = row
                    dense_ids.append(row["id"])
                    dense_scores[row["id"]] = float(row["score"])
                trace["dense_ms"] = ms_since(started)
            if self.mode in ("lexical", "hybrid"):
                started = time.perf_counter()
                result = await session.execute(
                    _LEXICAL_SQL, {"query": query, "ids": self.allowed_document_ids, "k": self.lexical_k}
                )
                for row in result.mappings():
                    rows.setdefault(row["id"], row)
                    lexical_ids.append(row["id"])
                    lexical_scores[row["id"]] = float(row["score"])
                trace["lexical_ms"] = ms_since(started)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion([dense_ids, lexical_ids], k=self.rrf_k)
        ranked = sorted(fused, key=fused.get, reverse=True)[: self.top_k]
        documents = [
            Document(
                page_content=rows[chunk_id]["content"],
                metadata={
                    **(rows[chunk_id]["metadata"] or {}),
                    "chunk_id": chunk_id,
                    "document_id": rows[chunk_id]["document_id"],
                    "filename": rows[chunk_id]["filename"],
                    "page": rows[chunk_id]["page"],
                    "section_path": rows[chunk_id]["section_path"],
                    "rrf_score": round(fused[chunk_id], 6),
                    "dense_rank": dense_ids.index(chunk_id) + 1 if chunk_id in dense_scores else None,
                    "dense_score": round(dense_scores[chunk_id], 4) if chunk_id in dense_scores else None,
                    "lexical_rank": lexical_ids.index(chunk_id) + 1 if chunk_id in lexical_scores else None,
                },
            )
            for chunk_id in ranked
        ]
        trace.update(
            fuse_ms=ms_since(started),
            dense_hits=len(dense_ids),
            lexical_hits=len(lexical_ids),
            candidates=len(documents),
        )
        return documents, trace
