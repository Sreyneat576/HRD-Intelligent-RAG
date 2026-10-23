"""Chunk persistence (pgvector). Vectors are passed as text and cast to halfvec in SQL."""

import json

from langchain_core.documents import Document
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import to_pgvector

# Kept as table columns, not repeated inside the metadata jsonb.
_COLUMN_FIELDS = {"page", "section_path", "token_count", "chunk_index"}

_INSERT_CHUNK = text(
    """
    INSERT INTO chunks (document_id, chunk_index, content, embedding, page, section_path, token_count, metadata)
    VALUES (:document_id, :chunk_index, :content, CAST(CAST(:embedding AS text) AS halfvec),
            :page, :section_path, :token_count, CAST(:metadata AS jsonb))
    """
)

# Rewrites the "[File: <name>" context header after a rename/copy (the embedding is not recomputed).
_HEADER_SWAP = (
    "CASE WHEN starts_with(content, '[File: ' || :old_name) "
    "THEN '[File: ' || :new_name || substr(content, length('[File: ' || :old_name) + 1) ELSE content END"
)


async def replace_chunks(
    session: AsyncSession, document_id: str, chunks: list[Document], vectors: list[list[float]]
) -> None:
    """Delete a document's old chunks and insert the new chunks with their vectors."""
    await session.execute(text("DELETE FROM chunks WHERE document_id = :id"), {"id": document_id})
    rows = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        meta = chunk.metadata
        rows.append(
            {
                "document_id": document_id,
                "chunk_index": meta["chunk_index"],
                "content": chunk.page_content,
                "embedding": to_pgvector(vector),
                "page": meta.get("page"),
                "section_path": meta.get("section_path"),
                "token_count": meta["token_count"],
                "metadata": json.dumps({k: v for k, v in meta.items() if k not in _COLUMN_FIELDS}, default=str),
            }
        )
    if rows:
        await session.execute(_INSERT_CHUNK, rows)


async def rename_in_chunks(session: AsyncSession, document_id: str, old_name: str, new_name: str) -> None:
    """Rewrite the file name in the chunk headers after a rename."""
    await session.execute(
        text(f"UPDATE chunks SET content = {_HEADER_SWAP} WHERE document_id = :id"),
        {"id": document_id, "old_name": old_name, "new_name": new_name},
    )


async def copy_chunks(session: AsyncSession, source_id: str, target_id: str, old_name: str, new_name: str) -> None:
    """Copy all chunks (vectors included) of one document to another."""
    await session.execute(
        text(
            f"""
            INSERT INTO chunks (document_id, chunk_index, content, embedding, page, section_path, token_count, metadata)
            SELECT :target, chunk_index, {_HEADER_SWAP}, embedding, page, section_path, token_count, metadata
            FROM chunks WHERE document_id = :source ORDER BY chunk_index
            """
        ),
        {"source": source_id, "target": target_id, "old_name": old_name, "new_name": new_name},
    )
