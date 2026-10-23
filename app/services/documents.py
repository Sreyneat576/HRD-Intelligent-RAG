"""Document lifecycle used by the documents API: upload / replace, rename / move, copy, delete, retry."""

import hashlib
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import DocStatus, Document
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.store import copy_chunks, rename_in_chunks

_READ_SIZE = 1024 * 1024


async def save_upload(upload: UploadFile, upload_dir: Path, extension: str, max_bytes: int) -> tuple[Path, str, int]:
    """Stream the uploaded file to disk under a random name while computing its SHA-256 and size.
    Rejects files larger than `max_bytes` with 413. Returns (path, sha256, size)."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex}{extension}"
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as out:
        while data := await upload.read(_READ_SIZE):
            size += len(data)
            if size > max_bytes:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {max_bytes // (1024 * 1024)} MB")
            digest.update(data)
            out.write(data)
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")
    return path, digest.hexdigest(), size


def file_extension(filename: str) -> str:
    """Return the lower-case extension of `filename`, or raise 415 if the type is not supported."""
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type '{extension}'. Allowed: {allowed}")
    return extension


async def get_document_or_404(session: AsyncSession, document_id: str, *, for_update: bool = False) -> Document:
    """Load a document row or raise 404."""
    doc = await session.get(Document, document_id, with_for_update=for_update)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{document_id}' not found")
    return doc


async def upsert_document(
    session: AsyncSession,
    settings: Settings,
    *,
    document_id: str,
    owner_id: str,
    folder_id: str | None,
    filename: str,
    upload: UploadFile,
    must_exist: bool = False,
) -> tuple[Document, bool]:
    """Create or replace a document and queue it for indexing.

    If the same content (same SHA-256) is already indexed or queued, only the metadata is updated and
    nothing is re-indexed. Returns (document, queued)."""
    extension = file_extension(filename)
    path, sha256, size = await save_upload(upload, settings.upload_dir, extension, settings.max_upload_mb * 1024 * 1024)
    try:
        doc = await session.get(Document, document_id, with_for_update=True)
        if doc is None and must_exist:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{document_id}' not found")

        if doc is not None and doc.sha256 == sha256 and doc.status != DocStatus.FAILED:
            path.unlink(missing_ok=True)
            await _apply_metadata(session, doc, filename=filename, folder_id=folder_id, owner_id=owner_id)
            await session.commit()
            return doc, False

        old_path = doc.storage_path if doc is not None else None
        if doc is None:
            doc = Document(id=document_id)
            session.add(doc)
        # Old chunks stay searchable until the new version is indexed (replaced atomically by the worker).
        doc.owner_id = owner_id
        doc.folder_id = folder_id
        doc.filename = filename
        doc.extension = extension
        doc.size_bytes = size
        doc.sha256 = sha256
        doc.storage_path = str(path)
        doc.status = DocStatus.PENDING
        doc.error = None
        doc.attempts = 0
        doc.locked_at = None
        await session.commit()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if old_path:
        Path(old_path).unlink(missing_ok=True)
    return doc, True


async def _apply_metadata(
    session: AsyncSession,
    doc: Document,
    *,
    filename: str | None = None,
    folder_id: str | None = None,
    owner_id: str | None = None,
) -> None:
    """Update rename / move / owner fields. A rename also rewrites the "[File: ...]" chunk headers."""
    if filename and filename != doc.filename:
        if Path(filename).suffix.lower() != doc.extension:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Renaming cannot change the file extension")
        await rename_in_chunks(session, doc.id, doc.filename, filename)
        doc.filename = filename
    if folder_id is not None:
        doc.folder_id = folder_id
    if owner_id is not None:
        doc.owner_id = owner_id


async def update_document(
    session: AsyncSession,
    document_id: str,
    *,
    filename: str | None,
    folder_id: str | None,
    owner_id: str | None,
) -> Document:
    """Rename / move / change owner without re-indexing."""
    doc = await get_document_or_404(session, document_id, for_update=True)
    await _apply_metadata(session, doc, filename=filename, folder_id=folder_id, owner_id=owner_id)
    await session.commit()
    return doc


async def copy_document(
    session: AsyncSession,
    source_id: str,
    *,
    new_document_id: str,
    folder_id: str | None,
    owner_id: str | None,
    filename: str | None,
) -> Document:
    """Duplicate an indexed document by copying its chunks and vectors (no re-embedding)."""
    source = await get_document_or_404(session, source_id)
    if source.status != DocStatus.READY:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Document '{source_id}' is not indexed yet ({source.status})")
    if await session.get(Document, new_document_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Document '{new_document_id}' already exists")
    new_name = filename or source.filename
    copy = Document(
        id=new_document_id,
        folder_id=folder_id if folder_id is not None else source.folder_id,
        owner_id=owner_id or source.owner_id,
        filename=new_name,
        extension=source.extension,
        size_bytes=source.size_bytes,
        sha256=source.sha256,
        status=DocStatus.READY,
        attempts=0,
        page_count=source.page_count,
        chunk_count=source.chunk_count,
        embed_model=source.embed_model,
        indexed_at=source.indexed_at,
    )
    session.add(copy)
    await session.flush()
    await copy_chunks(session, source.id, copy.id, source.filename, new_name)
    await session.commit()
    return copy


async def delete_document(session: AsyncSession, document_id: str) -> None:
    """Delete a document; its chunks go with it (ON DELETE CASCADE)."""
    doc = await get_document_or_404(session, document_id, for_update=True)
    storage_path = doc.storage_path
    await session.delete(doc)
    await session.commit()
    if storage_path:
        Path(storage_path).unlink(missing_ok=True)


async def retry_document(session: AsyncSession, document_id: str) -> Document:
    """Put a failed document back in the queue (the uploaded file is kept on failure)."""
    doc = await get_document_or_404(session, document_id, for_update=True)
    if doc.status != DocStatus.FAILED:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Only failed documents can be retried (status: {doc.status})")
    if not doc.storage_path or not Path(doc.storage_path).exists():
        raise HTTPException(status.HTTP_409_CONFLICT, "The uploaded file is gone; upload the document again")
    doc.status = DocStatus.PENDING
    doc.attempts = 0
    doc.error = None
    await session.commit()
    return doc


async def list_documents(
    session: AsyncSession, *, folder_id: str | None, owner_id: str | None, status_filter: str | None, limit: int
) -> list[Document]:
    """List documents, newest first, optionally filtered by folder / owner / status."""
    query = select(Document).order_by(Document.created_at.desc()).limit(limit)
    if folder_id:
        query = query.where(Document.folder_id == folder_id)
    if owner_id:
        query = query.where(Document.owner_id == owner_id)
    if status_filter:
        query = query.where(Document.status == status_filter)
    return list((await session.scalars(query)).all())
