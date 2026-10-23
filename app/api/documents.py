"""Documents API - called by Spring's data-management service when files change."""

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_container, get_session
from app.api.schemas import ChunkOut, DocumentCopy, DocumentOut, DocumentUpdate
from app.container import Container
from app.core.security import require_service_token
from app.db.models import Chunk
from app.services import documents as service

router = APIRouter(prefix="/api/v1/documents", tags=["documents"], dependencies=[Depends(require_service_token)])


def _out(doc, queued: bool | None = None) -> DocumentOut:
    """ORM row -> response model."""
    return DocumentOut.model_validate(doc).model_copy(update={"queued": queued})


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    document_id: str = Form(..., max_length=128),
    owner_id: str = Form(..., max_length=128),
    folder_id: str | None = Form(default=None, max_length=128),
    filename: str | None = Form(default=None, description="Defaults to the uploaded file's name"),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
) -> DocumentOut:
    """Upload a new file (or a new version of an existing id) and queue it for indexing."""
    doc, queued = await service.upsert_document(
        session,
        container.settings,
        document_id=document_id,
        owner_id=owner_id,
        folder_id=folder_id,
        filename=filename or file.filename or "upload",
        upload=file,
    )
    if queued:
        container.worker.notify()
    return _out(doc, queued)


@router.put("/{document_id}", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentOut)
async def replace_document(
    document_id: str,
    file: UploadFile = File(...),
    owner_id: str = Form(..., max_length=128),
    folder_id: str | None = Form(default=None, max_length=128),
    filename: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
) -> DocumentOut:
    """Replace the file of an existing document and re-index it."""
    doc, queued = await service.upsert_document(
        session,
        container.settings,
        document_id=document_id,
        owner_id=owner_id,
        folder_id=folder_id,
        filename=filename or file.filename or "upload",
        upload=file,
        must_exist=True,
    )
    if queued:
        container.worker.notify()
    return _out(doc, queued)


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    folder_id: str | None = None,
    owner_id: str | None = None,
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=200, le=1000),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentOut]:
    """List documents (newest first)."""
    docs = await service.list_documents(
        session, folder_id=folder_id, owner_id=owner_id, status_filter=status_filter, limit=limit
    )
    return [_out(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, session: AsyncSession = Depends(get_session)) -> DocumentOut:
    """Indexing status of one document (pending / processing / ready / failed)."""
    return _out(await service.get_document_or_404(session, document_id))


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
async def get_chunks(
    document_id: str, limit: int = Query(default=200, le=2000), session: AsyncSession = Depends(get_session)
) -> list[ChunkOut]:
    """The chunks a document was split into (for debugging chunking quality)."""
    await service.get_document_or_404(session, document_id)
    rows = await session.scalars(
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index).limit(limit)
    )
    return [ChunkOut.model_validate(r) for r in rows]


@router.patch("/{document_id}", response_model=DocumentOut)
async def update_document(
    document_id: str, body: DocumentUpdate, session: AsyncSession = Depends(get_session)
) -> DocumentOut:
    """Rename / move / change owner. No re-indexing."""
    doc = await service.update_document(
        session, document_id, filename=body.filename, folder_id=body.folder_id, owner_id=body.owner_id
    )
    return _out(doc)


@router.post("/{document_id}/copy", status_code=status.HTTP_201_CREATED, response_model=DocumentOut)
async def copy_document(
    document_id: str, body: DocumentCopy, session: AsyncSession = Depends(get_session)
) -> DocumentOut:
    """Duplicate a document (Spring's "Duplicate" action); chunks and vectors are copied."""
    doc = await service.copy_document(
        session,
        document_id,
        new_document_id=body.new_document_id,
        folder_id=body.folder_id,
        owner_id=body.owner_id,
        filename=body.filename,
    )
    return _out(doc)


@router.post("/{document_id}/retry", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentOut)
async def retry_document(
    document_id: str, session: AsyncSession = Depends(get_session), container: Container = Depends(get_container)
) -> DocumentOut:
    """Re-queue a failed document."""
    doc = await service.retry_document(session, document_id)
    container.worker.notify()
    return _out(doc, True)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str, session: AsyncSession = Depends(get_session)) -> Response:
    """Delete a document and all its chunks."""
    await service.delete_document(session, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
