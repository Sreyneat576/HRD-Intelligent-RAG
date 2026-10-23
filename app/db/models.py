"""ORM models. The schema itself is created by the Alembic migration in `migrations/`."""

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import BigInteger, Computed, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import get_settings


class Base(DeclarativeBase):
    # Fetch server-generated values (timestamps, ids) with RETURNING right after INSERT/UPDATE,
    # so async code never needs a lazy load.
    __mapper_args__ = {"eager_defaults": True}


class DocStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    """One uploaded file. `id` is the document id from Spring's data-management service."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    folder_id: Mapped[str | None] = mapped_column(String(128))
    owner_id: Mapped[str] = mapped_column(String(128))
    filename: Mapped[str] = mapped_column(Text)
    extension: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default=DocStatus.PENDING)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embed_model: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(HALFVEC(get_settings().embed_dim))
    tsv: Mapped[Any] = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))
    page: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(128))
    title: Mapped[str | None] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String(16), default="general")  # general | folder
    folder_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
