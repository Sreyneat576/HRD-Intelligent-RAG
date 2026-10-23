"""Request / response models (they also define the OpenAPI contract Spring codes against: /docs)."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    folder_id: str | None
    owner_id: str
    filename: str
    extension: str
    size_bytes: int
    status: Literal["pending", "processing", "ready", "failed"]
    error: str | None
    attempts: int
    page_count: int | None
    chunk_count: int
    embed_model: str | None
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None
    queued: bool | None = Field(default=None, description="Upload only: false when identical content was already indexed")


class DocumentUpdate(BaseModel):
    filename: str | None = None
    folder_id: str | None = None
    owner_id: str | None = None


class DocumentCopy(BaseModel):
    new_document_id: str = Field(min_length=1, max_length=128)
    folder_id: str | None = None
    owner_id: str | None = None
    filename: str | None = None


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_index: int
    page: int | None
    section_path: str | None
    token_count: int
    content: str
    meta: dict[str, Any]


class SessionCreate(BaseModel):
    scope: Literal["general", "folder"] = "general"
    folder_id: str | None = None
    title: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _folder_needs_id(self) -> "SessionCreate":
        """A folder chat must say which folder it belongs to."""
        if self.scope == "folder" and not self.folder_id:
            raise ValueError("folder_id is required when scope is 'folder'")
        return self


class SessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    scope: str
    folder_id: str | None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None


class Citation(BaseModel):
    n: int
    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    snippet: str
    score: float | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    citations: list[dict[str, Any]]
    created_at: datetime


class AskRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    allowed_document_ids: list[str] = Field(
        default_factory=list,
        max_length=20000,
        description="Documents this user may read (general chat: all accessible; folder chat: the selected files)",
    )
    stream: bool = True
    debug: bool = Field(default=False, description="Also return the search query and retrieved chunks with scores")


class AskResponse(BaseModel):
    message: MessageOut
    search_query: str | None = None
    sources: list[dict[str, Any]] | None = None
