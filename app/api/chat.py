"""Chat API - called by Spring's chat endpoint on behalf of a logged-in Admin / Instructor."""

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.deps import get_container, get_session
from app.api.schemas import AskRequest, AskResponse, MessageOut, SessionCreate, SessionOut, SessionUpdate
from app.container import Container
from app.core.security import UserContext, get_user
from app.services import chat as service

router = APIRouter(prefix="/api/v1/chat/sessions", tags=["chat"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionOut)
async def create_session(
    body: SessionCreate,
    user: UserContext = Depends(get_user),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
) -> SessionOut:
    """Start a general or folder-specific chat session."""
    chat = await service.create_session(
        session,
        user,
        scope=body.scope,
        folder_id=body.folder_id,
        title=body.title,
        max_sessions=container.settings.max_sessions_per_user,
    )
    return SessionOut.model_validate(chat)


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    q: str | None = Query(default=None, description="Search in session titles"),
    user: UserContext = Depends(get_user),
    session: AsyncSession = Depends(get_session),
) -> list[SessionOut]:
    """The user's recent chat sessions."""
    return [SessionOut.model_validate(c) for c in await service.list_sessions(session, user, q)]


@router.get("/{session_id}", response_model=SessionOut)
async def get_session_details(
    session_id: uuid.UUID, user: UserContext = Depends(get_user), session: AsyncSession = Depends(get_session)
) -> SessionOut:
    """One chat session."""
    return SessionOut.model_validate(await service.get_session_or_404(session, user, session_id))


@router.patch("/{session_id}", response_model=SessionOut)
async def rename_session(
    session_id: uuid.UUID,
    body: SessionUpdate,
    user: UserContext = Depends(get_user),
    session: AsyncSession = Depends(get_session),
) -> SessionOut:
    """Rename a chat session."""
    return SessionOut.model_validate(await service.rename_session(session, user, session_id, body.title))


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID, user: UserContext = Depends(get_user), session: AsyncSession = Depends(get_session)
) -> Response:
    """Delete a chat session and its messages."""
    await service.delete_session(session, user, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(
    session_id: uuid.UUID, user: UserContext = Depends(get_user), session: AsyncSession = Depends(get_session)
) -> list[MessageOut]:
    """Conversation history of a session."""
    return [MessageOut.model_validate(m) for m in await service.list_messages(session, user, session_id)]


@router.post("/{session_id}/messages", response_model=AskResponse)
async def ask(
    session_id: uuid.UUID,
    body: AskRequest,
    user: UserContext = Depends(get_user),
    session: AsyncSession = Depends(get_session),
    container: Container = Depends(get_container),
):
    """Ask a question. With `stream=true` (default) the answer arrives as Server-Sent Events:
    `retrieval` (debug only), `token`*, `citations`, `done` - or `error`."""
    await service.get_session_or_404(session, user, session_id)  # 404 before the stream starts
    events = service.ask_events(
        container, user, session_id, body.content, body.allowed_document_ids, debug=body.debug
    )

    if body.stream:
        async def sse():
            """Wrap each event as an SSE message with its type as the SSE event name."""
            async for event in events:
                name = event.pop("event")
                yield {"event": name, "data": json.dumps(event, ensure_ascii=False)}

        return EventSourceResponse(sse(), ping=15)

    result: dict = {}
    async for event in events:
        if event["event"] == "error":
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, event["message"])
        if event["event"] == "retrieval":
            result["sources"] = event["sources"]
        if event["event"] == "citations":
            result["citations"] = event["citations"]
        if event["event"] == "done":
            result["done"] = event
    done = result["done"]
    return AskResponse(
        message=MessageOut(
            id=done["message_id"],
            role="assistant",
            content=done["answer"],
            citations=result.get("citations", []),
            created_at=done["created_at"],
        ),
        search_query=done.get("search_query"),
        sources=result.get("sources"),
    )
