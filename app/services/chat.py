"""Chat sessions, history and one question/answer turn (streams events from the LangGraph RAG graph)."""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.container import Container
from app.core.security import UserContext
from app.core.timing import ms_since
from app.db.models import ChatMessage, ChatSession, Document
from app.rag.prompts import ThinkTagFilter, source_payload, strip_citations

log = logging.getLogger(__name__)

_TITLE_CHARS = 60


# ---------------------------------------------------------------------------- sessions
async def create_session(
    session: AsyncSession, user: UserContext, *, scope: str, folder_id: str | None, title: str | None, max_sessions: int
) -> ChatSession:
    """Create a chat session and keep only the user's `max_sessions` most recent ones (spec: 10)."""
    chat = ChatSession(id=uuid.uuid4(), user_id=user.user_id, scope=scope, folder_id=folder_id, title=title)
    session.add(chat)
    await session.flush()
    recent = func.coalesce(ChatSession.last_message_at, ChatSession.created_at)
    old_ids = select(ChatSession.id).where(ChatSession.user_id == user.user_id).order_by(recent.desc()).offset(max_sessions)
    await session.execute(delete(ChatSession).where(ChatSession.id.in_(old_ids)))
    await session.commit()
    return chat


async def list_sessions(session: AsyncSession, user: UserContext, query: str | None) -> list[ChatSession]:
    """The user's sessions, most recent first; `query` searches titles."""
    recent = func.coalesce(ChatSession.last_message_at, ChatSession.created_at)
    stmt = select(ChatSession).where(ChatSession.user_id == user.user_id).order_by(recent.desc())
    if query:
        stmt = stmt.where(ChatSession.title.ilike(f"%{query}%"))
    return list((await session.scalars(stmt)).all())


async def get_session_or_404(session: AsyncSession, user: UserContext, session_id: uuid.UUID) -> ChatSession:
    """Load a session owned by this user, or 404 (also for other users' sessions)."""
    chat = await session.get(ChatSession, session_id)
    if chat is None or chat.user_id != user.user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found")
    return chat


async def rename_session(session: AsyncSession, user: UserContext, session_id: uuid.UUID, title: str) -> ChatSession:
    """Change a session title."""
    chat = await get_session_or_404(session, user, session_id)
    chat.title = title
    await session.commit()
    return chat


async def delete_session(session: AsyncSession, user: UserContext, session_id: uuid.UUID) -> None:
    """Delete a session and its messages."""
    chat = await get_session_or_404(session, user, session_id)
    await session.delete(chat)
    await session.commit()


async def list_messages(session: AsyncSession, user: UserContext, session_id: uuid.UUID) -> list[ChatMessage]:
    """All messages of a session in order."""
    await get_session_or_404(session, user, session_id)
    rows = await session.scalars(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
    )
    return list(rows.all())


# ----------------------------------------------------------------------------- one turn
async def _load_history(session: AsyncSession, session_id: uuid.UUID, turns: int) -> list[BaseMessage]:
    """The last `turns` question/answer pairs as LangChain messages (citation markers removed)."""
    rows = await session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(turns * 2)
    )
    messages: list[BaseMessage] = []
    for row in reversed(list(rows.all())):
        if row.role == "user":
            messages.append(HumanMessage(row.content))
        else:
            messages.append(AIMessage(strip_citations(row.content)))
    return messages


async def _resolve_allowed_ids(session: AsyncSession, chat: ChatSession, allowed_ids: list[str]) -> list[str]:
    """De-duplicate the ids Spring allowed; a folder chat is further limited to documents in its folder."""
    ids = list(dict.fromkeys(i for i in allowed_ids if i))
    if chat.scope == "folder" and ids:
        rows = await session.scalars(
            select(Document.id).where(Document.id.in_(ids), Document.folder_id == chat.folder_id)
        )
        in_folder = set(rows.all())
        ids = [i for i in ids if i in in_folder]
    return ids


async def ask_events(
    container: Container,
    user: UserContext,
    session_id: uuid.UUID,
    question: str,
    allowed_document_ids: list[str],
    *,
    debug: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Run one chat turn and yield events:

    {"event": "retrieval", ...}  (debug only) search query + ranked sources with scores
    {"event": "token", "text"}   answer text as it is generated
    {"event": "citations", ...}  sources cited in the answer
    {"event": "trace", ...}      per-stage timings (prepare, rewrite, embed, search, rerank, LLM, total)
    {"event": "done", ...}       saved assistant message id
    {"event": "error", ...}      something failed
    """
    started = time.perf_counter()
    settings = container.settings
    async with container.sessionmaker() as session:
        chat = await get_session_or_404(session, user, session_id)
        allowed = await _resolve_allowed_ids(session, chat, allowed_document_ids)
        history = await _load_history(session, chat.id, settings.history_turns)
        session.add(ChatMessage(session_id=chat.id, role="user", content=question, citations=[]))
        chat.last_message_at = datetime.now(UTC)
        if not chat.title:
            chat.title = question[:_TITLE_CHARS] + ("..." if len(question) > _TITLE_CHARS else "")
        await session.commit()

    trace: dict[str, Any] = {"prepare_ms": ms_since(started), "model": settings.llm_model,
                             "allowed_documents": len(allowed)}
    state: dict[str, Any] = {}
    think_filter = ThinkTagFilter()
    streamed: list[str] = []
    inputs = {"question": question, "history": history, "allowed_document_ids": allowed}
    try:
        async for mode, payload in container.rag_graph.astream(inputs, stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") == "generate" and chunk.content:
                    text = think_filter.feed(str(chunk.content))
                    if text:
                        trace.setdefault("first_token_ms", ms_since(started))
                        streamed.append(text)
                        yield {"event": "token", "text": text}
                continue
            for node, update in payload.items():
                update = dict(update or {})
                trace.update(update.pop("trace", None) or {})
                state.update(update)
                if node in ("not_found", "no_documents"):
                    trace.setdefault("first_token_ms", ms_since(started))
                    streamed.append(update["answer"])
                    yield {"event": "token", "text": update["answer"]}
                if node == "rerank" and debug:
                    yield {
                        "event": "retrieval",
                        "search_query": state.get("search_query"),
                        "rerank_error": state.get("rerank_error"),
                        "candidates": len(state.get("candidates") or []),
                        "sources": [source_payload(i, d) for i, d in enumerate(state.get("sources") or [], 1)],
                    }
        tail = think_filter.flush()
        if tail:
            yield {"event": "token", "text": tail}
    except Exception as exc:
        log.exception("Chat turn failed")
        yield {"event": "error", "message": f"The assistant failed to answer: {exc}"}
        return

    answer = state.get("answer") or "".join(streamed).strip()
    citations = state.get("citations") or []
    trace["total_ms"] = ms_since(started)
    trace["search_query"] = state.get("search_query")
    trace["sources"] = [
        {k: v for k, v in source_payload(i, d).items() if k != "snippet"}
        for i, d in enumerate(state.get("sources") or [], 1)
    ]
    async with container.sessionmaker() as session:
        message = ChatMessage(session_id=session_id, role="assistant", content=answer, citations=citations,
                              trace=trace)
        session.add(message)
        chat = await session.get(ChatSession, session_id)
        if chat is not None:
            chat.last_message_at = datetime.now(UTC)
        await session.commit()
        await session.refresh(message)

    yield {"event": "citations", "citations": citations}
    yield {"event": "trace", "trace": trace}
    yield {
        "event": "done",
        "message_id": message.id,
        "answer": answer,
        "search_query": state.get("search_query"),
        "created_at": message.created_at.isoformat(),
    }
