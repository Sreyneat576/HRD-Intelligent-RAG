"""HRD Intelligence - RAG test console (Streamlit).

Plays the role of the Spring backend + frontend so the AI service can be tested on its own:
upload documents, watch indexing, inspect chunks, choose which documents a user may access and
chat with streaming answers, citations and retrieval debug info.

Run:  streamlit run ui/app.py
"""

import json
import re
import uuid
from collections.abc import Iterator

import httpx
import pandas as pd
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_TYPES = ["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", "png", "jpg", "jpeg"]
STATUS_ICON = {"pending": "🕓 pending", "processing": "⚙️ processing", "ready": "✅ ready", "failed": "❌ failed"}


class UiSettings(BaseSettings):
    """Reads AI_SERVICE_URL and SERVICE_TOKEN from the environment or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    ai_service_url: str = "http://localhost:8000"
    service_token: str = "change-me"


# ------------------------------------------------------------------------------------ API client
class AiServiceClient:
    """Small HTTP client for the AI service, acting as Spring would (service token + user headers)."""

    def __init__(self, base_url: str, token: str, user_id: str, role: str):
        """Base URL plus the headers Spring would send (service token, user id, role)."""
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}", "X-User-Id": user_id, "X-User-Role": role}

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Send a request and raise a readable error for non-2xx responses."""
        response = httpx.request(method, self.base_url + path, headers=self.headers, timeout=60, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{response.status_code}: {detail}")
        return response

    def ready(self) -> dict:
        """Readiness checks of the service and its model endpoints."""
        return httpx.get(self.base_url + "/ready", timeout=10).json()

    def upload(self, file_name: str, data: bytes, document_id: str, folder_id: str, owner_id: str) -> dict:
        """POST /documents (multipart)."""
        files = {"file": (file_name, data)}
        form = {"document_id": document_id, "owner_id": owner_id, "folder_id": folder_id}
        return self._request("POST", "/api/v1/documents", files=files, data=form).json()

    def documents(self) -> list[dict]:
        """GET /documents."""
        return self._request("GET", "/api/v1/documents").json()

    def chunks(self, document_id: str) -> list[dict]:
        """GET /documents/{id}/chunks."""
        return self._request("GET", f"/api/v1/documents/{document_id}/chunks").json()

    def delete_document(self, document_id: str) -> None:
        """DELETE /documents/{id}."""
        self._request("DELETE", f"/api/v1/documents/{document_id}")

    def retry_document(self, document_id: str) -> None:
        """POST /documents/{id}/retry."""
        self._request("POST", f"/api/v1/documents/{document_id}/retry")

    def sessions(self) -> list[dict]:
        """GET /chat/sessions."""
        return self._request("GET", "/api/v1/chat/sessions").json()

    def create_session(self, scope: str, folder_id: str | None) -> dict:
        """POST /chat/sessions."""
        return self._request("POST", "/api/v1/chat/sessions", json={"scope": scope, "folder_id": folder_id}).json()

    def rename_session(self, session_id: str, title: str) -> None:
        """PATCH /chat/sessions/{id}."""
        self._request("PATCH", f"/api/v1/chat/sessions/{session_id}", json={"title": title})

    def delete_session(self, session_id: str) -> None:
        """DELETE /chat/sessions/{id}."""
        self._request("DELETE", f"/api/v1/chat/sessions/{session_id}")

    def messages(self, session_id: str) -> list[dict]:
        """GET /chat/sessions/{id}/messages."""
        return self._request("GET", f"/api/v1/chat/sessions/{session_id}/messages").json()

    def ask_stream(self, session_id: str, question: str, allowed_ids: list[str], debug: bool) -> Iterator[tuple[str, dict]]:
        """POST /chat/sessions/{id}/messages and yield (event, data) pairs from the SSE stream."""
        body = {"content": question, "allowed_document_ids": allowed_ids, "stream": True, "debug": debug}
        url = f"{self.base_url}/api/v1/chat/sessions/{session_id}/messages"
        with httpx.stream("POST", url, headers=self.headers, json=body, timeout=httpx.Timeout(300, connect=10)) as response:
            if response.status_code >= 400:
                response.read()
                raise RuntimeError(f"{response.status_code}: {response.text}")
            event, data = "message", []
            for line in response.iter_lines():
                if line == "":
                    if data:
                        yield event, json.loads("\n".join(data))
                    event, data = "message", []
                elif line.startswith(":"):
                    continue  # keep-alive ping
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())


# --------------------------------------------------------------------------------------- helpers
def new_document_id(file_name: str) -> str:
    """Simulate the id Spring would assign: readable slug + short random suffix."""
    slug = re.sub(r"[^a-z0-9]+", "-", file_name.lower()).strip("-")[:40]
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def show_citations(citations: list[dict]) -> None:
    """Render the sources an answer cited."""
    if not citations:
        return
    with st.expander(f"📎 Sources ({len(citations)})"):
        for c in citations:
            location = " · ".join(
                p for p in [f"page {c['page']}" if c.get("page") else "", c.get("section") or ""] if p
            )
            st.markdown(f"**[{c['n']}] {c['filename']}** {('— ' + location) if location else ''}")
            st.caption(c.get("snippet", ""))


def show_retrieval(info: dict) -> None:
    """Render the retrieval debug event: search query and ranked sources with their scores."""
    with st.expander(f"🔎 Retrieval debug — {len(info.get('sources', []))} sources from {info.get('candidates', 0)} candidates"):
        st.markdown(f"**Search query:** `{info.get('search_query')}`")
        if info.get("rerank_error"):
            st.warning(f"Reranker unavailable, hybrid ranking used: {info['rerank_error']}")
        if info.get("sources"):
            frame = pd.DataFrame(info["sources"])[
                ["n", "filename", "page", "section", "rerank_score", "rrf_score", "dense_rank", "lexical_rank", "snippet"]
            ]
            st.dataframe(frame, hide_index=True, width="stretch")


# --------------------------------------------------------------------------------------- layout
st.set_page_config(page_title="HRD Intelligence · RAG Console", page_icon="🎓", layout="wide")
settings = UiSettings()

with st.sidebar:
    st.title("🎓 HRD Intelligence")
    st.caption("RAG test console for the AI service")
    base_url = st.text_input("AI service URL", settings.ai_service_url)
    token = st.text_input("Service token", settings.service_token, type="password")
    user_id = st.text_input("Acting as user id", "admin-1", help="Sent as X-User-Id, like Spring would")
    role = st.selectbox("Role", ["ADMIN", "INSTRUCTOR"])
    client = AiServiceClient(base_url, token, user_id, role)
    try:
        readiness = client.ready()
        for name, value in readiness["checks"].items():
            st.markdown(f"{'🟢' if value in ('ok', 'disabled') else '🔴'} **{name}**: {value}")
    except Exception as exc:
        st.error(f"AI service not reachable at {base_url}: {exc}")
        st.stop()

documents_tab, chat_tab = st.tabs(["📄 Documents", "💬 Chat"])

# ------------------------------------------------------------------------------- documents tab
with documents_tab:
    st.subheader("Upload documents")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("Files", type=SUPPORTED_TYPES, accept_multiple_files=True)
        folder_id = st.text_input("Folder id", "general", help="Folder chats only search documents in their folder")
        submitted = st.form_submit_button("Upload & index", type="primary")
    if submitted and files:
        for f in files:
            try:
                result = client.upload(f.name, f.getvalue(), new_document_id(f.name), folder_id, user_id)
                note = "queued for indexing" if result.get("queued") else "same content already indexed"
                st.success(f"{f.name}: {note}")
            except Exception as exc:
                st.error(f"{f.name}: {exc}")

    def is_indexing(docs: list[dict]) -> bool:
        """True while any document is still waiting for or going through indexing."""
        return any(d["status"] in ("pending", "processing") for d in docs)

    try:
        polling = is_indexing(client.documents())
    except Exception:
        polling = False

    @st.fragment(run_every="3s" if polling else None)
    def documents_table() -> None:
        """Document list; refreshes every 3 seconds only while something is being indexed."""
        try:
            docs = client.documents()
        except Exception as exc:
            st.error(str(exc))
            return
        if polling and not is_indexing(docs):
            st.rerun()  # indexing finished: refresh the whole page once, which also stops the polling
        st.subheader(f"Documents ({len(docs)})")
        if not docs:
            st.info("No documents yet. Upload one above.")
            return
        frame = pd.DataFrame(
            [
                {
                    "status": STATUS_ICON.get(d["status"], d["status"]),
                    "filename": d["filename"],
                    "folder": d["folder_id"],
                    "chunks": d["chunk_count"],
                    "pages": d["page_count"],
                    "size KB": round(d["size_bytes"] / 1024, 1),
                    "error": d["error"] or "",
                    "id": d["id"],
                }
                for d in docs
            ]
        )
        st.dataframe(frame, hide_index=True, width="stretch")

    documents_table()

    st.subheader("Inspect a document")
    try:
        all_docs = client.documents()
    except Exception:
        all_docs = []
    if all_docs:
        labels = {d["id"]: f"{d['filename']} ({d['status']})" for d in all_docs}
        chosen = st.selectbox("Document", list(labels), format_func=labels.get)
        doc = next(d for d in all_docs if d["id"] == chosen)
        col_chunks, col_retry, col_delete = st.columns(3)
        if col_retry.button("🔁 Retry indexing", disabled=doc["status"] != "failed"):
            client.retry_document(chosen)
            st.rerun()
        if col_delete.button("🗑️ Delete document"):
            client.delete_document(chosen)
            st.rerun()
        if col_chunks.toggle("Show chunks", value=False):
            chunks = client.chunks(chosen)
            st.caption(f"{len(chunks)} chunks · tokens: {[c['token_count'] for c in chunks]}")
            for c in chunks:
                where = f"page {c['page']}" if c["page"] else ""
                with st.expander(f"#{c['chunk_index']} · {c['token_count']} tokens · {where} {c['section_path'] or ''}"):
                    st.text(c["content"])

# ------------------------------------------------------------------------------------ chat tab
with chat_tab:
    ready_docs = [d for d in client.documents() if d["status"] == "ready"]
    sessions = client.sessions()

    left, right = st.columns([1, 3])
    with left:
        st.markdown("**Chat sessions** (max 10 kept)")
        scope = st.radio("New chat type", ["general", "folder"], horizontal=True)
        folders = sorted({d["folder_id"] for d in ready_docs if d["folder_id"]})
        new_folder = st.selectbox("Folder", folders) if scope == "folder" and folders else None
        if st.button("➕ New chat", width="stretch", disabled=scope == "folder" and not new_folder):
            created = client.create_session(scope, new_folder)
            st.session_state["session_id"] = created["id"]
            st.rerun()
        for s in sessions:
            label = (s["title"] or "New chat") + (f"  📁{s['folder_id']}" if s["scope"] == "folder" else "")
            if st.button(label, key=f"s-{s['id']}", width="stretch"):
                st.session_state["session_id"] = s["id"]
                st.rerun()

    with right:
        current = next((s for s in sessions if s["id"] == st.session_state.get("session_id")), None)
        if current is None:
            st.info("Create or pick a chat session on the left.")
            st.stop()

        head, rename_col, delete_col = st.columns([4, 2, 1])
        head.markdown(f"### {current['title'] or 'New chat'}")
        new_title = rename_col.text_input("Rename", current["title"] or "", label_visibility="collapsed")
        if new_title and new_title != (current["title"] or ""):
            client.rename_session(current["id"], new_title)
            st.rerun()
        if delete_col.button("🗑️", help="Delete this chat"):
            client.delete_session(current["id"])
            st.session_state.pop("session_id", None)
            st.rerun()

        # What Spring would send as allowed_document_ids (the user's permissions / selected files).
        candidates = [d for d in ready_docs if current["scope"] == "general" or d["folder_id"] == current["folder_id"]]
        names = {d["id"]: d["filename"] for d in candidates}
        allowed = st.multiselect(
            "Documents this user may read (simulates Spring permissions / selected files)",
            list(names),
            default=list(names),
            format_func=names.get,
        )
        debug = st.toggle("Show retrieval debug", value=True)

        # Messages go in a container created before the input box, so the input stays below them.
        conversation = st.container()
        with conversation:
            for m in client.messages(current["id"]):
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m["role"] == "assistant":
                        show_citations(m["citations"])

        question = st.chat_input("Ask about your HRD documents…")
        if question:
            with conversation, st.chat_message("user"):
                st.markdown(question)
            with conversation, st.chat_message("assistant"):
                extra: dict = {}

                def answer_tokens() -> Iterator[str]:
                    """Yield answer text from the SSE stream; keep the other events for later."""
                    for event, data in client.ask_stream(current["id"], question, allowed, debug):
                        if event == "token":
                            yield data["text"]
                        else:
                            extra[event] = data

                try:
                    st.write_stream(answer_tokens())
                except Exception as exc:
                    st.error(str(exc))
                if "error" in extra:
                    st.error(extra["error"]["message"])
                show_citations(extra.get("citations", {}).get("citations", []))
                if "retrieval" in extra:
                    show_retrieval(extra["retrieval"])
