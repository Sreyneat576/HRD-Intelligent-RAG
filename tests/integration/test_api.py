"""End-to-end API tests against the real Postgres + pgvector (docker compose up -d db).

Models are faked (hashing embeddings, fixed-answer LLM, no reranker), so these tests check the
service logic: ingestion queue, hybrid SQL search, permission filtering, chat sessions, SSE.
Skipped automatically when the database is not reachable.
"""

import asyncio
import json
import os

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.container import build_container
from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeOcr, HashingEmbeddings, fake_chat_model

pytestmark = pytest.mark.integration

ADMIN_URL = os.environ.get("TEST_DATABASE_ADMIN_URL", "postgresql+asyncpg://hrd_ai:hrd_ai@localhost:5433/hrd_ai")
TEST_DB = "hrd_ai_test"
TOKEN = "test-token"


@pytest.fixture(scope="session")
async def database_url() -> str:
    """Create a fresh test database and run the Alembic migrations on it."""
    engine = create_async_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)"))
            await conn.execute(text(f"CREATE DATABASE {TEST_DB}"))
    except Exception as exc:
        pytest.skip(f"Postgres not reachable ({exc.__class__.__name__}); run: docker compose up -d db")
    finally:
        await engine.dispose()
    url = ADMIN_URL.rsplit("/", 1)[0] + "/" + TEST_DB
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    await asyncio.to_thread(command.upgrade, config, "head")
    return url


def test_settings(database_url, tmp_path, **overrides) -> Settings:
    """Settings for an isolated test app (no worker loop, no reranker, no query rewrite)."""
    return Settings(
        _env_file=None,
        database_url=database_url,
        upload_dir=tmp_path / "uploads",
        service_token=TOKEN,
        worker_enabled=False,
        rerank_enabled=False,
        query_rewrite=False,
        **overrides,
    )


@pytest.fixture
async def api(database_url, tmp_path):
    """(client, container) with an empty database; the worker is driven manually via `index_all`."""
    settings = test_settings(database_url, tmp_path)
    container = build_container(
        settings,
        chat_model=fake_chat_model(),
        embeddings=HashingEmbeddings(settings.embed_dim),
        reranker=None,
        ocr=FakeOcr(),
    )
    app = create_app(settings, container)
    app.state.container = container
    async with container.engine.begin() as conn:
        await conn.execute(text("TRUNCATE documents, chat_sessions CASCADE"))
    headers = {"Authorization": f"Bearer {TOKEN}", "X-User-Id": "user-1"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers) as client:
        yield client, container
    await container.engine.dispose()


async def upload(client, document_id: str, filename: str, content: bytes, folder_id: str = "folder-a") -> dict:
    """Upload a file the way Spring would."""
    response = await client.post(
        "/api/v1/documents",
        files={"file": (filename, content)},
        data={"document_id": document_id, "owner_id": "user-1", "folder_id": folder_id},
    )
    assert response.status_code == 202, response.text
    return response.json()


async def index_all(container) -> None:
    """Run the ingestion worker until the queue is empty."""
    while await container.worker.run_once():
        pass


async def ask(client, session_id: str, question: str, allowed: list[str]) -> dict:
    """Non-streaming question with retrieval debug info."""
    response = await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"content": question, "allowed_document_ids": allowed, "stream": False, "debug": True},
    )
    assert response.status_code == 200, response.text
    return response.json()


async def new_session(client, scope: str = "general", folder_id: str | None = None) -> str:
    response = await client.post("/api/v1/chat/sessions", json={"scope": scope, "folder_id": folder_id})
    assert response.status_code == 201, response.text
    return response.json()["id"]


MEAL = b"Meal allowance policy.\n\nEvery student receives a meal allowance of 50 dollars per month."
KOREAN = b"Korean language allowance.\n\nStudents who pass the TOPIK Korean test receive the Korean allowance."


async def test_upload_index_and_skip_unchanged_content(api):
    client, container = api
    first = await upload(client, "doc-meal", "meal.txt", MEAL)
    assert first["status"] == "pending" and first["queued"] is True
    await index_all(container)

    doc = (await client.get("/api/v1/documents/doc-meal")).json()
    assert doc["status"] == "ready" and doc["chunk_count"] >= 1 and doc["embed_model"]
    chunks = (await client.get("/api/v1/documents/doc-meal/chunks")).json()
    assert chunks[0]["content"].startswith("[File: meal.txt]")

    again = await upload(client, "doc-meal", "meal.txt", MEAL)
    assert again["queued"] is False and again["status"] == "ready"


async def test_answers_cite_only_allowed_documents(api):
    client, container = api
    await upload(client, "doc-meal", "meal.txt", MEAL)
    await upload(client, "doc-korean", "korean.txt", KOREAN)
    await index_all(container)
    session_id = await new_session(client)

    # The Korean document matches best, but the user may only read the meal document.
    result = await ask(client, session_id, "Who receives the Korean allowance?", ["doc-meal"])
    assert {s["document_id"] for s in result["sources"]} == {"doc-meal"}

    result = await ask(client, session_id, "Who receives the Korean allowance?", ["doc-korean"])
    assert result["message"]["citations"][0]["document_id"] == "doc-korean"
    assert result["sources"][0]["document_id"] == "doc-korean"

    history = (await client.get(f"/api/v1/chat/sessions/{session_id}/messages")).json()
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]


async def test_no_allowed_documents_gives_fixed_answer(api):
    client, _ = api
    session_id = await new_session(client)
    result = await ask(client, session_id, "anything", [])
    assert "no documents available" in result["message"]["content"]


async def test_streaming_sse_events(api):
    client, container = api
    await upload(client, "doc-meal", "meal.txt", MEAL)
    await index_all(container)
    session_id = await new_session(client)
    response = await client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"content": "How much is the meal allowance?", "allowed_document_ids": ["doc-meal"]},
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line.split(":", 1)[1].strip() for line in response.text.splitlines() if line.startswith("event:")]
    assert "token" in events and events[-3:] == ["citations", "trace", "done"]
    done = next(
        json.loads(line.split(":", 1)[1])
        for line in response.text.splitlines()[::-1]
        if line.startswith("data:") and "message_id" in line
    )
    assert done["answer"] == "The answer is in the first source [1]."


async def test_folder_session_only_uses_its_folder(api):
    client, container = api
    await upload(client, "doc-a", "meal.txt", MEAL, folder_id="folder-a")
    await upload(client, "doc-b", "korean.txt", KOREAN, folder_id="folder-b")
    await index_all(container)
    session_id = await new_session(client, "folder", "folder-a")
    result = await ask(client, session_id, "Korean allowance", ["doc-a", "doc-b"])
    assert {s["document_id"] for s in result["sources"]} == {"doc-a"}


async def test_rename_copy_and_delete(api):
    client, container = api
    await upload(client, "doc-meal", "meal.txt", MEAL)
    await index_all(container)

    renamed = await client.patch("/api/v1/documents/doc-meal", json={"filename": "meal-policy.txt", "folder_id": "folder-z"})
    assert renamed.json()["filename"] == "meal-policy.txt" and renamed.json()["folder_id"] == "folder-z"
    chunks = (await client.get("/api/v1/documents/doc-meal/chunks")).json()
    assert all(c["content"].startswith("[File: meal-policy.txt]") for c in chunks)

    copy = await client.post("/api/v1/documents/doc-meal/copy", json={"new_document_id": "doc-copy", "filename": "copy.txt"})
    assert copy.status_code == 201 and copy.json()["status"] == "ready"
    copied = (await client.get("/api/v1/documents/doc-copy/chunks")).json()
    assert len(copied) == len(chunks) and copied[0]["content"].startswith("[File: copy.txt]")

    assert (await client.delete("/api/v1/documents/doc-meal")).status_code == 204
    async with container.engine.connect() as conn:
        left = await conn.scalar(text("SELECT count(*) FROM chunks WHERE document_id = 'doc-meal'"))
    assert left == 0
    assert (await client.get("/api/v1/documents/doc-meal")).status_code == 404


async def test_only_ten_sessions_are_kept(api):
    client, _ = api
    for _ in range(12):
        await new_session(client)
    assert len((await client.get("/api/v1/chat/sessions")).json()) == 10


async def test_failed_document_can_be_retried(api):
    client, container = api
    await upload(client, "doc-empty", "empty.txt", b"   \n\n  ")
    await index_all(container)
    doc = (await client.get("/api/v1/documents/doc-empty")).json()
    assert doc["status"] == "failed" and "No text" in doc["error"]
    retried = await client.post("/api/v1/documents/doc-empty/retry")
    assert retried.status_code == 202 and retried.json()["status"] == "pending"


async def test_auth_and_validation(api):
    client, _ = api
    assert (await client.get("/api/v1/documents", headers={"Authorization": "Bearer wrong"})).status_code == 401
    no_user = {"Authorization": f"Bearer {TOKEN}", "X-User-Id": ""}
    assert (await client.get("/api/v1/chat/sessions", headers=no_user)).status_code == 400
    bad_type = await client.post(
        "/api/v1/documents", files={"file": ("x.exe", b"MZ")}, data={"document_id": "x", "owner_id": "u"}
    )
    assert bad_type.status_code == 415


async def test_pipeline_traces_are_recorded_and_reported(api):
    client, container = api
    await upload(client, "doc-meal", "meal.txt", MEAL)
    await index_all(container)
    doc_trace = (await client.get("/api/v1/documents/doc-meal")).json()["trace"]
    for key in ("queue_ms", "parse_ms", "chunk_ms", "embed_ms", "store_ms", "total_ms"):
        assert isinstance(doc_trace[key], int), key
    assert doc_trace["chunks"] >= 1 and doc_trace["ocr"] is None

    session_id = await new_session(client)
    result = await ask(client, session_id, "How much is the meal allowance?", ["doc-meal"])
    trace = result["trace"]
    for key in ("prepare_ms", "embed_ms", "dense_ms", "lexical_ms", "llm_ttft_ms", "generate_ms", "first_token_ms", "total_ms"):
        assert isinstance(trace[key], int), key
    assert trace["sources"][0]["document_id"] == "doc-meal" and trace["search_query"]
    saved = (await client.get(f"/api/v1/chat/sessions/{session_id}/messages")).json()[-1]
    assert saved["trace"]["total_ms"] == trace["total_ms"]

    stats = (await client.get("/api/v1/stats")).json()
    assert stats["documents"]["total"] == 1 and stats["chat"]["answers"] == 1
    assert stats["chat"]["stages"]["total_ms"]["avg"] is not None and stats["ingestion"]["runs"] == 1
    traces = (await client.get("/api/v1/traces?limit=5")).json()
    assert traces["chat"][0]["question"] == "How much is the meal allowance?"
    assert traces["ingestion"][0]["id"] == "doc-meal"
    info = (await client.get("/api/v1/info")).json()
    assert info["llm"]["model"] and info["ocr"]["mode"]


async def test_console_signs_in_with_a_cookie(database_url, tmp_path):
    for enabled in (True, False):
        settings = test_settings(database_url, tmp_path, console_enabled=enabled)
        container = build_container(settings, chat_model=fake_chat_model(), embeddings=HashingEmbeddings(1024),
                                    reranker=None, ocr=FakeOcr())
        app = create_app(settings, container)
        app.state.container = container
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            page = await client.get("/console")
            if enabled:
                assert page.status_code == 200 and "HRD Intelligence" in page.text
                assert (await client.get("/api/v1/documents")).status_code == 200  # cookie, no token
                assert (await client.get("/console/assets/console.js")).status_code == 200
            else:
                assert page.status_code == 404
                client.cookies.set("hrd_console", app.state.console_secret)
                assert (await client.get("/api/v1/documents")).status_code == 401
        await container.engine.dispose()
