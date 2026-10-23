"""Background ingestion worker. The `documents` table is the job queue:

* claim:  pending -> processing with `FOR UPDATE SKIP LOCKED` (safe with several replicas)
* run:    the LangGraph ingestion pipeline
* finish: ready (done by the graph's store node) / pending again (retry with backoff) / failed
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import DocStatus
from app.db.models import Document as DocumentRow
from app.ingestion.errors import IngestionError

log = logging.getLogger(__name__)

_CLAIM = text(
    """
    UPDATE documents
    SET status = 'processing', locked_at = now(), attempts = attempts + 1, updated_at = now()
    WHERE id = (
        SELECT id FROM documents
        WHERE status = 'pending' AND updated_at <= now() - make_interval(secs => attempts * 15)
        ORDER BY updated_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    RETURNING id, filename, extension, storage_path, sha256, attempts
    """
)

_RESET_STALE = text(
    """
    UPDATE documents SET status = 'pending', locked_at = NULL, updated_at = now()
    WHERE status = 'processing' AND locked_at < now() - make_interval(mins => :minutes)
    """
)


@dataclass
class Job:
    id: str
    filename: str
    extension: str
    storage_path: str | None
    sha256: str
    attempts: int


class IngestionWorker:
    def __init__(self, settings: Settings, sessionmaker: async_sessionmaker[AsyncSession], graph):
        """Keep the settings, DB sessions and compiled ingestion graph."""
        self.settings = settings
        self.sessionmaker = sessionmaker
        self.graph = graph
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def notify(self) -> None:
        """Called after an upload so the job starts without waiting for the next poll."""
        self._wake.set()

    async def start(self) -> None:
        """Release stale locks, then start `worker_concurrency` background loops."""
        await self.reset_stale(minutes=self.settings.worker_lock_timeout_minutes)
        self._tasks = [
            asyncio.create_task(self._loop(), name=f"ingestion-worker-{i}")
            for i in range(self.settings.worker_concurrency)
        ]

    async def stop(self) -> None:
        """Cancel the background loops (on shutdown)."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def reset_stale(self, minutes: int) -> None:
        """Put 'processing' jobs older than `minutes` back to 'pending' (crashed worker)."""
        async with self.sessionmaker() as session, session.begin():
            await session.execute(_RESET_STALE, {"minutes": minutes})

    async def _loop(self) -> None:
        """Process jobs until the queue is empty, then sleep until notified or the poll interval passes."""
        polls = 0
        while True:
            try:
                if await self.run_once():
                    continue
                polls += 1
                if polls % 300 == 0:
                    await self.reset_stale(minutes=self.settings.worker_lock_timeout_minutes)
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self.settings.worker_poll_seconds)
                except TimeoutError:
                    pass
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Ingestion worker loop error")
                await asyncio.sleep(self.settings.worker_poll_seconds)

    async def run_once(self) -> bool:
        """Claim and process one pending document. Returns False when the queue is empty."""
        async with self.sessionmaker() as session, session.begin():
            row = (await session.execute(_CLAIM)).mappings().first()
        if row is None:
            return False
        await self._process(Job(**row))
        return True

    async def _process(self, job: Job) -> None:
        """Run the ingestion graph for one job and delete the uploaded file when done."""
        log.info("Indexing %s (%s), attempt %d", job.id, job.filename, job.attempts)
        try:
            if not job.storage_path or not Path(job.storage_path).exists():
                raise IngestionError("Uploaded file is missing on disk; upload it again")
            result = await self.graph.ainvoke(
                {
                    "document_id": job.id,
                    "filename": job.filename,
                    "extension": job.extension,
                    "path": job.storage_path,
                    "sha256": job.sha256,
                }
            )
        except Exception as exc:
            await self._fail(job, exc)
            return
        if result.get("stored"):
            Path(job.storage_path).unlink(missing_ok=True)
            log.info("Indexed %s: %d chunks", job.id, len(result["chunks"]))
            await self._callback(job.id, DocStatus.READY, None, len(result["chunks"]))

    async def _fail(self, job: Job, exc: Exception) -> None:
        """Record a failure: back to 'pending' for a retry, or 'failed' when final."""
        retryable = exc.retryable if isinstance(exc, IngestionError) else True
        if not isinstance(exc, IngestionError):
            log.exception("Unexpected error while indexing %s", job.id)
        message = str(exc) or exc.__class__.__name__
        async with self.sessionmaker() as session, session.begin():
            row = await session.get(DocumentRow, job.id, with_for_update=True)
            if row is None or row.sha256 != job.sha256:
                return
            final = not retryable or row.attempts >= self.settings.worker_max_attempts
            row.status = DocStatus.FAILED if final else DocStatus.PENDING
            row.error = message
            row.locked_at = None
        log.warning("Indexing %s failed (%s): %s", job.id, "final" if final else "will retry", message)
        if final:
            await self._callback(job.id, DocStatus.FAILED, message, 0)

    async def _callback(self, document_id: str, status: str, error: str | None, chunk_count: int) -> None:
        """Tell Spring the final indexing status (optional INGEST_CALLBACK_URL)."""
        url = self.settings.ingest_callback_url
        if not url:
            return
        payload = {"document_id": document_id, "status": status, "error": error, "chunk_count": chunk_count}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(url, json=payload, headers={"Authorization": f"Bearer {self.settings.service_token}"})
        except Exception as exc:
            log.warning("Ingestion callback to %s failed: %s", url, exc)
