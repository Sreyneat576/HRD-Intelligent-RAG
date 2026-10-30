"""FastAPI application: `uvicorn app.main:app`."""

import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import chat, documents, health, insights
from app.console.router import STATIC_DIR
from app.console.router import router as console_router
from app.container import Container, build_container
from app.core.config import Settings, get_settings

log = logging.getLogger("app")


async def _register_eureka(settings: Settings) -> None:
    """Register with Eureka so Spring Cloud Gateway / services can call `lb://ai-service`."""
    try:
        import py_eureka_client.eureka_client as eureka_client
    except ImportError:
        log.warning("EUREKA_ENABLED=true but py-eureka-client is not installed (pip install -e .[eureka])")
        return
    await eureka_client.init_async(
        eureka_server=settings.eureka_server,
        app_name=settings.eureka_app_name,
        instance_port=settings.service_port,
    )


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings/container; production uses the environment."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Start-up: build services, start the ingestion worker, register with Eureka. Shutdown: stop them."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        if settings.service_token == "change-me":
            log.warning("SERVICE_TOKEN is the default value - set a real secret outside local development")
        app.state.container = container or build_container(settings)
        if settings.worker_enabled:
            await app.state.container.worker.start()
        if settings.eureka_enabled:
            await _register_eureka(settings)
        yield
        await app.state.container.worker.stop()
        app.state.container.ocr.close()  # stops the Surya worker process, if it was started
        await app.state.container.engine.dispose()

    app = FastAPI(
        title="HRD Intelligence AI Service",
        version="0.1.0",
        description="Document ingestion and RAG chat for HRD Intelligence (LangChain + LangGraph).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.console_secret = secrets.token_urlsafe(32)  # new per process: console cookies expire on restart
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(insights.router)
    app.include_router(console_router)
    app.mount("/console/assets", StaticFiles(directory=STATIC_DIR), name="console-assets")
    return app


app = create_app()
