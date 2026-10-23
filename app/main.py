"""FastAPI application: `uvicorn app.main:app`."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, documents, health
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
        await app.state.container.engine.dispose()

    app = FastAPI(
        title="HRD Intelligence AI Service",
        version="0.1.0",
        description="Document ingestion and RAG chat for HRD Intelligence (LangChain + LangGraph).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    return app


app = create_app()
