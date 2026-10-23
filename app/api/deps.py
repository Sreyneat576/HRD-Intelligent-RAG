"""FastAPI dependencies shared by the routers."""

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.container import Container


def get_container(request: Request) -> Container:
    """The service container created at startup (see app/main.py)."""
    return request.app.state.container


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """One database session per request."""
    async with request.app.state.container.sessionmaker() as session:
        yield session
