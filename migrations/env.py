"""Alembic environment: runs the migrations in `versions/` against DATABASE_URL (async engine)."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _url() -> str:
    """Database URL: an explicit `sqlalchemy.url` option (used by tests) or the app's DATABASE_URL."""
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    """`alembic upgrade --sql`: print the SQL instead of running it."""
    context.configure(url=_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _run(connection) -> None:
    """Run the migrations on an open (sync-wrapped) connection."""
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Connect with the async engine and run the migrations."""
    engine = create_async_engine(_url())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
