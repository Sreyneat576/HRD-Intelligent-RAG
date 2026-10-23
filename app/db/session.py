from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


def create_engine(database_url: str) -> AsyncEngine:
    """Async SQLAlchemy engine (asyncpg driver) with a connection pool."""
    return create_async_engine(database_url, pool_pre_ping=True, pool_size=10, max_overflow=10)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Factory for async sessions; objects stay usable after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)


def to_pgvector(values: list[float]) -> str:
    """Vectors are sent as text and cast in SQL (`CAST(CAST(:v AS text) AS halfvec)`), which avoids
    needing an asyncpg codec for the pgvector types."""
    return "[" + ",".join(f"{v:.6g}" for v in values) + "]"
