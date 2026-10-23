"""Initial schema: documents, chunks (pgvector halfvec + full-text), chat sessions/messages.

Revision ID: 0001
Revises:
Create Date: 2026-10-19
"""

from alembic import op

from app.core.config import get_settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the pgvector extension, the tables and the HNSW / GIN / btree indexes."""
    dim = get_settings().embed_dim
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE documents (
            id            varchar(128) PRIMARY KEY,
            folder_id     varchar(128),
            owner_id      varchar(128) NOT NULL,
            filename      text NOT NULL,
            extension     varchar(16) NOT NULL,
            size_bytes    bigint NOT NULL,
            sha256        varchar(64) NOT NULL,
            storage_path  text,
            status        varchar(16) NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
            error         text,
            attempts      integer NOT NULL DEFAULT 0,
            locked_at     timestamptz,
            page_count    integer,
            chunk_count   integer NOT NULL DEFAULT 0,
            embed_model   text,
            created_at    timestamptz NOT NULL DEFAULT now(),
            updated_at    timestamptz NOT NULL DEFAULT now(),
            indexed_at    timestamptz
        )
        """
    )
    op.execute("CREATE INDEX ix_documents_folder ON documents (folder_id)")
    op.execute("CREATE INDEX ix_documents_pending ON documents (updated_at) WHERE status = 'pending'")

    op.execute(
        f"""
        CREATE TABLE chunks (
            id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id   varchar(128) NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            chunk_index   integer NOT NULL,
            content       text NOT NULL,
            embedding     halfvec({dim}) NOT NULL,
            tsv           tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            page          integer,
            section_path  text,
            token_count   integer NOT NULL,
            metadata      jsonb NOT NULL DEFAULT '{{}}'::jsonb
        )
        """
    )
    op.execute("CREATE INDEX ix_chunks_document ON chunks (document_id)")
    # Dense index: HNSW (no training step, works on a growing table), cosine on half precision.
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks "
        "USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 128)"
    )
    # Lexical index: full-text search for names, emails, codes.
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")

    op.execute(
        """
        CREATE TABLE chat_sessions (
            id               uuid PRIMARY KEY,
            user_id          varchar(128) NOT NULL,
            title            text,
            scope            varchar(16) NOT NULL DEFAULT 'general' CHECK (scope IN ('general', 'folder')),
            folder_id        varchar(128),
            created_at       timestamptz NOT NULL DEFAULT now(),
            updated_at       timestamptz NOT NULL DEFAULT now(),
            last_message_at  timestamptz
        )
        """
    )
    op.execute("CREATE INDEX ix_chat_sessions_user ON chat_sessions (user_id, last_message_at DESC)")
    op.execute(
        """
        CREATE TABLE chat_messages (
            id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            session_id  uuid NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
            role        varchar(16) NOT NULL CHECK (role IN ('user', 'assistant')),
            content     text NOT NULL,
            citations   jsonb NOT NULL DEFAULT '[]'::jsonb,
            created_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX ix_chat_messages_session ON chat_messages (session_id, id)")


def downgrade() -> None:
    """Drop all tables created by `upgrade`."""
    op.execute("DROP TABLE IF EXISTS chat_messages")
    op.execute("DROP TABLE IF EXISTS chat_sessions")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS documents")
