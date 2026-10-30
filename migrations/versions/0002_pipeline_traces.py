"""Pipeline traces: per-stage timings for document ingestion and chat answers.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-23
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add `trace` (jsonb) to documents and chat_messages."""
    op.execute("ALTER TABLE documents ADD COLUMN trace jsonb NOT NULL DEFAULT '{}'::jsonb")
    op.execute("ALTER TABLE chat_messages ADD COLUMN trace jsonb NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    """Drop the trace columns."""
    op.execute("ALTER TABLE chat_messages DROP COLUMN trace")
    op.execute("ALTER TABLE documents DROP COLUMN trace")
