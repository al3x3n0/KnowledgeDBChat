"""
Repair/normalize schema to match current models and ensure admin sync tables exist.

Revision ID: 0008_repair_schema
Revises: 0007_add_source_sync_logs
Create Date: 2025-12-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = '0008_repair_schema'
down_revision = '0007_add_source_sync_logs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) Rename legacy JSON column `metadata` -> `extra_metadata` where needed
    # documents
    op.execute(
        """
        DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='documents' AND column_name='metadata'
        ) AND NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='documents' AND column_name='extra_metadata'
        ) THEN
            ALTER TABLE documents RENAME COLUMN metadata TO extra_metadata;
        END IF;
        END $$;
        """
    )
    # document_chunks
    op.execute(
        """
        DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='document_chunks' AND column_name='metadata'
        ) AND NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='document_chunks' AND column_name='extra_metadata'
        ) THEN
            ALTER TABLE document_chunks RENAME COLUMN metadata TO extra_metadata;
        END IF;
        END $$;
        """
    )
    # chat_sessions
    op.execute(
        """
        DO $$ BEGIN
        IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='chat_sessions' AND column_name='metadata'
        ) AND NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='chat_sessions' AND column_name='extra_metadata'
        ) THEN
            ALTER TABLE chat_sessions RENAME COLUMN metadata TO extra_metadata;
        END IF;
        END $$;
        """
    )

    # 2) Ensure document_sources has is_syncing + last_error
    op.execute(
        """
        ALTER TABLE document_sources
            ADD COLUMN IF NOT EXISTS is_syncing boolean NOT NULL DEFAULT false;
        """
    )
    op.execute(
        """
        ALTER TABLE document_sources
            ADD COLUMN IF NOT EXISTS last_error text;
        """
    )

    # 3) Ensure document_source_sync_logs exists with index
    op.execute(
        """
        DO $$ BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name='document_source_sync_logs'
        ) THEN
            CREATE TABLE document_source_sync_logs (
                id uuid PRIMARY KEY,
                source_id uuid NOT NULL REFERENCES document_sources(id),
                task_id varchar(100),
                status varchar(20) NOT NULL DEFAULT 'running',
                started_at timestamptz DEFAULT now(),
                finished_at timestamptz,
                total_documents integer,
                processed integer,
                created integer,
                updated integer,
                errors integer,
                error_message text
            );
        END IF;
        END $$;
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_sync_logs_source_started
            ON document_source_sync_logs (source_id, started_at);
        """
    )


def downgrade() -> None:
    # Non-destructive downgrade: keep normalized column names and tables.
    # Optionally drop the index/table if needed.
    op.execute(
        """
        DROP INDEX IF EXISTS ix_sync_logs_source_started;
        """
    )
    # do not drop table to avoid data loss in downgrade by default
    # To force drop, uncomment below:
    # op.execute("DROP TABLE IF EXISTS document_source_sync_logs;")

