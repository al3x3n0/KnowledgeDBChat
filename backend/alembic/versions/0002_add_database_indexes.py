"""Add database indexes

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-02 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Indexes for users table
    op.create_index('ix_users_username', 'users', ['username'], unique=True, if_not_exists=True)
    op.create_index('ix_users_email', 'users', ['email'], unique=True, if_not_exists=True)
    op.create_index('ix_users_role', 'users', ['role'], if_not_exists=True)
    op.create_index('ix_users_is_active', 'users', ['is_active'], if_not_exists=True)
    
    # Indexes for documents table
    op.create_index('ix_documents_source_id', 'documents', ['source_id'], if_not_exists=True)
    op.create_index('ix_documents_is_processed', 'documents', ['is_processed'], if_not_exists=True)
    op.create_index('ix_documents_created_at', 'documents', ['created_at'], if_not_exists=True)
    op.create_index('ix_documents_updated_at', 'documents', ['updated_at'], if_not_exists=True)
    op.create_index('ix_documents_source_identifier', 'documents', ['source_id', 'source_identifier'], if_not_exists=True)
    
    # Indexes for document_chunks table
    op.create_index('ix_document_chunks_document_id', 'document_chunks', ['document_id'], if_not_exists=True)
    op.create_index('ix_document_chunks_chunk_index', 'document_chunks', ['document_id', 'chunk_index'], if_not_exists=True)
    
    # Indexes for chat_sessions table
    op.create_index('ix_chat_sessions_user_id', 'chat_sessions', ['user_id'], if_not_exists=True)
    op.create_index('ix_chat_sessions_last_message_at', 'chat_sessions', ['last_message_at'], if_not_exists=True)
    op.create_index('ix_chat_sessions_is_active', 'chat_sessions', ['is_active'], if_not_exists=True)
    op.create_index('ix_chat_sessions_user_active', 'chat_sessions', ['user_id', 'is_active'], if_not_exists=True)
    
    # Indexes for chat_messages table
    op.create_index('ix_chat_messages_session_id', 'chat_messages', ['session_id'], if_not_exists=True)
    op.create_index('ix_chat_messages_created_at', 'chat_messages', ['created_at'], if_not_exists=True)
    op.create_index('ix_chat_messages_session_created', 'chat_messages', ['session_id', 'created_at'], if_not_exists=True)
    op.create_index('ix_chat_messages_role', 'chat_messages', ['role'], if_not_exists=True)
    
    # Indexes for document_sources table
    op.create_index('ix_document_sources_source_type', 'document_sources', ['source_type'], if_not_exists=True)
    op.create_index('ix_document_sources_is_active', 'document_sources', ['is_active'], if_not_exists=True)
    
    # Indexes for conversation_memories table (already has some, adding more)
    op.create_index('ix_conversation_memories_user_type', 'conversation_memories', ['user_id', 'memory_type'], if_not_exists=True)
    op.create_index('ix_conversation_memories_is_active', 'conversation_memories', ['is_active'], if_not_exists=True)
    op.create_index('ix_conversation_memories_importance', 'conversation_memories', ['importance_score'], if_not_exists=True)
    
    # Indexes for memory_interactions table
    op.create_index('ix_memory_interactions_memory_id', 'memory_interactions', ['memory_id'], if_not_exists=True)
    op.create_index('ix_memory_interactions_session_id', 'memory_interactions', ['session_id'], if_not_exists=True)
    op.create_index('ix_memory_interactions_created_at', 'memory_interactions', ['created_at'], if_not_exists=True)


def downgrade() -> None:
    # Drop indexes in reverse order
    op.drop_index('ix_memory_interactions_created_at', 'memory_interactions', if_exists=True)
    op.drop_index('ix_memory_interactions_session_id', 'memory_interactions', if_exists=True)
    op.drop_index('ix_memory_interactions_memory_id', 'memory_interactions', if_exists=True)
    
    op.drop_index('ix_conversation_memories_importance', 'conversation_memories', if_exists=True)
    op.drop_index('ix_conversation_memories_is_active', 'conversation_memories', if_exists=True)
    op.drop_index('ix_conversation_memories_user_type', 'conversation_memories', if_exists=True)
    
    op.drop_index('ix_document_sources_is_active', 'document_sources', if_exists=True)
    op.drop_index('ix_document_sources_source_type', 'document_sources', if_exists=True)
    
    op.drop_index('ix_chat_messages_role', 'chat_messages', if_exists=True)
    op.drop_index('ix_chat_messages_session_created', 'chat_messages', if_exists=True)
    op.drop_index('ix_chat_messages_created_at', 'chat_messages', if_exists=True)
    op.drop_index('ix_chat_messages_session_id', 'chat_messages', if_exists=True)
    
    op.drop_index('ix_chat_sessions_user_active', 'chat_sessions', if_exists=True)
    op.drop_index('ix_chat_sessions_is_active', 'chat_sessions', if_exists=True)
    op.drop_index('ix_chat_sessions_last_message_at', 'chat_sessions', if_exists=True)
    op.drop_index('ix_chat_sessions_user_id', 'chat_sessions', if_exists=True)
    
    op.drop_index('ix_document_chunks_chunk_index', 'document_chunks', if_exists=True)
    op.drop_index('ix_document_chunks_document_id', 'document_chunks', if_exists=True)
    
    op.drop_index('ix_documents_source_identifier', 'documents', if_exists=True)
    op.drop_index('ix_documents_updated_at', 'documents', if_exists=True)
    op.drop_index('ix_documents_created_at', 'documents', if_exists=True)
    op.drop_index('ix_documents_is_processed', 'documents', if_exists=True)
    op.drop_index('ix_documents_source_id', 'documents', if_exists=True)
    
    op.drop_index('ix_users_is_active', 'users', if_exists=True)
    op.drop_index('ix_users_role', 'users', if_exists=True)
    op.drop_index('ix_users_email', 'users', if_exists=True)
    op.drop_index('ix_users_username', 'users', if_exists=True)

