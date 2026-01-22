"""add knowledge graph tables

Revision ID: 0003_add_knowledge_graph
Revises: 0002
Create Date: 2025-11-28
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '0003_add_knowledge_graph'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'kg_entities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('canonical_name', sa.String(length=512), nullable=False),
        sa.Column('entity_type', sa.String(length=64), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('properties', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_kg_entities_type_name', 'kg_entities', ['entity_type', 'canonical_name'])

    op.create_table(
        'kg_entity_mentions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('kg_entities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('text', sa.String(length=512), nullable=False),
        sa.Column('start_pos', sa.Integer(), nullable=True),
        sa.Column('end_pos', sa.Integer(), nullable=True),
        sa.Column('sentence', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_kg_mentions_doc', 'kg_entity_mentions', ['document_id'])
    op.create_index('ix_kg_mentions_chunk', 'kg_entity_mentions', ['chunk_id'])

    op.create_table(
        'kg_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('relation_type', sa.String(length=64), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('inferred', sa.Boolean(), nullable=True),
        sa.Column('source_entity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('kg_entities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('target_entity_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('kg_entities.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index('ix_kg_relations_type', 'kg_relationships', ['relation_type'])
    op.create_index('ix_kg_relations_doc', 'kg_relationships', ['document_id'])
    op.create_unique_constraint('uq_kg_relation_once_per_doc', 'kg_relationships', ['relation_type', 'source_entity_id', 'target_entity_id', 'document_id'])


def downgrade() -> None:
    op.drop_constraint('uq_kg_relation_once_per_doc', 'kg_relationships', type_='unique')
    op.drop_index('ix_kg_relations_doc', table_name='kg_relationships')
    op.drop_index('ix_kg_relations_type', table_name='kg_relationships')
    op.drop_table('kg_relationships')

    op.drop_index('ix_kg_mentions_chunk', table_name='kg_entity_mentions')
    op.drop_index('ix_kg_mentions_doc', table_name='kg_entity_mentions')
    op.drop_table('kg_entity_mentions')

    op.drop_index('ix_kg_entities_type_name', table_name='kg_entities')
    op.drop_table('kg_entities')
