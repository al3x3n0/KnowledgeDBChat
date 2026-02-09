"""Add AI training tables for AI Hub.

Revision ID: 0038
Revises: 0037
Create Date: 2024-01-29

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '0038'
down_revision = '0037'
branch_labels = None
depends_on = None


def upgrade():
    # Create training_datasets table
    op.create_table(
        'training_datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('dataset_type', sa.String(50), nullable=False, server_default='instruction'),
        sa.Column('format', sa.String(50), nullable=False, server_default='alpaca'),
        sa.Column('source_document_ids', postgresql.JSON(), nullable=True),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('token_count', sa.BigInteger(), nullable=False, server_default='0'),
        sa.Column('is_validated', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('validation_errors', postgresql.JSON(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('parent_dataset_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('status', sa.String(30), nullable=False, server_default='draft'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['parent_dataset_id'], ['training_datasets.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_training_datasets_user_id', 'training_datasets', ['user_id'])
    op.create_index('ix_training_datasets_status', 'training_datasets', ['status'])
    op.create_index('ix_training_datasets_dataset_type', 'training_datasets', ['dataset_type'])
    op.create_index('ix_training_datasets_created_at', 'training_datasets', ['created_at'])

    # Create dataset_samples table
    op.create_table(
        'dataset_samples',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('sample_index', sa.Integer(), nullable=False),
        sa.Column('content', postgresql.JSON(), nullable=False),
        sa.Column('source_document_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('input_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('output_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_flagged', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('flag_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['training_datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['source_document_id'], ['documents.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_dataset_samples_dataset_id', 'dataset_samples', ['dataset_id'])
    op.create_index('ix_dataset_samples_sample_index', 'dataset_samples', ['sample_index'])

    # Create model_adapters table (before training_jobs since training_jobs references it)
    op.create_table(
        'model_adapters',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('display_name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('base_model', sa.String(200), nullable=False),
        sa.Column('adapter_type', sa.String(30), nullable=False, server_default='lora'),
        sa.Column('adapter_config', postgresql.JSON(), nullable=True),
        sa.Column('adapter_path', sa.String(500), nullable=True),
        sa.Column('adapter_size', sa.BigInteger(), nullable=True),
        sa.Column('training_job_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('training_metrics', postgresql.JSON(), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('status', sa.String(30), nullable=False, server_default='ready'),
        sa.Column('is_deployed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('deployment_config', postgresql.JSON(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('tags', postgresql.JSON(), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_model_adapters_user_id', 'model_adapters', ['user_id'])
    op.create_index('ix_model_adapters_status', 'model_adapters', ['status'])
    op.create_index('ix_model_adapters_base_model', 'model_adapters', ['base_model'])
    op.create_index('ix_model_adapters_is_deployed', 'model_adapters', ['is_deployed'])

    # Create training_jobs table
    op.create_table(
        'training_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('training_method', sa.String(30), nullable=False, server_default='lora'),
        sa.Column('training_backend', sa.String(30), nullable=False, server_default='local'),
        sa.Column('base_model', sa.String(200), nullable=False),
        sa.Column('base_model_provider', sa.String(50), nullable=False, server_default='ollama'),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('hyperparameters', postgresql.JSON(), nullable=True),
        sa.Column('resource_config', postgresql.JSON(), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(30), nullable=False, server_default='pending'),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('current_step', sa.Integer(), nullable=True),
        sa.Column('total_steps', sa.Integer(), nullable=True),
        sa.Column('current_epoch', sa.Integer(), nullable=True),
        sa.Column('total_epochs', sa.Integer(), nullable=True),
        sa.Column('training_metrics', postgresql.JSON(), nullable=True),
        sa.Column('final_metrics', postgresql.JSON(), nullable=True),
        sa.Column('output_adapter_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('celery_task_id', sa.String(100), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['training_datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['output_adapter_id'], ['model_adapters.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_training_jobs_user_id', 'training_jobs', ['user_id'])
    op.create_index('ix_training_jobs_status', 'training_jobs', ['status'])
    op.create_index('ix_training_jobs_dataset_id', 'training_jobs', ['dataset_id'])
    op.create_index('ix_training_jobs_created_at', 'training_jobs', ['created_at'])

    # Add training_job_id FK to model_adapters now that training_jobs exists
    op.create_foreign_key(
        'fk_model_adapters_training_job_id',
        'model_adapters',
        'training_jobs',
        ['training_job_id'],
        ['id'],
        ondelete='SET NULL'
    )

    # Create training_checkpoints table
    op.create_table(
        'training_checkpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('step', sa.Integer(), nullable=False),
        sa.Column('epoch', sa.Float(), nullable=True),
        sa.Column('checkpoint_path', sa.String(500), nullable=True),
        sa.Column('loss', sa.Float(), nullable=True),
        sa.Column('metrics', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['training_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_training_checkpoints_job_id', 'training_checkpoints', ['job_id'])
    op.create_index('ix_training_checkpoints_step', 'training_checkpoints', ['step'])


def downgrade():
    # Drop training_checkpoints
    op.drop_index('ix_training_checkpoints_step', table_name='training_checkpoints')
    op.drop_index('ix_training_checkpoints_job_id', table_name='training_checkpoints')
    op.drop_table('training_checkpoints')

    # Drop FK from model_adapters to training_jobs
    op.drop_constraint('fk_model_adapters_training_job_id', 'model_adapters', type_='foreignkey')

    # Drop training_jobs
    op.drop_index('ix_training_jobs_created_at', table_name='training_jobs')
    op.drop_index('ix_training_jobs_dataset_id', table_name='training_jobs')
    op.drop_index('ix_training_jobs_status', table_name='training_jobs')
    op.drop_index('ix_training_jobs_user_id', table_name='training_jobs')
    op.drop_table('training_jobs')

    # Drop model_adapters
    op.drop_index('ix_model_adapters_is_deployed', table_name='model_adapters')
    op.drop_index('ix_model_adapters_base_model', table_name='model_adapters')
    op.drop_index('ix_model_adapters_status', table_name='model_adapters')
    op.drop_index('ix_model_adapters_user_id', table_name='model_adapters')
    op.drop_table('model_adapters')

    # Drop dataset_samples
    op.drop_index('ix_dataset_samples_sample_index', table_name='dataset_samples')
    op.drop_index('ix_dataset_samples_dataset_id', table_name='dataset_samples')
    op.drop_table('dataset_samples')

    # Drop training_datasets
    op.drop_index('ix_training_datasets_created_at', table_name='training_datasets')
    op.drop_index('ix_training_datasets_dataset_type', table_name='training_datasets')
    op.drop_index('ix_training_datasets_status', table_name='training_datasets')
    op.drop_index('ix_training_datasets_user_id', table_name='training_datasets')
    op.drop_table('training_datasets')
