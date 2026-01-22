"""Add presentation_templates table and template support to presentation_jobs.

Revision ID: 0018_add_presentation_templates
Revises: 0017_add_presentation_jobs
Create Date: 2024-06-22 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "0018_add_presentation_templates"
down_revision = "0017_add_presentation_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create presentation_templates table
    op.create_table(
        "presentation_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),  # NULL = system template
        # Template info
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # Template type
        sa.Column("template_type", sa.String(length=50), nullable=False, server_default=sa.text("'theme'")),
        # For PPTX templates
        sa.Column("file_path", sa.String(length=500), nullable=True),
        # Theme configuration (JSON)
        sa.Column("theme_config", postgresql.JSON(), nullable=True),
        # Preview image
        sa.Column("preview_path", sa.String(length=500), nullable=True),
        # Flags
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        # Foreign key
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    # Create indexes
    op.create_index("ix_presentation_templates_user_id", "presentation_templates", ["user_id"])
    op.create_index("ix_presentation_templates_is_system", "presentation_templates", ["is_system"])
    op.create_index("ix_presentation_templates_is_public", "presentation_templates", ["is_public"])

    # Add template_id and custom_theme columns to presentation_jobs
    op.add_column(
        "presentation_jobs",
        sa.Column("template_id", postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.add_column(
        "presentation_jobs",
        sa.Column("custom_theme", postgresql.JSON(), nullable=True)
    )

    # Add foreign key constraint
    op.create_foreign_key(
        "fk_presentation_jobs_template_id",
        "presentation_jobs",
        "presentation_templates",
        ["template_id"],
        ["id"],
        ondelete="SET NULL"
    )

    # Create index for template lookups
    op.create_index("ix_presentation_jobs_template_id", "presentation_jobs", ["template_id"])

    # Insert system templates (built-in themes)
    op.execute("""
        INSERT INTO presentation_templates (id, name, description, template_type, theme_config, is_system, is_public, is_active)
        VALUES
        (
            gen_random_uuid(),
            'Professional',
            'Clean, corporate look with dark blue accents',
            'theme',
            '{"colors": {"title_color": "#1a365d", "accent_color": "#2e86ab", "text_color": "#333333", "bg_color": "#ffffff"}, "fonts": {"title_font": "Calibri", "body_font": "Calibri"}, "sizes": {"title_size": 44, "subtitle_size": 24, "heading_size": 36, "body_size": 20, "bullet_size": 18}}',
            true,
            true,
            true
        ),
        (
            gen_random_uuid(),
            'Casual',
            'Friendly and approachable with warm colors',
            'theme',
            '{"colors": {"title_color": "#4a90d9", "accent_color": "#ff6b6b", "text_color": "#2d3a4a", "bg_color": "#f8f9fa"}, "fonts": {"title_font": "Arial", "body_font": "Arial"}, "sizes": {"title_size": 48, "subtitle_size": 26, "heading_size": 38, "body_size": 22, "bullet_size": 20}}',
            true,
            true,
            true
        ),
        (
            gen_random_uuid(),
            'Technical',
            'Developer-focused with monospace fonts',
            'theme',
            '{"colors": {"title_color": "#007acc", "accent_color": "#28a745", "text_color": "#24292e", "bg_color": "#ffffff"}, "fonts": {"title_font": "Consolas", "body_font": "Segoe UI"}, "sizes": {"title_size": 40, "subtitle_size": 22, "heading_size": 32, "body_size": 18, "bullet_size": 16}}',
            true,
            true,
            true
        ),
        (
            gen_random_uuid(),
            'Modern',
            'Contemporary design with bold contrasts',
            'theme',
            '{"colors": {"title_color": "#2c3e50", "accent_color": "#e74c3c", "text_color": "#34495e", "bg_color": "#ecf0f1"}, "fonts": {"title_font": "Segoe UI", "body_font": "Segoe UI"}, "sizes": {"title_size": 46, "subtitle_size": 24, "heading_size": 36, "body_size": 20, "bullet_size": 18}}',
            true,
            true,
            true
        ),
        (
            gen_random_uuid(),
            'Minimal',
            'Simple and clean with subtle grays',
            'theme',
            '{"colors": {"title_color": "#000000", "accent_color": "#95a5a6", "text_color": "#2c3e50", "bg_color": "#ffffff"}, "fonts": {"title_font": "Helvetica", "body_font": "Helvetica"}, "sizes": {"title_size": 42, "subtitle_size": 22, "heading_size": 34, "body_size": 18, "bullet_size": 16}}',
            true,
            true,
            true
        ),
        (
            gen_random_uuid(),
            'Corporate',
            'Traditional business style with orange accents',
            'theme',
            '{"colors": {"title_color": "#003d7a", "accent_color": "#f5a623", "text_color": "#333333", "bg_color": "#ffffff"}, "fonts": {"title_font": "Arial", "body_font": "Arial"}, "sizes": {"title_size": 44, "subtitle_size": 24, "heading_size": 36, "body_size": 20, "bullet_size": 18}}',
            true,
            true,
            true
        ),
        (
            gen_random_uuid(),
            'Creative',
            'Artistic with purple and turquoise',
            'theme',
            '{"colors": {"title_color": "#9b59b6", "accent_color": "#1abc9c", "text_color": "#2c3e50", "bg_color": "#fdfbf7"}, "fonts": {"title_font": "Georgia", "body_font": "Verdana"}, "sizes": {"title_size": 48, "subtitle_size": 26, "heading_size": 38, "body_size": 20, "bullet_size": 18}}',
            true,
            true,
            true
        ),
        (
            gen_random_uuid(),
            'Dark',
            'Dark theme for low-light presentations',
            'theme',
            '{"colors": {"title_color": "#ffffff", "accent_color": "#3db9d3", "text_color": "#e0e0e0", "bg_color": "#1e1e2e"}, "fonts": {"title_font": "Segoe UI", "body_font": "Segoe UI"}, "sizes": {"title_size": 44, "subtitle_size": 24, "heading_size": 36, "body_size": 20, "bullet_size": 18}}',
            true,
            true,
            true
        );
    """)


def downgrade() -> None:
    # Drop foreign key and index
    op.drop_index("ix_presentation_jobs_template_id", table_name="presentation_jobs")
    op.drop_constraint("fk_presentation_jobs_template_id", "presentation_jobs", type_="foreignkey")

    # Drop columns from presentation_jobs
    op.drop_column("presentation_jobs", "custom_theme")
    op.drop_column("presentation_jobs", "template_id")

    # Drop indexes
    op.drop_index("ix_presentation_templates_is_public", table_name="presentation_templates")
    op.drop_index("ix_presentation_templates_is_system", table_name="presentation_templates")
    op.drop_index("ix_presentation_templates_user_id", table_name="presentation_templates")

    # Drop table
    op.drop_table("presentation_templates")
