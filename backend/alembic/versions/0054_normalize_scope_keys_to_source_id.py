"""Normalize legacy scope key target_source_id to source_id in JSON configs.

Revision ID: 0054_normalize_scope_keys_to_source_id
Revises: 0053_add_ldap_auth_fields
Create Date: 2026-02-24 00:00:00.000000
"""

from __future__ import annotations

from typing import Any, Tuple

from alembic import op
import sqlalchemy as sa


revision = "0054_normalize_scope_keys_to_source_id"
down_revision = "0053_add_ldap_auth_fields"
branch_labels = None
depends_on = None


def _normalize_scope_keys(value: Any) -> Tuple[Any, bool]:
    """Recursively replace target_source_id with source_id (if source_id is missing)."""
    if isinstance(value, list):
        changed = False
        out: list[Any] = []
        for item in value:
            norm, item_changed = _normalize_scope_keys(item)
            out.append(norm)
            changed = changed or item_changed
        return out, changed

    if isinstance(value, dict):
        changed = False
        out: dict[str, Any] = {}
        for k, v in value.items():
            norm_v, v_changed = _normalize_scope_keys(v)
            out[str(k)] = norm_v
            changed = changed or v_changed

        source = str(out.get("source_id") or "").strip()
        target = str(out.get("target_source_id") or "").strip()
        if not source and target:
            out["source_id"] = target
            changed = True
        if "target_source_id" in out:
            out.pop("target_source_id", None)
            changed = True
        return out, changed

    return value, False


def _normalize_json_column(
    bind,
    *,
    table_name: str,
    id_column: str,
    json_column: str,
) -> int:
    """Normalize a JSON column in-place; returns number of rows updated."""
    table = sa.table(
        table_name,
        sa.column(id_column),
        sa.column(json_column),
    )
    rows = bind.execute(
        sa.select(getattr(table.c, id_column), getattr(table.c, json_column)).where(
            getattr(table.c, json_column).isnot(None)
        )
    ).fetchall()

    updated = 0
    for row_id, payload in rows:
        normalized, changed = _normalize_scope_keys(payload)
        if not changed:
            continue
        bind.execute(
            sa.update(table)
            .where(getattr(table.c, id_column) == row_id)
            .values({json_column: normalized})
        )
        updated += 1
    return updated


def upgrade() -> None:
    bind = op.get_bind()

    # Autonomous jobs: config + chain definitions stored on jobs.
    _normalize_json_column(bind, table_name="agent_jobs", id_column="id", json_column="config")
    _normalize_json_column(bind, table_name="agent_jobs", id_column="id", json_column="chain_config")

    # Job templates used for new jobs.
    _normalize_json_column(bind, table_name="agent_job_templates", id_column="id", json_column="default_config")

    # Chain definitions used for multi-step runs.
    _normalize_json_column(bind, table_name="agent_job_chain_definitions", id_column="id", json_column="default_settings")
    _normalize_json_column(bind, table_name="agent_job_chain_definitions", id_column="id", json_column="chain_steps")


def downgrade() -> None:
    # Data-only migration; intentionally no-op.
    pass

