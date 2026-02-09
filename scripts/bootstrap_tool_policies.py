#!/usr/bin/env python3
"""
Bootstrap recommended tool policies.

This script seeds a baseline policy set that matches the platform semantics:
- allow-by-default
- explicit denies block tools
- approvals can be required by allow policies

It writes directly to the database using backend models/settings.
Run it from the repo root (or anywhere) with DATABASE_URL set.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, Optional
from uuid import UUID


def _add_backend_to_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(here, "..", "backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)


def _parse_allowed_domains(v: Optional[str]) -> Optional[list[str]]:
    if not v:
        return None
    out: list[str] = []
    for part in v.split(","):
        dom = part.strip().lower().rstrip(".")
        if dom:
            out.append(dom)
    return out or None


async def _ensure_policy(
    *,
    db,
    subject_type: str,
    subject_id: Optional[UUID],
    subject_key: Optional[str],
    tool_name: str,
    effect: str,
    require_approval: bool,
    constraints: Optional[Dict[str, Any]],
    dry_run: bool,
) -> bool:
    from sqlalchemy import and_, select

    from app.models.tool_policy import ToolPolicy

    stmt = select(ToolPolicy).where(
        and_(
            ToolPolicy.subject_type == subject_type,
            ToolPolicy.subject_id == subject_id,
            ToolPolicy.subject_key == subject_key,
            ToolPolicy.tool_name == tool_name,
            ToolPolicy.effect == effect,
        )
    )
    res = await db.execute(stmt)
    existing = res.scalars().first()
    if existing:
        return False

    if dry_run:
        return True

    pol = ToolPolicy(
        subject_type=subject_type,
        subject_id=subject_id,
        subject_key=subject_key,
        tool_name=tool_name,
        effect=effect,
        require_approval=bool(require_approval),
        constraints=constraints,
    )
    db.add(pol)
    return True


async def main_async() -> int:
    parser = argparse.ArgumentParser(description="Seed recommended tool policies into the database.")
    parser.add_argument(
        "--subject-type",
        default="global",
        choices=["global", "role", "user", "agent_definition", "api_key"],
        help="Policy subject scope (default: global).",
    )
    parser.add_argument("--subject-id", default=None, help="UUID for user/agent_definition/api_key subjects.")
    parser.add_argument("--subject-key", default=None, help="Key for role subjects (e.g., 'admin').")
    parser.add_argument(
        "--allowed-domains",
        default=None,
        help="Comma-separated hostname suffix allowlist applied to network tools (example: wiki.company.com,github.com).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without writing to DB.")
    args = parser.parse_args()

    _add_backend_to_path()

    subject_type = str(args.subject_type).strip().lower()
    subject_id = UUID(args.subject_id) if args.subject_id else None
    subject_key = str(args.subject_key).strip() if args.subject_key else None
    allowed_domains = _parse_allowed_domains(args.allowed_domains)

    # Import after sys.path is set.
    from app.core.database import AsyncSessionLocal

    base_net_constraints: Dict[str, Any] = {"deny_private_networks": True}
    if allowed_domains:
        base_net_constraints["allowed_domains"] = allowed_domains

    # Baseline: require approval for any custom user tool invocation.
    policies: list[dict] = [
        dict(
            tool_name="user_tool:*",
            effect="allow",
            require_approval=True,
            constraints=None,
        ),
        # Network tools (agent + MCP)
        dict(tool_name="web_scrape", effect="allow", require_approval=True, constraints=base_net_constraints),
        dict(tool_name="mcp:web_scrape", effect="allow", require_approval=True, constraints=base_net_constraints),
        dict(tool_name="ingest_url", effect="allow", require_approval=True, constraints=base_net_constraints),
        dict(tool_name="mcp:ingest_url", effect="allow", require_approval=True, constraints=base_net_constraints),
        dict(tool_name="mcp:create_repo_report", effect="allow", require_approval=True, constraints=base_net_constraints),
        # High-impact write tools (require human approval)
        dict(tool_name="delete_document", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="batch_delete_documents", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="update_document_tags", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="create_document_from_text", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="merge_entities", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="delete_entity", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="rebuild_document_knowledge_graph", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="run_workflow", effect="allow", require_approval=True, constraints=None),
        dict(tool_name="run_custom_tool", effect="allow", require_approval=True, constraints=None),
    ]

    created = 0
    async with AsyncSessionLocal() as db:
        for p in policies:
            did_create = await _ensure_policy(
                db=db,
                subject_type=subject_type,
                subject_id=subject_id,
                subject_key=subject_key,
                tool_name=p["tool_name"],
                effect=p["effect"],
                require_approval=bool(p["require_approval"]),
                constraints=p["constraints"],
                dry_run=bool(args.dry_run),
            )
            if did_create:
                created += 1

        if args.dry_run:
            print(f"[dry-run] Would create {created} policies (skipping existing).")
            return 0

        await db.commit()

    print(f"Created {created} tool policies (skipping existing).")
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

