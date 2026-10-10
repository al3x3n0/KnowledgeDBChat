"""The jobs a backlog item starts: a repair run for a slice, and the run that
applies a chosen proposal.

Each is chained to a small continuation job so the orchestrator hears when the
child ends. These lived in ``api/endpoints/coding_backlog.py`` and raised
``HTTPException``; they raise ``ActionRefused`` now, which the endpoint maps.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.coding_backlog import CodingBacklogItem
from app.models.document import DocumentSource
from app.modules.coding_backlog.application.backlog_slices import (
    append_manual_promotion_history,
    normalize_str_list,
)
from app.modules.coding_backlog.application.errors import ActionRefused
from app.services.agent_job_templates import (
    REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
    get_builtin_agent_job_template,
)
from app.services.coding_backlog_decomposition import (
    append_artifact_history,
    append_lineage_id,
    append_slice_timeline,
    timeline_entry,
)
from app.services.job_dispatch import enqueue_agent_job


def build_orchestrator_chain_config(
    backlog_item_id: UUID, previous_child_kind: str
) -> dict[str, Any]:
    return {
        "trigger_condition": "on_any_end",
        "inherit_results": True,
        "inherit_config": False,
        "child_jobs": [
            {
                "name": "Coding Backlog — Continue",
                "job_type": "analysis",
                "goal": "Continue backlog orchestration after a child repair/apply run completes.",
                "config": {
                    "deterministic_runner": "coding_backlog_orchestrator",
                    "coding_backlog_item_id": str(backlog_item_id),
                    "coding_backlog_previous_child_kind": str(previous_child_kind or "")
                    .strip()
                    .lower()
                    or "repair",
                },
                "max_iterations": 1,
                "max_tool_calls": 0,
                "max_llm_calls": 0,
                "max_runtime_minutes": 10,
            }
        ],
    }


def attach_terminal_continuation(
    chain_config: Optional[dict], backlog_item_id: UUID, previous_child_kind: str
) -> Optional[dict]:
    if not isinstance(chain_config, dict):
        return None
    updated = deepcopy(chain_config)
    cursor = updated
    while isinstance(cursor.get("child_jobs"), list) and cursor.get("child_jobs"):
        child = cursor["child_jobs"][-1]
        if not isinstance(child, dict):
            break
        if not isinstance(child.get("chain_config"), dict):
            child["chain_config"] = build_orchestrator_chain_config(
                backlog_item_id, previous_child_kind
            )
            return updated
        cursor = child["chain_config"]
    return updated


async def spawn_slice_repair_job(
    item: CodingBacklogItem,
    slice_state: dict[str, Any],
    *,
    db: AsyncSession,
    operator_note: Optional[str] = None,
) -> AgentJob:
    source = await db.get(DocumentSource, item.source_id) if item.source_id else None
    template = get_builtin_agent_job_template(REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID)
    if not template:
        raise ActionRefused(
            "unavailable", detail="Repo bug triage template unavailable"
        )

    symptom = (
        str(item.failure_symptom or "").strip()
        or str(slice_state.get("goal") or "").strip()
        or str(item.portfolio_goal or "").strip()[:4000]
    )
    scope = (
        str(slice_state.get("scope") or item.scope or "auto").strip().lower() or "auto"
    )
    file_paths = normalize_str_list(slice_state.get("file_paths"))
    commands = normalize_str_list(slice_state.get("commands"))
    search_query = (
        str(slice_state.get("search_query") or "").strip()
        or " ".join(
            part
            for part in [
                ("" if scope == "auto" else scope),
                symptom,
                " ".join(file_paths[:2]),
            ]
            if part
        ).strip()[:500]
    )
    merged_config = dict(template.default_config or {})
    merged_config.update(
        {
            "source_id": str(item.source_id),
            "launch_mode": "quick_start_repo_bug_triage",
            "failure_symptom": symptom,
            "scope": scope,
            "search_query": search_query,
            "quick_start": {
                "profile": "repo_bug_triage",
                "version": "v2",
                "source_name": str(getattr(source, "name", "") or ""),
                "source_type": str(getattr(source, "source_type", "") or "")
                .strip()
                .lower(),
                "scope": scope,
                "autonomy_mode": "patch_proposal",
                "execution_depth": "workspace_planned",
            },
            "coding_backlog_item_id": str(item.id),
            "coding_backlog_child_kind": "repair",
            "coding_backlog_goal_type": "portfolio_goal",
            "coding_backlog_slice_id": str(slice_state.get("slice_id") or ""),
            "coding_backlog_slice_title": str(slice_state.get("title") or ""),
        }
    )
    if file_paths:
        merged_config["file_paths"] = file_paths
    if commands:
        merged_config["commands"] = commands
    if str(item.error_output or "").strip():
        merged_config["error_output"] = str(item.error_output).strip()[:4000]
    if operator_note:
        merged_config["coding_backlog_operator_note"] = str(operator_note).strip()[
            :5000
        ]

    repair_job = AgentJob(
        name=f"{str(item.title or '').strip()[:120]} — {str(slice_state.get('title') or 'Repair')[:80]}",
        description="Repair slice launched from coding backlog action.",
        job_type=template.job_type,
        goal=str(slice_state.get("goal") or item.portfolio_goal or "").strip()[:8000]
        or template.default_goal,
        config=merged_config,
        user_id=item.user_id,
        status=AgentJobStatus.PENDING.value,
        parent_job_id=item.orchestrator_job_id,
        root_job_id=item.orchestrator_job_id,
        chain_depth=1,
        chain_config=attach_terminal_continuation(
            template.default_chain_config, item.id, "repair"
        ),
        max_iterations=template.default_max_iterations,
        max_tool_calls=template.default_max_tool_calls,
        max_llm_calls=template.default_max_llm_calls,
        max_runtime_minutes=template.default_max_runtime_minutes,
    )
    db.add(repair_job)
    await db.flush()
    prev_status = str(slice_state.get("status") or "").strip() or None
    append_slice_timeline(
        slice_state,
        timeline_entry(
            actor="system",
            action="repair_job_started",
            previous_status=prev_status,
            new_status="repairing",
            note=operator_note,
            related_job_id=str(repair_job.id),
        ),
    )
    append_lineage_id(slice_state, "repair_job_ids", str(repair_job.id))
    if operator_note:
        append_manual_promotion_history(
            slice_state,
            action="relaunch_slice" if prev_status else "repair_job_started",
            operator_note=operator_note,
        )
    enqueue_agent_job(db, repair_job.id, item.user_id)
    return repair_job


async def spawn_slice_apply_job(
    item: CodingBacklogItem,
    slice_state: dict[str, Any],
    *,
    db: AsyncSession,
    proposal_id: str,
    operator_note: Optional[str] = None,
) -> AgentJob:
    apply_job = AgentJob(
        name=f"{str(item.title or '').strip()[:120]} — Apply {str(slice_state.get('title') or 'Patch')[:74]}",
        description="Operator-triggered backlog apply job.",
        job_type="analysis",
        goal="Apply the selected code patch proposal to the knowledge base.",
        config={
            "deterministic_runner": "code_patch_apply_to_kb",
            "proposal_id": proposal_id,
            "proposal_strategy": "explicit",
            "apply_patch_to_kb": True,
            "dry_run": False,
            "require_experiments_ok": True,
            "require_dry_run_first": False,
            "fail_on_block": True,
            "coding_backlog_item_id": str(item.id),
            "coding_backlog_child_kind": "apply",
            "coding_backlog_slice_id": str(slice_state.get("slice_id") or ""),
            "coding_backlog_operator_note": str(operator_note or "").strip()[:5000]
            or None,
        },
        user_id=item.user_id,
        status=AgentJobStatus.PENDING.value,
        parent_job_id=item.orchestrator_job_id,
        root_job_id=item.orchestrator_job_id,
        chain_depth=1,
        chain_config=build_orchestrator_chain_config(item.id, "apply"),
        max_iterations=1,
        max_tool_calls=0,
        max_llm_calls=0,
        max_runtime_minutes=15,
    )
    db.add(apply_job)
    await db.flush()
    append_slice_timeline(
        slice_state,
        timeline_entry(
            actor="user",
            action="apply_override_started",
            previous_status=str(slice_state.get("status") or "").strip() or None,
            new_status="applying",
            note=operator_note,
            related_job_id=str(apply_job.id),
            related_proposal_id=proposal_id,
        ),
    )
    append_lineage_id(slice_state, "apply_job_ids", str(apply_job.id))
    append_lineage_id(slice_state, "proposal_ids", proposal_id)
    append_artifact_history(slice_state, "proposal", proposal_id, "Selected proposal")
    append_manual_promotion_history(
        slice_state,
        action="apply_override",
        operator_note=operator_note,
        proposal_id=proposal_id,
        apply_job_id=str(apply_job.id),
    )
    enqueue_agent_job(db, apply_job.id, item.user_id)
    return apply_job
