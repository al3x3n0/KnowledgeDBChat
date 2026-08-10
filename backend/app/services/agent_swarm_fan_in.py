"""Deterministic swarm fan-in merging.

Pure helpers extracted from AutonomousAgentExecutor: they take plain payload
dicts and return plain dicts, so they are covered directly by unit tests rather
than through the executor.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.agent_job import AgentJobStatus


def normalize_role_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        return ""

    alias_map = {
        "research": "researcher",
        "researcher": "researcher",
        "researcher_documents": "researcher",
        "researcher_docs": "researcher",
        "researcher_arxiv": "researcher",
        "knowledge_researcher": "researcher",
        "document_researcher": "researcher",
        "docs_researcher": "researcher",
        "literature_researcher": "researcher",
        "paper_researcher": "researcher",
        "arxiv_researcher": "researcher",
        "critic": "critic",
        "reviewer": "critic",
        "analyst": "critic",
        "synth": "synthesizer",
        "synthesizer": "synthesizer",
        "writer": "synthesizer",
        "aggregator": "synthesizer",
        "verify": "verifier",
        "verifier": "verifier",
        "validator": "verifier",
        "qa": "verifier",
        "monitor": "verifier",
        "reproducer": "verifier",
        "repro": "verifier",
        "root_cause": "critic",
        "rootcause": "critic",
        "patcher": "coder",
        "repairer": "coder",
        "implementer": "coder",
        "primary_implementer": "coder",
    }
    if token in alias_map:
        return alias_map[token]

    parts = [p for p in token.split("_") if p]
    if "analyst" in parts or "critic" in parts or "reviewer" in parts:
        return "critic"
    if (
        any(p.startswith("synth") for p in parts)
        or "aggregator" in parts
        or "writer" in parts
    ):
        return "synthesizer"
    if (
        "monitor" in parts
        or "qa" in parts
        or any(p.startswith("verif") for p in parts)
        or any(p.startswith("validat") for p in parts)
        or any(p.startswith("repro") for p in parts)
    ):
        return "verifier"
    if "root" in parts or "cause" in parts:
        return "critic"
    if (
        "researcher" in parts
        or "research" in parts
        or "arxiv" in parts
        or "literature" in parts
    ):
        return "researcher"
    if "patch" in parts or "repair" in parts or "implementer" in parts:
        return "coder"
    return token


def build_swarm_fan_in_result(
    payload: Dict[str, Any],
    *,
    fan_in_group_id: str = "",
) -> Dict[str, Any]:
    """Build deterministic merged result from swarm sibling outputs."""

    def _norm_text(text: Any) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        return re.sub(r"\s+", " ", raw).strip()

    def _extract_points(results: Dict[str, Any]) -> List[str]:
        points: List[str] = []
        if not isinstance(results, dict):
            return points

        findings = results.get("findings")
        if isinstance(findings, list):
            for row in findings:
                text = ""
                if isinstance(row, dict):
                    text = str(
                        row.get("title")
                        or row.get("summary")
                        or row.get("message")
                        or row.get("insight")
                        or row.get("content")
                        or ""
                    ).strip()
                else:
                    text = str(row or "").strip()
                text = _norm_text(text)
                if text:
                    points.append(text[:280])

        research = results.get("research")
        if isinstance(research, dict):
            for key in ("top_insights", "top_documents", "top_papers"):
                items = research.get(key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    text = _norm_text(item)
                    if text:
                        points.append(text[:280])

        summary = _norm_text(results.get("summary"))
        if summary:
            points.append(summary[:280])

        seen: set[str] = set()
        deduped: List[str] = []
        for point in points:
            k = point.lower()
            if not point or k in seen:
                continue
            seen.add(k)
            deduped.append(point)
            if len(deduped) >= 12:
                break
        return deduped

    def _extract_paths(results: Dict[str, Any]) -> List[str]:
        if not isinstance(results, dict):
            return []
        buckets: List[Any] = []
        for key in (
            "file_paths",
            "suspect_files",
            "touched_files",
            "modified_files",
            "changed_files",
            "impacted_files",
        ):
            value = results.get(key)
            if isinstance(value, list):
                buckets.extend(value)
        code_exec = (
            results.get("code_patch_execution")
            if isinstance(results.get("code_patch_execution"), dict)
            else {}
        )
        workspace = (
            code_exec.get("workspace")
            if isinstance(code_exec.get("workspace"), dict)
            else {}
        )
        for key in ("modified_files", "changed_files", "added_files"):
            value = workspace.get(key)
            if isinstance(value, list):
                buckets.extend(value)
        out: List[str] = []
        seen: set[str] = set()
        for raw in buckets:
            path = str(raw or "").replace("\\", "/").strip().lstrip("/")
            while path.startswith("./"):
                path = path[2:]
            if not path or ":" in path:
                continue
            parts = [seg for seg in path.split("/") if seg not in {"", ".", ".."}]
            normalized = "/".join(parts)[:500]
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            out.append(normalized)
            if len(out) >= 12:
                break
        return out

    def _extract_commands(results: Dict[str, Any]) -> List[str]:
        if not isinstance(results, dict):
            return []
        buckets: List[Any] = []
        for key in ("commands", "verification_commands"):
            value = results.get(key)
            if isinstance(value, list):
                buckets.extend(value)
        experiment = (
            results.get("experiment_run")
            if isinstance(results.get("experiment_run"), dict)
            else {}
        )
        for key in ("verification_commands", "commands", "failed_commands"):
            value = experiment.get(key)
            if isinstance(value, list):
                buckets.extend(value)
        out: List[str] = []
        seen: set[str] = set()
        for raw in buckets:
            command = _norm_text(raw)[:500]
            key = command.lower()
            if not command or key in seen:
                continue
            seen.add(key)
            out.append(command)
            if len(out) >= 8:
                break
        return out

    def _path_cluster_keys(path: str) -> List[str]:
        normalized = str(path or "").replace("\\", "/").strip().strip("/")
        if not normalized:
            return []
        parts = [
            segment
            for segment in normalized.split("/")
            if segment not in {"", ".", ".."}
        ]
        if not parts:
            return []
        keys: List[str] = ["/".join(parts).lower()]
        if len(parts) >= 2:
            keys.append("/".join(parts[-2:]).lower())
        keys.append(parts[-1].lower())
        return [key for idx, key in enumerate(keys) if key and key not in keys[:idx]]

    def _path_cluster_label(path: str) -> str:
        normalized = str(path or "").replace("\\", "/").strip().strip("/")
        if not normalized:
            return ""
        parts = [
            segment
            for segment in normalized.split("/")
            if segment not in {"", ".", ".."}
        ]
        if not parts:
            return ""
        return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    def _canonical_command(command: str) -> str:
        lowered = _norm_text(command).lower()
        if not lowered:
            return ""
        lowered = re.sub(
            r"^\s*(ci=true|node_env=\S+|pythonunbuffered=\S+)\s+", "", lowered
        )
        return lowered.strip()

    sibling_jobs = payload.get("sibling_jobs")
    if not isinstance(sibling_jobs, list):
        sibling_jobs = []
    coding_swarm_enabled = bool(payload.get("coding_swarm_enabled")) or (
        str(payload.get("coding_swarm_profile") or "").strip().lower() == "bug_triage"
    )
    coding_harness_enabled = bool(payload.get("coding_harness_enabled"))
    fallback_paths = (
        [str(p).strip() for p in (payload.get("file_paths") or []) if str(p).strip()]
        if isinstance(payload.get("file_paths"), list)
        else []
    )
    fallback_commands = (
        [str(c).strip() for c in (payload.get("commands") or []) if str(c).strip()]
        if isinstance(payload.get("commands"), list)
        else []
    )
    confidence_threshold = float(
        payload.get("coding_swarm_confidence_threshold") or 0.70
    )
    tiebreaker_threshold = float(
        payload.get("coding_swarm_tiebreaker_threshold") or 0.50
    )
    expected = int(payload.get("expected_siblings", 0) or 0)
    if expected <= 0:
        expected = len(sibling_jobs)
    terminal_count = int(payload.get("terminal_siblings", 0) or 0)
    if terminal_count <= 0:
        terminal_statuses = {
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }
        terminal_count = len(
            [
                s
                for s in sibling_jobs
                if str((s or {}).get("status") or "") in terminal_statuses
            ]
        )

    support_map: Dict[str, Dict[str, Any]] = {}
    role_summaries: List[Dict[str, Any]] = []
    sibling_status: List[Dict[str, Any]] = []
    roles_ordered: List[str] = []
    completed_count = 0
    failed_roles: List[str] = []
    ranked_candidates: List[Dict[str, Any]] = []
    winning_candidate: Optional[Dict[str, Any]] = None
    role_file_hints: List[Dict[str, Any]] = []
    role_command_hints: List[Dict[str, Any]] = []

    for row in sibling_jobs:
        if not isinstance(row, dict):
            continue
        role = _norm_text(row.get("role") or row.get("name") or "unknown_role")[:120]
        status = _norm_text(row.get("status") or "unknown").lower()
        normalized_role = normalize_role_token(role)
        if role and role not in roles_ordered:
            roles_ordered.append(role)
        if status == AgentJobStatus.COMPLETED.value:
            completed_count += 1
        if status in {AgentJobStatus.FAILED.value, AgentJobStatus.CANCELLED.value}:
            failed_roles.append(role or "unknown_role")

        row_results = row.get("results") if isinstance(row.get("results"), dict) else {}
        points = _extract_points(row_results)
        candidate_paths = _extract_paths(row_results)
        candidate_commands = _extract_commands(row_results)
        role_summaries.append(
            {
                "role": role,
                "status": status,
                "key_points": points[:3],
            }
        )
        sibling_status.append(
            {
                "job_id": str(row.get("job_id") or ""),
                "role": role,
                "status": status,
                "progress": int(row.get("progress", 0) or 0),
            }
        )
        if coding_swarm_enabled:
            harness_evidence = (
                row_results.get("coding_harness")
                if isinstance(row_results.get("coding_harness"), dict)
                else {}
            )
            verification_eligible = bool(
                harness_evidence.get("completion_eligible", False)
            )
            candidate_snapshot = (
                harness_evidence.get("candidate_snapshot")
                if isinstance(harness_evidence.get("candidate_snapshot"), dict)
                else None
            )
            promotion_eligible = not coding_harness_enabled or (
                normalized_role == "coder"
                and verification_eligible
                and bool(candidate_snapshot)
            )
            role_bonus = {
                "coder": 0.18,
                "critic": 0.14,
                "verifier": 0.16,
            }.get(normalized_role, 0.08)
            completion_bonus = 0.25 if status == AgentJobStatus.COMPLETED.value else 0.0
            score = (
                completion_bonus
                + role_bonus
                + min(0.25, len(points) * 0.04)
                + min(0.16, len(candidate_paths) * 0.04)
                + (0.18 if coding_harness_enabled and verification_eligible else 0.0)
            )
            candidate = {
                "job_id": str(row.get("job_id") or ""),
                "role": role,
                "normalized_role": normalized_role,
                "status": status,
                "score": round(score, 4),
                "suspect_files": candidate_paths[:8],
                "recommended_commands": candidate_commands[:6],
                "verification_eligible": verification_eligible,
                "verification_state": str(
                    harness_evidence.get("verification_state") or "unknown"
                ),
                "modified_files": (
                    harness_evidence.get("modified_files")
                    if isinstance(harness_evidence.get("modified_files"), list)
                    else []
                )[:12],
                "candidate_snapshot": deepcopy(candidate_snapshot),
            }
            ranked_candidates.append(candidate)
            if promotion_eligible:
                if winning_candidate is None or float(
                    candidate.get("score") or 0.0
                ) > float(winning_candidate.get("score") or 0.0):
                    winning_candidate = candidate
            if candidate_paths:
                role_file_hints.append(
                    {
                        "role": role,
                        "paths": candidate_paths[:8],
                    }
                )
            if candidate_commands:
                role_command_hints.append(
                    {
                        "role": role,
                        "commands": candidate_commands[:6],
                    }
                )

        used_keys: set[str] = set()
        for point in points:
            k = point.lower()
            if not k or k in used_keys:
                continue
            used_keys.add(k)
            slot = support_map.get(k)
            if not isinstance(slot, dict):
                slot = {"finding": point, "roles": set(), "count": 0}
            roles_set = slot.get("roles")
            if not isinstance(roles_set, set):
                roles_set = set()
            roles_set.add(role)
            slot["roles"] = roles_set
            slot["count"] = int(slot.get("count", 0) or 0) + 1
            support_map[k] = slot

    support_rows: List[Dict[str, Any]] = []
    for k, slot in support_map.items():
        roles = sorted([str(r) for r in slot.get("roles", set()) if str(r).strip()])
        support_rows.append(
            {
                "key": k,
                "finding": str(slot.get("finding") or ""),
                "support_count": int(slot.get("count", 0) or 0),
                "supporting_roles": roles,
            }
        )
    support_rows.sort(
        key=lambda r: (
            -int(r.get("support_count", 0) or 0),
            str(r.get("finding") or ""),
        )
    )

    consensus = [r for r in support_rows if int(r.get("support_count", 0) or 0) >= 2][
        :10
    ]
    singletons = [r for r in support_rows if int(r.get("support_count", 0) or 0) <= 1][
        :10
    ]

    conflicts: List[Dict[str, Any]] = []
    if failed_roles and completed_count > 0:
        conflicts.append(
            {
                "type": "execution_divergence",
                "description": f"{len(failed_roles)} swarm role(s) failed or were cancelled while others completed.",
                "roles": failed_roles[:8],
            }
        )
    if not consensus and len(roles_ordered) >= 2 and support_rows:
        conflicts.append(
            {
                "type": "low_alignment",
                "description": "Role outputs show low overlap; no repeated findings across roles.",
                "roles": roles_ordered[:8],
            }
        )
    if terminal_count < expected:
        conflicts.append(
            {
                "type": "incomplete_swarm",
                "description": f"Only {terminal_count}/{expected} sibling jobs reached a terminal state.",
                "roles": roles_ordered[:8],
            }
        )

    file_cluster_support: Dict[str, Dict[str, Any]] = {}
    for row in role_file_hints:
        role = str(row.get("role") or "").strip() or "unknown_role"
        seen_role_clusters: set[str] = set()
        for raw_path in row.get("paths") or []:
            path = str(raw_path or "").strip()
            if not path:
                continue
            label = _path_cluster_label(path)
            for cluster_key in _path_cluster_keys(path):
                if cluster_key in seen_role_clusters:
                    continue
                seen_role_clusters.add(cluster_key)
                slot = file_cluster_support.get(cluster_key)
                if not isinstance(slot, dict):
                    slot = {
                        "cluster": label or cluster_key,
                        "roles": set(),
                        "support_count": 0,
                    }
                roles_set = slot.get("roles")
                if not isinstance(roles_set, set):
                    roles_set = set()
                roles_set.add(role)
                slot["roles"] = roles_set
                slot["support_count"] = len(roles_set)
                file_cluster_support[cluster_key] = slot

    command_support: Dict[str, Dict[str, Any]] = {}
    for row in role_command_hints:
        role = str(row.get("role") or "").strip() or "unknown_role"
        seen_role_commands: set[str] = set()
        for raw_command in row.get("commands") or []:
            command = str(raw_command or "").strip()
            canonical = _canonical_command(command)
            if not canonical or canonical in seen_role_commands:
                continue
            seen_role_commands.add(canonical)
            slot = command_support.get(canonical)
            if not isinstance(slot, dict):
                slot = {"command": command, "roles": set(), "support_count": 0}
            roles_set = slot.get("roles")
            if not isinstance(roles_set, set):
                roles_set = set()
            roles_set.add(role)
            slot["roles"] = roles_set
            slot["support_count"] = len(roles_set)
            command_support[canonical] = slot

    top_file_cluster = None
    if file_cluster_support:
        top_file_cluster = max(
            file_cluster_support.values(),
            key=lambda item: (
                int(item.get("support_count") or 0),
                str(item.get("cluster") or ""),
            ),
        )
    top_command_cluster = None
    if command_support:
        top_command_cluster = max(
            command_support.values(),
            key=lambda item: (
                int(item.get("support_count") or 0),
                str(item.get("command") or ""),
            ),
        )

    file_convergence_support = int((top_file_cluster or {}).get("support_count") or 0)
    command_convergence_support = int(
        (top_command_cluster or {}).get("support_count") or 0
    )
    file_converged = file_convergence_support >= 2
    command_converged = command_convergence_support >= 2
    if coding_swarm_enabled and role_file_hints and not file_converged:
        conflicts.append(
            {
                "type": "suspect_file_disagreement",
                "description": "Roles disagree on the primary suspect file cluster.",
                "roles": [
                    str(row.get("role") or "")
                    for row in role_file_hints[:8]
                    if str(row.get("role") or "").strip()
                ],
            }
        )
    if coding_swarm_enabled and role_command_hints and not command_converged:
        conflicts.append(
            {
                "type": "command_disagreement",
                "description": "Roles disagree on the strongest reproduction or verification command.",
                "roles": [
                    str(row.get("role") or "")
                    for row in role_command_hints[:8]
                    if str(row.get("role") or "").strip()
                ],
            }
        )

    coverage = float(min(1.0, float(len(sibling_jobs)) / float(max(1, expected))))
    completion = float(
        min(1.0, float(completed_count) / float(max(1, len(sibling_jobs))))
    )
    agreement = 0.0
    if consensus:
        agreement = float(
            sum(
                min(
                    1.0,
                    float(int(r.get("support_count", 0) or 0))
                    / float(max(1, len(sibling_jobs))),
                )
                for r in consensus
            )
        )
        agreement = max(0.0, min(1.0, agreement / float(max(1, len(consensus)))))
    overall = max(
        0.0, min(1.0, (0.35 * coverage) + (0.35 * completion) + (0.3 * agreement))
    )

    action_plan: List[Dict[str, Any]] = []
    for row in consensus[:3]:
        action_plan.append(
            {
                "priority": "high",
                "action": f"Validate and operationalize: {str(row.get('finding') or '')[:200]}",
                "rationale": f"Supported by {int(row.get('support_count', 0) or 0)} swarm roles.",
            }
        )
    for conflict in conflicts[:2]:
        action_plan.append(
            {
                "priority": "medium",
                "action": f"Resolve conflict: {str(conflict.get('type') or 'conflict')}",
                "rationale": str(conflict.get("description") or "")[:220],
            }
        )
    promotion_reason = ""
    recommended_commands: List[str] = []
    candidate_paths: List[Any] = []
    review_state = "informational"
    review_reason = ""
    review_required = False
    tiebreaker_attempted = bool(payload.get("tie_breaker_attempted"))
    tie_breaker_job_id = str(payload.get("tie_breaker_job_id") or "").strip()
    tie_breaker_source_job_id = str(
        payload.get("tie_breaker_source_job_id") or ""
    ).strip()
    if coding_swarm_enabled:
        ranked_candidates.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                str(item.get("role") or ""),
            )
        )
        candidate_paths = ranked_candidates[:6]
        if winning_candidate:
            recommended_commands.extend(
                [
                    str(cmd).strip()
                    for cmd in (winning_candidate.get("recommended_commands") or [])
                    if str(cmd).strip()
                ]
            )
        recommended_commands.extend(fallback_commands)
        dedup_commands: List[str] = []
        seen_commands: set[str] = set()
        for command in recommended_commands:
            key = command.lower()
            if not command or key in seen_commands:
                continue
            seen_commands.add(key)
            dedup_commands.append(command)
            if len(dedup_commands) >= 8:
                break
        recommended_commands = dedup_commands

        if not candidate_paths and fallback_paths:
            candidate_paths = [
                {
                    "job_id": "",
                    "role": "Config scope",
                    "status": "configured",
                    "score": 0.0,
                    "suspect_files": fallback_paths[:8],
                }
            ]
        verification_guardrail_met = bool(
            winning_candidate
            and (
                not coding_harness_enabled
                or (
                    winning_candidate.get("verification_eligible")
                    and winning_candidate.get("candidate_snapshot")
                )
            )
        )
        guardrails_met = (
            bool(winning_candidate)
            and file_converged
            and command_converged
            and verification_guardrail_met
        )
        if coding_harness_enabled and not winning_candidate:
            review_state = "needs_review"
            review_required = True
            review_reason = (
                "No mutation-owner result included both changed files and "
                "successful verification evidence."
            )
            promotion_reason = review_reason
        elif overall >= confidence_threshold and winning_candidate and guardrails_met:
            promotion_reason = (
                f"Auto-promote {str(winning_candidate.get('role') or 'top candidate')} at swarm confidence "
                f"{overall:.2f}."
            )
            review_state = "auto_promoted"
            action_plan.insert(
                0,
                {
                    "priority": "high",
                    "action": f"Auto-promote winning coding slice: {str(winning_candidate.get('role') or '')}",
                    "rationale": promotion_reason,
                },
            )
        elif (
            overall >= confidence_threshold and winning_candidate and not guardrails_met
        ):
            review_state = "needs_review"
            review_required = True
            review_reason = (
                "Confidence cleared the promotion threshold, but file-cluster and command convergence "
                "did not both meet the promotion guardrail."
            )
            promotion_reason = review_reason
        elif overall >= tiebreaker_threshold:
            if tiebreaker_attempted:
                review_state = "insufficient_swarm_consensus"
                review_required = True
                review_reason = (
                    f"Confidence {overall:.2f} remained below auto-promotion threshold {confidence_threshold:.2f} "
                    "after a verifier tie-break."
                )
                promotion_reason = review_reason
            else:
                review_state = "tie_break_needed"
                review_reason = (
                    f"Confidence {overall:.2f} is below auto-promotion threshold {confidence_threshold:.2f}; "
                    "launch a verifier tie-break before operator review."
                )
                promotion_reason = review_reason
        else:
            review_state = "consensus_failed"
            review_required = True
            review_reason = (
                f"Confidence {overall:.2f} is too low for automatic repair handoff."
            )
            promotion_reason = review_reason
    if not action_plan and singletons:
        for row in singletons[:2]:
            action_plan.append(
                {
                    "priority": "medium",
                    "action": f"Investigate unique signal: {str(row.get('finding') or '')[:180]}",
                    "rationale": "Appears in only one role; needs validation.",
                }
            )
    if len(action_plan) < 3:
        action_plan.append(
            {
                "priority": "low",
                "action": "Produce a consolidated brief with evidence links and clear owner-assigned next steps.",
                "rationale": "Ensures swarm output is actionable for downstream execution.",
            }
        )
    action_plan = action_plan[:6]

    return {
        "swarm_parent_job_id": str(payload.get("swarm_parent_job_id") or ""),
        "fan_in_group_id": str(
            fan_in_group_id or payload.get("swarm_fan_in_group_id") or ""
        ),
        "expected_siblings": int(expected),
        "received_siblings": int(len(sibling_jobs)),
        "terminal_siblings": int(terminal_count),
        "roles": roles_ordered[:20],
        "role_summaries": role_summaries[:20],
        "sibling_status": sibling_status[:20],
        "consensus_findings": [
            {
                "finding": str(r.get("finding") or "")[:280],
                "support_count": int(r.get("support_count", 0) or 0),
                "supporting_roles": r.get("supporting_roles", [])[:10],
            }
            for r in consensus
        ],
        "conflicts": conflicts[:10],
        "confidence": {
            "overall": round(overall, 4),
            "coverage": round(coverage, 4),
            "completion": round(completion, 4),
            "agreement": round(agreement, 4),
        },
        "action_plan": action_plan,
        "winning_slice_id": str((winning_candidate or {}).get("job_id") or ""),
        "winning_role": str((winning_candidate or {}).get("role") or ""),
        "winning_candidate_snapshot": deepcopy(
            (winning_candidate or {}).get("candidate_snapshot")
        ),
        "promotion_reason": promotion_reason,
        "review_state": review_state,
        "review_reason": review_reason,
        "review_required": review_required,
        "coding_harness_enabled": coding_harness_enabled,
        "verification_guardrail_met": bool(
            winning_candidate
            and (
                not coding_harness_enabled
                or (
                    winning_candidate.get("verification_eligible")
                    and winning_candidate.get("candidate_snapshot")
                )
            )
        ),
        "tie_breaker_attempted": tiebreaker_attempted,
        "tie_breaker_job_id": tie_breaker_job_id,
        "tie_breaker_source_job_id": tie_breaker_source_job_id,
        "file_converged": file_converged,
        "file_convergence_support": file_convergence_support,
        "top_file_cluster": (
            {
                "cluster": str(top_file_cluster.get("cluster") or ""),
                "support_count": int(top_file_cluster.get("support_count") or 0),
                "roles": sorted(
                    [
                        str(role)
                        for role in (top_file_cluster.get("roles") or set())
                        if str(role).strip()
                    ]
                )[:10],
            }
            if isinstance(top_file_cluster, dict)
            else None
        ),
        "command_converged": command_converged,
        "command_convergence_support": command_convergence_support,
        "top_command_cluster": (
            {
                "command": str(top_command_cluster.get("command") or ""),
                "support_count": int(top_command_cluster.get("support_count") or 0),
                "roles": sorted(
                    [
                        str(role)
                        for role in (top_command_cluster.get("roles") or set())
                        if str(role).strip()
                    ]
                )[:10],
            }
            if isinstance(top_command_cluster, dict)
            else None
        ),
        "candidate_paths": candidate_paths[:6],
        "recommended_commands": recommended_commands[:8],
        "generated_at": datetime.utcnow().isoformat(),
    }
