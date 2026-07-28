"""Deterministic planning for unresolved autonomous R&D evidence."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Dict, List, Mapping


class AutonomousRnDVerificationPlanner:
    """Create bounded local verification tasks without launching them."""

    _MAX_TASKS = 50

    def build_plan(self, outcome: Mapping[str, Any]) -> Dict[str, Any]:
        planned = deepcopy(dict(outcome))
        evidence = _records(planned.get("evidence"))
        claims = _records(planned.get("claims"))
        links = _records(planned.get("verification_links"))
        experiment = _mapping(planned.get("experiment"))
        verification_experiments = _records(planned.get("verification_experiments"))
        claim_refs = {
            evidence_id
            for claim in claims
            for evidence_id in _string_list(claim.get("evidence_ids"))
        }
        links_by_evidence: Dict[str, List[Dict[str, Any]]] = {}
        for link in links:
            evidence_id = str(
                link.get("external_evidence_id") or link.get("evidence_id") or ""
            ).strip()
            if evidence_id:
                links_by_evidence.setdefault(evidence_id, []).append(link)

        tasks = []
        for item in evidence:
            if str(item.get("kind") or "").strip() not in {
                "external_agent_response",
                "external_system_response",
            }:
                continue
            evidence_id = str(item.get("id") or "").strip()
            status = (
                str(item.get("verification_status") or "unverified").strip().lower()
            )
            if not evidence_id or status not in {"unverified", "corroborated"}:
                continue
            tasks.append(
                self._build_task(
                    item=item,
                    status=status,
                    claim_referenced=evidence_id in claim_refs,
                    links=links_by_evidence.get(evidence_id, []),
                    experiment=experiment,
                    verification_experiments=verification_experiments,
                )
            )

        tasks.sort(
            key=lambda task: (
                -int(task["priority_score"]),
                str(task["evidence_id"]),
            )
        )
        unresolved_count = len(tasks)
        tasks = tasks[: self._MAX_TASKS]
        planned["verification_plan"] = {
            "policy": "bounded_local_verification_v1",
            "launch_mode": "proposal_only",
            "task_count": len(tasks),
            "unresolved_external_evidence_count": unresolved_count,
            "truncated": unresolved_count > len(tasks),
            "autolaunch_eligible_count": 0,
            "tasks": tasks,
            "next_action": (
                "Review and launch the highest-priority local verification task."
                if tasks
                else "No unresolved external evidence requires verification."
            ),
        }
        return planned

    def _build_task(
        self,
        *,
        item: Dict[str, Any],
        status: str,
        claim_referenced: bool,
        links: List[Dict[str, Any]],
        experiment: Dict[str, Any],
        verification_experiments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        evidence_id = str(item.get("id") or "").strip()
        capability = str(item.get("capability") or "general").strip() or "general"
        min_repeat_count = max(
            [_positive_int(link.get("min_repeat_count"), 2) for link in links] or [2]
        )
        required_artifact_kinds = sorted(
            {
                kind
                for link in links
                for kind in _string_list(link.get("artifact_kinds"))
            }
        )
        if not required_artifact_kinds:
            required_artifact_kinds = self._default_artifacts(capability)

        has_local_evidence = bool(item.get("verified_by_evidence_ids"))
        has_artifact = bool(
            item.get("verified_by_artifact_ids")
            or item.get("verified_by_artifact_kinds")
        )
        linked_run_ids = {
            str(link.get("experiment_run_id") or "").strip()
            for link in links
            if str(link.get("experiment_run_id") or "").strip()
        }
        candidate_experiments = (
            [
                row
                for row in verification_experiments
                if str(row.get("run_id") or row.get("id") or "").strip()
                in linked_run_ids
            ]
            if linked_run_ids
            else [experiment]
        )
        experiment_ready = any(
            row.get("ran") is True
            and row.get("all_commands_ok") is True
            and _positive_int(row.get("repeat_count"), 0) >= min_repeat_count
            for row in candidate_experiments
        )
        required_checks = []
        if not has_local_evidence:
            required_checks.append("collect_independent_local_evidence")
        if not has_artifact:
            required_checks.append("capture_replayable_artifacts")
        if not experiment_ready:
            required_checks.append("run_repeated_controlled_experiment")

        priority_score = 90 if claim_referenced else 60
        if status == "corroborated":
            priority_score += 10
        task_hash = hashlib.sha256(evidence_id.encode("utf-8")).hexdigest()[:12]
        return {
            "id": f"verify-{task_hash}",
            "evidence_id": evidence_id,
            "current_status": status,
            "priority": "critical" if claim_referenced else "standard",
            "priority_score": priority_score,
            "capability": capability,
            "objective": (
                "Independently verify or reject the external-agent response using "
                "locally observed, replayable evidence."
            ),
            "required_checks": required_checks,
            "experiment_spec": {
                "execution_environment": "local_sandbox",
                "repeat_count": min_repeat_count,
                "required_artifact_kinds": required_artifact_kinds,
                "external_agents_allowed": False,
                "success_criteria": [
                    "all_commands_ok",
                    "independent_runtime_evidence_recorded",
                    "replayable_artifacts_captured",
                    "explicit_support_or_contradiction_link_recorded",
                ],
            },
            "stop_conditions": [
                "verification_status becomes verified",
                "verification_status becomes rejected",
                "experiment reaches configured resource limit",
            ],
            "autolaunch_eligible": False,
            "approval_required": True,
        }

    @staticmethod
    def _default_artifacts(capability: str) -> List[str]:
        normalized = capability.lower()
        if "compiler" in normalized:
            return [
                "benchmark_output",
                "compiler_logs",
                "ir_or_codegen_artifacts",
            ]
        if "retrieval" in normalized or "search" in normalized:
            return ["evaluation_metrics", "retrieval_trace"]
        return ["experiment_log", "local_observation"]


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _records(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


autonomous_rnd_verification_planner = AutonomousRnDVerificationPlanner()
