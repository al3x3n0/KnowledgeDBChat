"""Project persisted agent trajectories into autonomous R&D evaluation outcomes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional

from app.services.autonomous_rnd_evidence_verification_service import (
    autonomous_rnd_evidence_verifier,
)
from app.services.autonomous_rnd_verification_planner_service import (
    autonomous_rnd_verification_planner,
)


class AutonomousRnDTrajectoryAdapter:
    """Conservatively normalize AgentJob and ExperimentRun records for grading."""

    _EXTERNAL_PROVENANCE_FIELDS = (
        "external_agent_id",
        "external_agent_name",
        "endpoint_origin",
        "provider_type",
        "capability",
        "request_id",
        "remote_references",
        "audit_id",
        "received_at",
        "response_sha256",
        "response_bytes",
        "execution_time_ms",
        "evidence_identity",
        "sync_subscription_id",
    )

    def build_outcome(
        self,
        job: Any,
        *,
        experiment_runs: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        results = _mapping(getattr(job, "results", None))
        canonical = deepcopy(_mapping(results.get("evaluation_outcome")))
        structured = _mapping(results.get("structured_output"))
        runs = list(experiment_runs or [])

        outcome: Dict[str, Any] = canonical
        outcome["schema_version"] = 3
        outcome["generated_by"] = "trajectory_adapter"
        outcome["status"] = str(getattr(job, "status", "") or "").strip().lower()
        outcome["claims"] = self._records(
            canonical.get("claims"), results.get("claims"), structured.get("claims")
        )
        actions = self._actions(
            canonical.get("actions"),
            results,
            getattr(job, "execution_log", None),
        )
        outcome["evidence"] = self._evidence(
            canonical.get("evidence"),
            results.get("evidence"),
            structured.get("evidence"),
            results.get("findings"),
        )
        outcome["evidence"].extend(
            self._external_agent_evidence(actions, outcome["evidence"])
        )
        outcome["artifacts"] = self._artifacts(
            canonical.get("artifacts"),
            getattr(job, "output_artifacts", None),
            results,
            runs,
        )
        outcome["actions"] = actions
        outcome["experiment"] = self._experiment(
            canonical.get("experiment"), results, runs
        )
        outcome["verification_links"] = self._records(
            canonical.get("verification_links"),
            results.get("verification_links"),
            structured.get("verification_links"),
        )
        outcome["verification_experiments"] = self._records(
            canonical.get("verification_experiments"),
            results.get("verification_experiments"),
        )

        decision = self._first_mapping(
            canonical.get("decision"),
            results.get("decision"),
            structured.get("decision"),
        )
        if decision:
            outcome["decision"] = decision
        measurement = self._measurement_summary(results, runs)
        if measurement:
            outcome["metrics"] = measurement
        outcome["trajectory"] = {
            "job_id": str(getattr(job, "id", "") or ""),
            "job_type": str(getattr(job, "job_type", "") or ""),
            "iterations": int(getattr(job, "iteration", 0) or 0),
            "tool_calls_used": int(getattr(job, "tool_calls_used", 0) or 0),
            "llm_calls_used": int(getattr(job, "llm_calls_used", 0) or 0),
            "experiment_run_ids": [
                str(getattr(run, "id", "") or "")
                for run in runs
                if getattr(run, "id", None)
            ],
        }
        if getattr(job, "error", None):
            outcome["error"] = str(job.error)[:1000]
        verified = autonomous_rnd_evidence_verifier.verify(outcome)
        return autonomous_rnd_verification_planner.build_plan(verified)

    def compact_action_ledger(
        self, actions_taken: Any, *, max_actions: int = 200
    ) -> List[Dict[str, Any]]:
        """Project runtime actions without retaining params or raw tool output.

        A long run is cut to the most recent ``max_actions``. The ledger says so
        rather than dropping the earlier ones quietly: a truncated list reads as
        the whole run, and the beginning is where a trajectory usually goes
        wrong.
        """
        if not isinstance(actions_taken, list) or max_actions <= 0:
            return []

        omitted = max(0, len(actions_taken) - max_actions)
        compact: List[Dict[str, Any]] = []
        if omitted:
            compact.append(
                {
                    "tool": "__omitted__",
                    "success": None,
                    "note": (
                        f"{omitted} earlier action(s) omitted; showing the most "
                        f"recent {max_actions}"
                    ),
                }
            )
        for raw_item in actions_taken[-max_actions:]:
            if not isinstance(raw_item, Mapping):
                continue
            item = _mapping(raw_item)
            action = _mapping(item.get("action")) or item
            result = _mapping(item.get("result"))
            tool = str(
                action.get("tool") or item.get("tool") or result.get("tool") or ""
            ).strip()
            if not tool:
                continue

            status = str(result.get("status") or "").strip().lower()
            success = result.get("success")
            if not isinstance(success, bool):
                if status:
                    success = status in {"completed", "succeeded", "success"}
                else:
                    success = not bool(result.get("error"))

            row: Dict[str, Any] = {"tool": tool, "success": success}
            if status:
                row["status"] = status
            for field in ("iteration", "node", "step_id", "parent_tool"):
                value = item.get(field)
                if value is not None and value != "":
                    row[field] = value

            # A repeated identical failure carries its escalation. The ledger
            # is where an operator reviews what a run did, and "this call
            # failed three times and was told to stop" is exactly the shape of
            # trouble worth seeing without replaying the whole run.
            diagnosis = _mapping(result.get("diagnosis"))
            if diagnosis:
                row["repeat_attempt"] = diagnosis.get("attempt")
                row["failure_class"] = diagnosis.get("error_class")
                if diagnosis.get("protocol"):
                    row["diagnosis_escalated"] = True

            tool_name = str(
                result.get("tool_name")
                or _mapping(action.get("params")).get("tool_name")
                or ""
            ).strip()
            if tool_name:
                row["delegated_tool_name"] = tool_name
            tool_type = str(result.get("tool_type") or "").strip()
            if tool_type:
                row["tool_type"] = tool_type

            # A tool that failed and was replaced by a fallback still reports
            # success, under the name of the tool the agent asked for. Without
            # these fields the row is indistinguishable from the requested tool
            # having worked, which is how a broken arXiv search was read as
            # seven relevant findings that were really unrelated KB documents.
            primary_tool = str(result.get("primary_tool") or "").strip()
            executed_tool = str(result.get("tool") or "").strip()
            if primary_tool and executed_tool and primary_tool != executed_tool:
                row["substituted"] = True
                row["requested_tool"] = primary_tool
                row["executed_tool"] = executed_tool
                primary_error = str(result.get("primary_error") or "").strip()
                if primary_error:
                    row["primary_error"] = primary_error[:300]

            provenance = self._find_external_agent_provenance(result)
            if provenance:
                row["external_agent_provenance"] = provenance
            compact.append(row)
        return compact

    @staticmethod
    def _records(*candidates: Any) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, list):
                continue
            for item in candidate:
                if not isinstance(item, Mapping):
                    continue
                row = dict(item)
                key = str(row.get("id") or "").strip() or repr(sorted(row.items()))
                if key in seen:
                    continue
                seen.add(key)
                records.append(row)
        return records

    def _evidence(self, *candidates: Any) -> List[Dict[str, Any]]:
        canonical = candidates[0] if len(candidates) > 0 else None
        result_evidence = candidates[1] if len(candidates) > 1 else None
        structured_evidence = candidates[2] if len(candidates) > 2 else None
        findings = candidates[3] if len(candidates) > 3 else None
        evidence: List[Dict[str, Any]] = []
        seen_ids = set()
        for candidate, origin in (
            (canonical, "persisted_outcome"),
            (result_evidence, "job_result"),
            (structured_evidence, "structured_output"),
        ):
            if not isinstance(candidate, list):
                continue
            for item in candidate:
                if not isinstance(item, Mapping):
                    continue
                row = dict(item)
                row.setdefault("record_origin", origin)
                key = str(row.get("id") or "").strip() or repr(sorted(row.items()))
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                evidence.append(row)
        if not isinstance(findings, list):
            return evidence
        for index, finding in enumerate(findings):
            if not isinstance(finding, Mapping):
                continue
            finding_id = str(
                finding.get("id")
                or finding.get("document_id")
                or finding.get("paper_id")
                or ""
            ).strip()
            if not finding_id or finding_id in seen_ids:
                continue
            evidence.append(
                {
                    "id": finding_id,
                    "kind": str(
                        finding.get("kind") or finding.get("type") or "finding"
                    ).strip(),
                    "source": finding.get("source"),
                    "title": finding.get("title"),
                    "finding_index": index,
                    "record_origin": "runtime_finding",
                }
            )
            seen_ids.add(finding_id)
        return evidence

    def _artifacts(
        self,
        canonical: Any,
        job_artifacts: Any,
        results: Mapping[str, Any],
        runs: List[Any],
    ) -> List[Dict[str, Any]]:
        artifacts = self._records(canonical, job_artifacts)
        seen_kinds = {
            str(item.get("kind") or item.get("type") or "").strip()
            for item in artifacts
        }
        normalized = []
        for item in artifacts:
            row = dict(item)
            kind = str(
                row.get("kind") or row.get("type") or row.get("artifact_type") or ""
            ).strip()
            if kind:
                row["kind"] = kind
            normalized.append(row)

        inventories: List[Any] = [
            _mapping(results.get("measurement_summary")).get("artifact_inventory"),
            _mapping(results.get("compiler_artifacts")).get("artifact_inventory"),
        ]
        for run in runs:
            run_results = _mapping(getattr(run, "results", None))
            run_config = _mapping(getattr(run, "config", None))
            scientific = _mapping(run_config.get("scientific_validation"))
            inventories.extend(
                [
                    run_results.get("artifact_inventory"),
                    _mapping(run_results.get("measurement_summary")).get(
                        "artifact_inventory"
                    ),
                    _mapping(scientific.get("measurement_summary")).get(
                        "artifact_inventory"
                    ),
                    _mapping(scientific.get("compiler_observability")).get(
                        "artifact_inventory"
                    ),
                ]
            )
        for inventory in inventories:
            if not isinstance(inventory, list):
                continue
            for raw_kind in inventory:
                kind = str(raw_kind or "").strip()
                if not kind or kind in seen_kinds:
                    continue
                normalized.append({"kind": kind, "source": "artifact_inventory"})
                seen_kinds.add(kind)
        return normalized

    def _actions(
        self, canonical: Any, results: Mapping[str, Any], execution_log: Any
    ) -> List[Dict[str, Any]]:
        actions = self._records(canonical, results.get("actions"))
        execution = _mapping(results.get("execution_strategy"))
        sources = [execution.get("step_events"), execution_log]
        seen = {
            (
                str(item.get("tool") or ""),
                str(item.get("iteration") or ""),
                str(item.get("type") or item.get("phase") or ""),
            )
            for item in actions
        }
        for source in sources:
            if not isinstance(source, list):
                continue
            for item in source:
                if not isinstance(item, Mapping):
                    continue
                action_value = item.get("action")
                action_map = _mapping(action_value)
                tool = str(
                    item.get("tool")
                    or action_map.get("tool")
                    or (action_value if isinstance(action_value, str) else "")
                    or ""
                ).strip()
                if not tool:
                    continue
                row = {
                    "tool": tool,
                    "iteration": item.get("iteration"),
                    "type": item.get("type") or item.get("phase"),
                }
                key = (
                    tool,
                    str(row.get("iteration") or ""),
                    str(row.get("type") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                actions.append(row)
        return actions

    def _find_external_agent_provenance(
        self, result: Mapping[str, Any]
    ) -> Dict[str, Any]:
        candidates = [
            result.get("provenance"),
            _mapping(result.get("output")).get("provenance"),
            _mapping(_mapping(result.get("output")).get("output")).get("provenance"),
            _mapping(result.get("data")).get("provenance"),
            _mapping(_mapping(result.get("data")).get("output")).get("provenance"),
        ]
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            if not str(candidate.get("external_agent_id") or "").strip():
                continue
            provenance = {
                field: candidate[field]
                for field in self._EXTERNAL_PROVENANCE_FIELDS
                if candidate.get(field) is not None and candidate.get(field) != ""
            }
            if provenance:
                return provenance
        return {}

    def _external_agent_evidence(
        self,
        actions: List[Dict[str, Any]],
        existing_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = {
            str(item.get("id") or "").strip()
            for item in existing_evidence
            if str(item.get("id") or "").strip()
        }
        evidence: List[Dict[str, Any]] = []
        for action in actions:
            provenance = _mapping(action.get("external_agent_provenance"))
            if not provenance:
                continue
            request_id = str(provenance.get("request_id") or "").strip()
            digest = str(provenance.get("response_sha256") or "").strip()
            identity = (
                str(provenance.get("evidence_identity") or "").strip()
                or request_id
                or digest
            )
            if not identity:
                continue
            provider_type = str(
                provenance.get("provider_type") or "generic_agent"
            ).strip()
            is_external_system = provider_type != "generic_agent"
            evidence_id = (
                f"external-system:{identity}"
                if is_external_system
                else f"external-agent:{identity}"
            )
            if evidence_id in seen:
                continue
            row: Dict[str, Any] = {
                "id": evidence_id,
                "kind": (
                    "external_system_response"
                    if is_external_system
                    else "external_agent_response"
                ),
                "record_origin": (
                    "external_system_gateway"
                    if is_external_system
                    else "external_agent_gateway"
                ),
                "verification_status": "unverified",
                "source": provenance.get("endpoint_origin"),
                "external_agent_id": provenance.get("external_agent_id"),
                "external_agent_name": provenance.get("external_agent_name"),
                "external_system_type": (provider_type if is_external_system else None),
                "capability": provenance.get("capability"),
                "request_id": request_id or None,
                "response_sha256": digest or None,
                "remote_references": provenance.get("remote_references"),
                "audit_id": provenance.get("audit_id"),
                "sync_subscription_id": provenance.get("sync_subscription_id"),
            }
            for field in ("received_at", "response_bytes", "execution_time_ms"):
                if provenance.get(field) is not None:
                    row[field] = provenance[field]
            evidence.append(
                {key: value for key, value in row.items() if value is not None}
            )
            seen.add(evidence_id)
        return evidence

    def _experiment(
        self, canonical: Any, results: Mapping[str, Any], runs: List[Any]
    ) -> Dict[str, Any]:
        experiment = deepcopy(_mapping(canonical))
        direct = _mapping(results.get("experiment_run"))
        repeat_counts = [
            experiment.get("repeat_count"),
            _mapping(results.get("measurement_summary")).get("repeat_count"),
            direct.get("repeat_count"),
        ]
        command_results = (
            direct.get("runs") if isinstance(direct.get("runs"), list) else []
        )
        if command_results:
            repeat_counts.append(len(command_results))

        all_commands_ok = experiment.get("all_commands_ok")
        if not isinstance(all_commands_ok, bool):
            if isinstance(direct.get("ok"), bool):
                all_commands_ok = direct["ok"]
            elif command_results:
                all_commands_ok = all(
                    isinstance(item, Mapping) and item.get("ok") is True
                    for item in command_results
                )

        for run in runs:
            run_results = _mapping(getattr(run, "results", None))
            run_config = _mapping(getattr(run, "config", None))
            scientific = _mapping(run_config.get("scientific_validation"))
            repeat_counts.extend(
                [
                    run_results.get("repeat_count"),
                    _mapping(run_results.get("measurement_summary")).get(
                        "repeat_count"
                    ),
                    _mapping(scientific.get("measurement_summary")).get("repeat_count"),
                    _mapping(scientific.get("compiler_observability")).get(
                        "repeat_count"
                    ),
                ]
            )
        parsed_counts = [_positive_int(value) for value in repeat_counts]
        parsed_counts = [value for value in parsed_counts if value is not None]
        experiment["repeat_count"] = max(parsed_counts) if parsed_counts else 0
        experiment["all_commands_ok"] = all_commands_ok is True
        experiment["ran"] = bool(direct.get("ran", bool(command_results)))
        return experiment

    @staticmethod
    def _measurement_summary(
        results: Mapping[str, Any], runs: List[Any]
    ) -> Dict[str, Any]:
        merged = deepcopy(_mapping(results.get("measurement_summary")))
        for run in runs:
            run_results = _mapping(getattr(run, "results", None))
            run_config = _mapping(getattr(run, "config", None))
            scientific = _mapping(run_config.get("scientific_validation"))
            for candidate in (
                scientific.get("measurement_summary"),
                run_results.get("measurement_summary"),
            ):
                if isinstance(candidate, Mapping):
                    merged.update(candidate)
        return merged

    @staticmethod
    def _first_mapping(*values: Any) -> Dict[str, Any]:
        for value in values:
            if isinstance(value, Mapping) and value:
                return deepcopy(dict(value))
        return {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


autonomous_rnd_trajectory_adapter = AutonomousRnDTrajectoryAdapter()
