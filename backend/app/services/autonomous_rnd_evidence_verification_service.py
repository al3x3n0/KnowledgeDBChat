"""Deterministic verification state transitions for external R&D evidence."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping


class AutonomousRnDEvidenceVerifier:
    """Promote external evidence only through explicit, locally grounded links."""

    _TRUSTED_LOCAL_EVIDENCE_ORIGINS = {
        "runtime_finding",
        "local_experiment",
        "local_tool",
    }

    def verify(self, outcome: Mapping[str, Any]) -> Dict[str, Any]:
        verified = deepcopy(dict(outcome))
        evidence = _records(verified.get("evidence"))
        artifacts = _records(verified.get("artifacts"))
        links = _records(verified.get("verification_links"))
        experiment = _mapping(verified.get("experiment"))
        verification_experiments = _records(verified.get("verification_experiments"))

        evidence_by_id = {
            str(item.get("id") or "").strip(): item
            for item in evidence
            if str(item.get("id") or "").strip()
        }
        artifact_ids = {
            str(item.get("id") or "").strip()
            for item in artifacts
            if str(item.get("id") or "").strip()
        }
        artifact_kinds = {
            str(item.get("kind") or item.get("type") or "").strip()
            for item in artifacts
            if str(item.get("kind") or item.get("type") or "").strip()
        }
        links_by_evidence: Dict[str, List[Dict[str, Any]]] = {}
        for link in links:
            evidence_id = str(
                link.get("external_evidence_id") or link.get("evidence_id") or ""
            ).strip()
            if evidence_id:
                links_by_evidence.setdefault(evidence_id, []).append(link)

        decisions: List[Dict[str, Any]] = []
        counts = {
            "unverified": 0,
            "corroborated": 0,
            "verified": 0,
            "rejected": 0,
        }
        for item in evidence:
            if str(item.get("kind") or "").strip() not in {
                "external_agent_response",
                "external_system_response",
            }:
                continue
            evidence_id = str(item.get("id") or "").strip()
            decision = self._evaluate_links(
                evidence_id=evidence_id,
                links=links_by_evidence.get(evidence_id, []),
                evidence_by_id=evidence_by_id,
                artifact_ids=artifact_ids,
                artifact_kinds=artifact_kinds,
                experiment=experiment,
                verification_experiments=verification_experiments,
            )
            item["verification_status"] = decision["status"]
            item["verification_reason"] = decision["reason"]
            if decision["local_evidence_ids"]:
                item["verified_by_evidence_ids"] = decision["local_evidence_ids"]
            if decision["artifact_ids"]:
                item["verified_by_artifact_ids"] = decision["artifact_ids"]
            if decision["artifact_kinds"]:
                item["verified_by_artifact_kinds"] = decision["artifact_kinds"]
            counts[decision["status"]] += 1
            decisions.append({"evidence_id": evidence_id, **decision})

        verified["evidence"] = evidence
        verified["evidence_verification"] = {
            "policy": "explicit_local_corroboration_v1",
            "external_evidence_count": sum(counts.values()),
            "status_counts": counts,
            "decisions": decisions,
        }
        return verified

    def _evaluate_links(
        self,
        *,
        evidence_id: str,
        links: List[Dict[str, Any]],
        evidence_by_id: Dict[str, Dict[str, Any]],
        artifact_ids: set[str],
        artifact_kinds: set[str],
        experiment: Dict[str, Any],
        verification_experiments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not links:
            return self._decision(
                "unverified",
                "No explicit local verification link was recorded.",
            )

        valid_support: List[Dict[str, Any]] = []
        valid_contradictions: List[Dict[str, Any]] = []
        for link in links:
            local_evidence_ids = [
                item
                for item in _string_list(link.get("local_evidence_ids"))
                if item != evidence_id
                and item in evidence_by_id
                and str(evidence_by_id[item].get("kind") or "").strip()
                not in {"external_agent_response", "external_system_response"}
                and str(evidence_by_id[item].get("record_origin") or "").strip().lower()
                in self._TRUSTED_LOCAL_EVIDENCE_ORIGINS
            ]
            valid_artifact_ids = [
                item
                for item in _string_list(link.get("artifact_ids"))
                if item in artifact_ids
            ]
            valid_artifact_kinds = [
                item
                for item in _string_list(link.get("artifact_kinds"))
                if item in artifact_kinds
            ]
            grounded = bool(
                local_evidence_ids or valid_artifact_ids or valid_artifact_kinds
            )
            if not grounded:
                continue
            normalized = {
                "local_evidence_ids": local_evidence_ids,
                "artifact_ids": valid_artifact_ids,
                "artifact_kinds": valid_artifact_kinds,
                "min_repeat_count": _positive_int(link.get("min_repeat_count"), 2),
                "experiment": self._resolve_link_experiment(
                    link,
                    fallback=experiment,
                    verification_experiments=verification_experiments,
                ),
            }
            verdict = str(link.get("verdict") or "supports").strip().lower()
            if verdict in {"contradicts", "rejects", "rejected"}:
                valid_contradictions.append(normalized)
            elif verdict in {"supports", "corroborates", "verified"}:
                valid_support.append(normalized)

        if valid_contradictions:
            refs = self._merge_references(valid_contradictions)
            return self._decision(
                "rejected",
                "Independent local evidence contradicts the external response.",
                **refs,
            )
        if not valid_support:
            return self._decision(
                "unverified",
                "Verification links did not resolve to local evidence or artifacts.",
            )

        verified_support = [
            item
            for item in valid_support
            if item["local_evidence_ids"]
            and (item["artifact_ids"] or item["artifact_kinds"])
            and item["experiment"].get("ran") is True
            and item["experiment"].get("all_commands_ok") is True
            and _positive_int(item["experiment"].get("repeat_count"), 0)
            >= item["min_repeat_count"]
        ]
        if verified_support:
            refs = self._merge_references(verified_support)
            min_repeats = max(item["min_repeat_count"] for item in verified_support)
            return self._decision(
                "verified",
                "Independent evidence and replayable artifacts were confirmed "
                f"by {min_repeats} or more successful experiment runs.",
                **refs,
            )
        refs = self._merge_references(valid_support)
        return self._decision(
            "corroborated",
            "At least one explicit local evidence or artifact reference was resolved; "
            "the repeatable-experiment verification threshold was not met.",
            **refs,
        )

    @staticmethod
    def _resolve_link_experiment(
        link: Dict[str, Any],
        *,
        fallback: Dict[str, Any],
        verification_experiments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        run_id = str(link.get("experiment_run_id") or "").strip()
        if not run_id:
            return fallback
        return next(
            (
                experiment
                for experiment in verification_experiments
                if str(experiment.get("run_id") or experiment.get("id") or "").strip()
                == run_id
            ),
            {},
        )

    @staticmethod
    def _merge_references(rows: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        return {
            field: sorted(
                {
                    value
                    for row in rows
                    for value in row.get(field, [])
                    if str(value).strip()
                }
            )
            for field in (
                "local_evidence_ids",
                "artifact_ids",
                "artifact_kinds",
            )
        }

    @staticmethod
    def _decision(
        status: str,
        reason: str,
        *,
        local_evidence_ids: List[str] | None = None,
        artifact_ids: List[str] | None = None,
        artifact_kinds: List[str] | None = None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "reason": reason,
            "local_evidence_ids": local_evidence_ids or [],
            "artifact_ids": artifact_ids or [],
            "artifact_kinds": artifact_kinds or [],
        }


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


autonomous_rnd_evidence_verifier = AutonomousRnDEvidenceVerifier()
