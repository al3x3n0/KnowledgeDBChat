"""Safely link audited external-system calls to autonomous R&D jobs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.models.agent_job import AgentJob
from app.models.workflow import UserTool
from app.services.autonomous_rnd_trajectory_service import (
    autonomous_rnd_trajectory_adapter,
)


class ExternalSystemEvidenceLinkError(RuntimeError):
    """Raised when a remote result cannot be safely linked to a job."""


class ExternalSystemEvidenceLinkService:
    """Append only allowlisted provenance; keep remote output in tool audit."""

    _PROVENANCE_FIELDS = (
        "external_agent_id",
        "external_agent_name",
        "endpoint_origin",
        "provider_type",
        "capability",
        "request_id",
        "received_at",
        "response_sha256",
        "response_bytes",
        "execution_time_ms",
        "evidence_identity",
        "sync_subscription_id",
    )
    _REMOTE_REFERENCE_FIELDS = (
        "project_id",
        "workflow_id",
        "run_id",
        "batch_id",
        "study_id",
        "artifact_id",
        "action_id",
    )

    async def link(
        self,
        *,
        job_id: UUID,
        user_id: UUID,
        tool: UserTool,
        gateway_result: Mapping[str, Any],
        audit_id: UUID,
        db: AsyncSession,
        evidence_key: str | None = None,
    ) -> bool:
        config = tool.config if isinstance(tool.config, dict) else {}
        if str(config.get("provider_type") or "") != "compops":
            raise ExternalSystemEvidenceLinkError(
                "Only typed CompOps results can be linked as external-system evidence"
            )
        job = (
            await db.execute(
                select(AgentJob).where(
                    AgentJob.id == job_id,
                    AgentJob.user_id == user_id,
                )
            )
        ).scalar_one_or_none()
        if job is None:
            raise ExternalSystemEvidenceLinkError("Agent job was not found")
        if not isinstance(job.results, dict) or not isinstance(
            job.results.get("evaluation_outcome"), dict
        ):
            raise ExternalSystemEvidenceLinkError(
                "Agent job does not have a canonical R&D outcome"
            )

        provenance = self._safe_provenance(gateway_result.get("provenance"))
        if provenance.get("provider_type") != "compops":
            raise ExternalSystemEvidenceLinkError(
                "CompOps provenance was missing from the external-system result"
            )
        request_id = str(provenance.get("request_id") or "").strip()
        capability = str(provenance.get("capability") or "").strip()
        if not request_id or not capability:
            raise ExternalSystemEvidenceLinkError(
                "External-system provenance requires request_id and capability"
            )

        normalized_evidence_key = str(evidence_key or "").strip()
        if normalized_evidence_key:
            provenance["evidence_identity"] = normalized_evidence_key[:200]
            provenance["sync_subscription_id"] = normalized_evidence_key[:200]
        link_id = (
            f"compops-sync:{normalized_evidence_key}"
            if normalized_evidence_key
            else f"compops:{tool.id}:{capability}:{request_id}"
        )
        results = deepcopy(job.results)
        actions = [
            dict(item)
            for item in results.get("actions", [])
            if isinstance(item, Mapping)
        ]
        existing_index = next(
            (
                index
                for index, item in enumerate(actions)
                if str(item.get("evidence_link_id") or "") == link_id
            ),
            None,
        )
        if existing_index is not None:
            existing_provenance = actions[existing_index].get(
                "external_agent_provenance"
            )
            if isinstance(existing_provenance, Mapping) and str(
                existing_provenance.get("response_sha256") or ""
            ) == str(provenance.get("response_sha256") or ""):
                return False
        provenance["audit_id"] = str(audit_id)
        action = {
            "evidence_link_id": link_id,
            "tool": "external_system:compops",
            "tool_type": "external_agent",
            "status": "completed",
            "success": True,
            "external_agent_provenance": provenance,
        }
        if existing_index is None:
            actions.append(action)
        else:
            actions[existing_index] = action
            evidence_id = f"external-system:{normalized_evidence_key}"
            canonical = results.get("evaluation_outcome")
            if isinstance(canonical, dict):
                if isinstance(canonical.get("evidence"), list):
                    canonical["evidence"] = [
                        item
                        for item in canonical["evidence"]
                        if not isinstance(item, Mapping)
                        or str(item.get("id") or "") != evidence_id
                    ]
                if isinstance(canonical.get("actions"), list):
                    canonical["actions"] = [
                        item
                        for item in canonical["actions"]
                        if not isinstance(item, Mapping)
                        or str(item.get("evidence_link_id") or "") != link_id
                    ]
        results["actions"] = actions[-500:]
        job.results = results
        job.results[
            "evaluation_outcome"
        ] = autonomous_rnd_trajectory_adapter.build_outcome(job)
        flag_modified(job, "results")
        await db.flush()
        return True

    def _safe_provenance(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        safe = {
            field: value[field]
            for field in self._PROVENANCE_FIELDS
            if value.get(field) is not None and value.get(field) != ""
        }
        references = value.get("remote_references")
        if isinstance(references, Mapping):
            safe["remote_references"] = {
                field: str(references[field])[:200]
                for field in self._REMOTE_REFERENCE_FIELDS
                if references.get(field) is not None
            }
        return safe


external_system_evidence_link_service = ExternalSystemEvidenceLinkService()
