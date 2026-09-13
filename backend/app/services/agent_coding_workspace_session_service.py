"""Durable workspace-session handoffs for autonomous coding agents."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, Optional

from app.models.agent_job import AgentJob


class AgentCodingWorkspaceSessionService:
    """Bind coding jobs to sessions and persist exact candidate deltas."""

    SESSION_VERSION = "v1"

    @staticmethod
    def session_id_for_job(job_id: Any) -> str:
        return f"coding-session-{str(job_id)}"

    def bind_job(self, job: AgentJob) -> str:
        """Attach a stable workspace session after the job has an ID."""
        config = dict(job.config) if isinstance(job.config, dict) else {}
        session_id = str(config.get("coding_workspace_session_id") or "").strip()
        if not session_id:
            session_id = self.session_id_for_job(job.id)
        config["coding_workspace_session_id"] = session_id
        config["coding_workspace_session"] = {
            "version": self.SESSION_VERSION,
            "session_id": session_id,
            "root_job_id": str(job.root_job_id or job.id),
            "job_id": str(job.id),
            "persistence": "immutable_candidate_delta",
        }
        job.config = config
        return session_id

    def child_session_config(
        self,
        parent_job: AgentJob,
        *,
        role: str,
        role_index: int,
    ) -> Dict[str, Any]:
        """Build the inherited session view for a swarm child."""
        parent_config = parent_job.config if isinstance(parent_job.config, dict) else {}
        session_id = str(
            parent_config.get("coding_workspace_session_id")
            or self.session_id_for_job(parent_job.id)
        ).strip()
        return {
            "coding_workspace_session_id": session_id,
            "coding_workspace_session": {
                "version": self.SESSION_VERSION,
                "session_id": session_id,
                "root_job_id": str(parent_job.root_job_id or parent_job.id),
                "parent_job_id": str(parent_job.id),
                "workspace_view": "isolated_role_candidate",
                "role": str(role or "").strip(),
                "role_index": int(role_index),
            },
        }

    @staticmethod
    def _snapshot_id(manifest: Dict[str, Any]) -> str:
        identity = {
            "job_id": manifest.get("job_id"),
            "workspace_id": manifest.get("workspace_id"),
            "source_id": manifest.get("source_id"),
            "repo_url": manifest.get("repo_url"),
            "base_digest": manifest.get("base_digest"),
            "files": manifest.get("files") or [],
            "deleted_files": manifest.get("deleted_files") or [],
        }
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"candidate-{hashlib.sha256(encoded).hexdigest()[:24]}"

    @staticmethod
    def _existing_snapshot(
        artifacts: list[Dict[str, Any]],
        workspace_id: str,
    ) -> Optional[Dict[str, Any]]:
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            if str(artifact.get("workspace_id") or "") == workspace_id and str(
                artifact.get("type") or ""
            ) in {"workspace_snapshot", "workspace_delta_snapshot"}:
                return artifact
        return None

    async def persist_candidate_snapshot(
        self,
        executor: Any,
        job: AgentJob,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Persist this job's active workspace before fan-in is triggered."""
        config = job.config if isinstance(job.config, dict) else {}
        if not bool(config.get("coding_harness_enabled")):
            return None

        workspace_id = str(state.get("coding_workspace_id") or "").strip()
        if not workspace_id:
            return None
        workspace = executor.workspace_manager.get(workspace_id)
        if workspace is None:
            return None
        if workspace.owner_job_id and workspace.owner_job_id != str(job.id):
            return None

        artifacts = (
            list(job.output_artifacts) if isinstance(job.output_artifacts, list) else []
        )
        existing = self._existing_snapshot(artifacts, workspace_id)
        if existing is not None:
            self._attach_snapshot_to_results(job, existing)
            return existing

        result = await executor.workspace_manager.persist_workspace(
            workspace=workspace,
            job_id=str(job.id),
            user_id=str(job.user_id),
            document_workspace=(
                state.get("document_workspace")
                if isinstance(state.get("document_workspace"), dict)
                else None
            ),
        )
        raw_manifest = result.get("manifest")
        if not isinstance(raw_manifest, dict):
            return None

        manifest = deepcopy(raw_manifest)
        session_id = str(
            config.get("coding_workspace_session_id")
            or self.session_id_for_job(job.root_job_id or job.id)
        ).strip()
        manifest.update(
            {
                "type": "workspace_delta_snapshot",
                "snapshot_id": self._snapshot_id(manifest),
                "session_id": session_id,
                "session_version": self.SESSION_VERSION,
                "role": str(
                    config.get("coding_harness_role")
                    or config.get("swarm_role_key")
                    or ""
                ).strip(),
                "checkpoint_kind": "candidate",
                "immutable": True,
            }
        )
        artifacts.append(manifest)
        job.output_artifacts = artifacts
        state["artifacts"] = artifacts
        self._attach_snapshot_to_results(job, manifest)
        return manifest

    @staticmethod
    def _attach_snapshot_to_results(
        job: AgentJob,
        manifest: Dict[str, Any],
    ) -> None:
        results = dict(job.results) if isinstance(job.results, dict) else {}
        harness = (
            dict(results.get("coding_harness"))
            if isinstance(results.get("coding_harness"), dict)
            else {}
        )
        reference = {
            key: deepcopy(manifest.get(key))
            for key in (
                "type",
                "snapshot_id",
                "session_id",
                "job_id",
                "workspace_id",
                "source_id",
                "repo_url",
                "base_digest",
                "base_files_count",
                "files",
                "deleted_files",
                "failed_files",
                "changes_summary",
                "persistence_complete",
                "immutable",
            )
            if manifest.get(key) is not None
        }
        snapshots = (
            list(harness.get("workspace_snapshots"))
            if isinstance(harness.get("workspace_snapshots"), list)
            else []
        )
        if not any(
            str(row.get("snapshot_id") or "") == str(reference.get("snapshot_id") or "")
            for row in snapshots
            if isinstance(row, dict)
        ):
            snapshots.append(reference)
        harness["workspace_snapshots"] = snapshots
        if (
            str(manifest.get("type") or "") == "workspace_delta_snapshot"
            and manifest.get("persistence_complete") is True
        ) and bool(
            (job.config or {}).get("coding_harness_may_mutate")
            if isinstance(job.config, dict)
            else False
        ):
            harness["candidate_snapshot"] = reference
        results["coding_harness"] = harness
        job.results = results


agent_coding_workspace_session_service = AgentCodingWorkspaceSessionService()
