"""Restart-safe workspace checkpoints for the autonomous coding harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from sqlalchemy.orm.attributes import flag_modified

from app.models.agent_job import AgentJob


class AgentCodingDurableCheckpointService:
    """Persist and restore job-bound workspace deltas across worker lifetimes."""

    MAX_DURABLE_CHECKPOINTS = 24

    @staticmethod
    def _session_id(job: AgentJob) -> str:
        config = job.config if isinstance(job.config, dict) else {}
        return str(
            config.get("coding_workspace_session_id")
            or f"coding-session-{job.root_job_id or job.id}"
        ).strip()

    @staticmethod
    def _all_artifacts(job: AgentJob, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for collection in (
            job.output_artifacts if isinstance(job.output_artifacts, list) else [],
            state.get("artifacts") if isinstance(state.get("artifacts"), list) else [],
        ):
            for item in collection:
                if not isinstance(item, dict):
                    continue
                identity = str(
                    item.get("checkpoint_id")
                    or item.get("snapshot_id")
                    or item.get("id")
                    or ""
                )
                if identity and identity in seen:
                    continue
                if identity:
                    seen.add(identity)
                artifacts.append(deepcopy(item))
        return artifacts

    def list_checkpoints(
        self,
        job: AgentJob,
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        session_id = self._session_id(job)
        checkpoints = [
            artifact
            for artifact in self._all_artifacts(job, state)
            if str(artifact.get("type") or "") == "workspace_session_checkpoint"
            and str(artifact.get("session_id") or "") == session_id
        ]
        return sorted(
            checkpoints,
            key=lambda item: str(item.get("persisted_at") or ""),
        )

    async def persist(
        self,
        executor: Any,
        job: AgentJob,
        state: Dict[str, Any],
        *,
        label: str = "",
        reason: str = "periodic",
        db: Any = None,
    ) -> Optional[Dict[str, Any]]:
        config = job.config if isinstance(job.config, dict) else {}
        if not bool(config.get("coding_harness_enabled")) or not bool(
            config.get("coding_harness_may_mutate")
        ):
            return None
        workspace_id = str(state.get("coding_workspace_id") or "").strip()
        workspace = executor.workspace_manager.get(workspace_id)
        if workspace is None:
            return None
        if workspace.owner_job_id and workspace.owner_job_id != str(job.id):
            return None

        session_id = self._session_id(job)
        state_digest = executor.workspace_manager.workspace_state_digest(workspace)
        existing = self.list_checkpoints(job, state)
        for checkpoint in reversed(existing):
            if (
                checkpoint.get("persistence_complete") is True
                and str(checkpoint.get("workspace_state_digest") or "") == state_digest
            ):
                self._record_latest(job, state, checkpoint)
                return checkpoint
        if len(existing) >= self.MAX_DURABLE_CHECKPOINTS:
            raise RuntimeError(
                f"Durable workspace checkpoint limit reached "
                f"({self.MAX_DURABLE_CHECKPOINTS})"
            )

        persisted = await executor.workspace_manager.persist_durable_checkpoint(
            workspace,
            job_id=str(job.id),
            user_id=str(job.user_id),
            session_id=session_id,
            label=label,
            reason=reason,
        )
        manifest = persisted.get("manifest")
        if not isinstance(manifest, dict):
            return None

        artifacts = self._all_artifacts(job, state)
        artifacts.append(deepcopy(manifest))
        job.output_artifacts = artifacts
        state["artifacts"] = deepcopy(artifacts)
        if manifest.get("persistence_complete") is True:
            self._record_latest(job, state, manifest)
        else:
            state["coding_durable_checkpoint_error"] = {
                "checkpoint_id": str(manifest.get("checkpoint_id") or ""),
                "failed_files": list(manifest.get("failed_files") or [])[:50],
            }
        if db is not None:
            flag_modified(job, "output_artifacts")
            flag_modified(job, "results")
            await db.commit()
        return manifest

    async def restore(
        self,
        executor: Any,
        job: AgentJob,
        state: Dict[str, Any],
        *,
        checkpoint_id: str,
    ) -> Dict[str, Any]:
        config = job.config if isinstance(job.config, dict) else {}
        if not bool(config.get("coding_harness_enabled")) or not bool(
            config.get("coding_harness_may_mutate")
        ):
            raise ValueError(
                "Durable checkpoint restore requires mutation-owner permission"
            )
        requested = str(checkpoint_id or "").strip()
        manifest = next(
            (
                checkpoint
                for checkpoint in self.list_checkpoints(job, state)
                if str(checkpoint.get("checkpoint_id") or "") == requested
            ),
            None,
        )
        if not isinstance(manifest, dict):
            raise ValueError("Durable workspace checkpoint not found for this session")
        if manifest.get("persistence_complete") is not True:
            raise ValueError("Durable workspace checkpoint is incomplete")

        workspace_id = str(state.get("coding_workspace_id") or "").strip()
        workspace = executor.workspace_manager.get(workspace_id)
        if workspace is None:
            raise ValueError(
                "No active workspace; reconstruct the repository baseline first"
            )
        result, error = await executor.workspace_manager.hydrate_candidate_snapshot(
            workspace,
            manifest,
        )
        if error:
            raise ValueError(error)
        state["coding_restored_durable_checkpoint_id"] = requested
        state["coding_modified_files"] = list(
            dict.fromkeys(
                [
                    *list((result or {}).get("hydrated_files") or []),
                    *list((result or {}).get("deleted_files") or []),
                ]
            )
        )[:200]
        return result or {}

    @staticmethod
    def _record_latest(
        job: AgentJob,
        state: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        checkpoint_id = str(manifest.get("checkpoint_id") or "")
        state["coding_last_durable_checkpoint_id"] = checkpoint_id
        state["coding_last_durable_state_digest"] = str(
            manifest.get("workspace_state_digest") or ""
        )
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
                "checkpoint_id",
                "snapshot_id",
                "session_id",
                "workspace_id",
                "source_id",
                "repo_url",
                "branch",
                "base_digest",
                "workspace_state_digest",
                "changes_summary",
                "persistence_complete",
                "label",
                "reason",
                "persisted_at",
            )
            if manifest.get(key) is not None
        }
        harness["latest_durable_checkpoint"] = reference
        results["coding_harness"] = harness
        job.results = results


agent_coding_durable_checkpoint_service = AgentCodingDurableCheckpointService()
