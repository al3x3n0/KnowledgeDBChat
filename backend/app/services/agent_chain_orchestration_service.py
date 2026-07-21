"""Orchestration helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomy_service import build_domain_profile_compat_policy


class AgentChainOrchestrationService:
    async def evaluate_swarm_fan_in_gate(
        self,
        executor: Any,
        parent_job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Decide whether a swarm fan-in child is allowed to trigger now."""
        chain_cfg = parent_job.chain_config if isinstance(parent_job.chain_config, dict) else {}
        chain_data = chain_cfg.get("chain_data") if isinstance(chain_cfg.get("chain_data"), dict) else {}
        if not bool(chain_data.get("swarm_fan_in_wait_for_all_siblings", False)):
            return {"enabled": False, "ready": True, "already_exists": False}

        group_id = str(chain_data.get("swarm_fan_in_group_id") or "").strip()
        sibling_parent_id = parent_job.parent_job_id
        if not sibling_parent_id:
            return {"enabled": True, "ready": True, "already_exists": False, "group_id": group_id}

        siblings_res = await db.execute(
            select(AgentJob).where(AgentJob.parent_job_id == sibling_parent_id)
        )
        siblings = siblings_res.scalars().all()
        terminal = {
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }
        total_siblings = len(siblings)
        terminal_count = len([s for s in siblings if str(s.status) in terminal])
        expected = int(chain_data.get("swarm_fan_in_expected_siblings", 0) or 0)
        if expected <= 0:
            expected = total_siblings
        ready = bool(total_siblings >= expected and terminal_count >= expected)

        already_exists = False
        if group_id and siblings:
            sibling_ids = [s.id for s in siblings if getattr(s, "id", None) is not None]
            if sibling_ids:
                child_res = await db.execute(
                    select(AgentJob).where(AgentJob.parent_job_id.in_(sibling_ids))
                )
                for child in child_res.scalars().all():
                    cfg = child.config if isinstance(child.config, dict) else {}
                    if str(cfg.get("origin") or "") != "swarm_fan_in_aggregator":
                        continue
                    if str(cfg.get("swarm_fan_in_group_id") or "") == group_id:
                        already_exists = True
                        break

        return {
            "enabled": True,
            "ready": ready,
            "already_exists": already_exists,
            "group_id": group_id,
            "expected_siblings": expected,
            "total_siblings": total_siblings,
            "terminal_siblings": terminal_count,
        }

    async def build_swarm_sibling_payload(
        self,
        executor: Any,
        parent_job: AgentJob,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """Collect sibling job outputs for swarm fan-in aggregators."""
        sibling_parent_id = parent_job.parent_job_id
        if not sibling_parent_id:
            return {}
        siblings_res = await db.execute(
            select(AgentJob).where(AgentJob.parent_job_id == sibling_parent_id)
        )
        siblings = siblings_res.scalars().all()
        if not siblings:
            return {}
        terminal = {
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }
        out: List[Dict[str, Any]] = []
        for s in siblings:
            cfg = s.config if isinstance(s.config, dict) else {}
            origin = str(cfg.get("origin") or "").strip().lower()
            if origin in {"swarm_fan_in_aggregator", "swarm_fan_in_rerun_aggregator"}:
                continue
            out.append(
                {
                    "job_id": str(s.id),
                    "name": str(s.name or "")[:200],
                    "status": str(s.status or ""),
                    "is_terminal": str(s.status or "") in terminal,
                    "progress": int(s.progress or 0),
                    "role": str(cfg.get("swarm_role") or ""),
                    "results": s.results if isinstance(s.results, dict) else {},
                }
            )
        return {
            "swarm_parent_job_id": str(sibling_parent_id),
            "expected_siblings": len(out),
            "terminal_siblings": len([x for x in out if bool(x.get("is_terminal"))]),
            "sibling_jobs": out,
        }

    async def trigger_chained_jobs(
        self,
        executor: Any,
        parent_job: AgentJob,
        event: str,
        db: AsyncSession,
        value: int = 0,
    ) -> List[str]:
        """
        Trigger chained jobs based on the parent job completion.

        Args:
            parent_job: The parent job that completed
            event: Event type ('complete', 'fail', 'progress', 'findings')
            db: Database session
            value: Event value (progress percentage or findings count)

        Returns:
            List of created child job IDs
        """
        # Check if chain should be triggered
        if parent_job.chain_triggered:
            logger.debug(f"Job {parent_job.id} already triggered its chain")
            return []

        if not parent_job.should_trigger_chain(event, value):
            logger.debug(f"Chain trigger condition not met for job {parent_job.id}")
            return []

        # Get chain configuration
        chain_config = parent_job.chain_config or {}
        if not chain_config:
            return []

        fan_in_gate = await executor._evaluate_swarm_fan_in_gate(parent_job, db)
        if bool(fan_in_gate.get("enabled", False)):
            if bool(fan_in_gate.get("already_exists", False)):
                executor._append_job_result_step_event(
                    parent_job,
                    {
                        "type": "swarm_fan_in_duplicate_skipped",
                        "iteration": int(parent_job.iteration or 0),
                        "group_id": str(fan_in_gate.get("group_id") or ""),
                    },
                )
                parent_job.chain_triggered = True
                parent_job.add_log_entry(
                    {
                        "phase": "swarm_fan_in_duplicate_skipped",
                        "group_id": str(fan_in_gate.get("group_id") or ""),
                    }
                )
                await db.commit()
                return []
            if not bool(fan_in_gate.get("ready", False)):
                executor._append_job_result_step_event(
                    parent_job,
                    {
                        "type": "swarm_fan_in_deferred",
                        "iteration": int(parent_job.iteration or 0),
                        "group_id": str(fan_in_gate.get("group_id") or ""),
                        "expected_siblings": int(fan_in_gate.get("expected_siblings", 0) or 0),
                        "terminal_siblings": int(fan_in_gate.get("terminal_siblings", 0) or 0),
                        "total_siblings": int(fan_in_gate.get("total_siblings", 0) or 0),
                    },
                )
                parent_job.add_log_entry(
                    {
                        "phase": "swarm_fan_in_deferred",
                        "group_id": str(fan_in_gate.get("group_id") or ""),
                        "expected_siblings": int(fan_in_gate.get("expected_siblings", 0) or 0),
                        "terminal_siblings": int(fan_in_gate.get("terminal_siblings", 0) or 0),
                        "total_siblings": int(fan_in_gate.get("total_siblings", 0) or 0),
                    }
                )
                await db.commit()
                return []

        # Mark parent as having triggered its chain
        parent_job.chain_triggered = True

        created_job_ids = []

        # Check for defined child jobs in chain_config
        child_jobs_config = chain_config.get("child_jobs", [])
        if child_jobs_config:
            for child_config in child_jobs_config:
                try:
                    child_job = await executor._create_chained_job(
                        parent_job=parent_job,
                        child_config=child_config,
                        db=db,
                    )
                    if child_job:
                        created_job_ids.append(str(child_job.id))
                        child_cfg = child_job.config if isinstance(child_job.config, dict) else {}
                        executor._append_job_result_step_event(
                            parent_job,
                            {
                                "type": "chain_child_spawned",
                                "iteration": int(parent_job.iteration or 0),
                                "child_job_id": str(child_job.id),
                                "child_job_name": str(child_job.name or "")[:140],
                                "child_job_type": str(child_job.job_type or ""),
                                "origin": str(child_cfg.get("origin") or ""),
                                "swarm_role": str(child_cfg.get("swarm_role") or ""),
                            },
                        )
                        logger.info(f"Created chained job {child_job.id} from parent {parent_job.id}")
                except Exception as e:
                    logger.error(f"Failed to create chained job: {e}")

        if created_job_ids:
            executor._append_job_result_step_event(
                parent_job,
                {
                    "type": "chain_triggered",
                    "iteration": int(parent_job.iteration or 0),
                    "trigger_event": str(event or ""),
                    "created_jobs_count": len(created_job_ids),
                    "child_job_ids": created_job_ids[:20],
                },
            )

        await db.commit()

        # Trigger execution of created jobs
        from app.tasks.agent_job_tasks import execute_agent_job_task
        for job_id in created_job_ids:
            execute_agent_job_task.delay(job_id, str(parent_job.user_id))

        return created_job_ids

    async def create_chained_job(
        self,
        executor: Any,
        parent_job: AgentJob,
        child_config: Dict[str, Any],
        db: AsyncSession,
    ) -> Optional[AgentJob]:
        """
        Create a chained child job from configuration.

        Args:
            parent_job: The parent job
            child_config: Configuration for the child job
            db: Database session

        Returns:
            Created AgentJob or None if creation failed
        """
        # Get data to pass to child
        chain_data = parent_job.get_chain_data_for_child()
        parent_config = parent_job.chain_config or {}

        # Build child job configuration
        child_job_config = child_config.get("config", {})

        # Merge inherited config if specified
        if parent_config.get("inherit_config") and parent_job.config:
            child_job_config = {**parent_job.config, **child_job_config}

        # Merge parent results if inheriting
        if parent_config.get("inherit_results", True) and parent_job.results:
            if "inherited_data" not in child_job_config:
                child_job_config["inherited_data"] = {}
            child_job_config["inherited_data"]["parent_results"] = parent_job.results
            child_job_config["inherited_data"]["parent_findings"] = parent_job.results.get("findings", [])

        if str(child_job_config.get("origin") or "") == "swarm_fan_in_aggregator":
            sibling_payload = await executor._build_swarm_sibling_payload(parent_job, db)
            if sibling_payload:
                if "inherited_data" not in child_job_config or not isinstance(child_job_config.get("inherited_data"), dict):
                    child_job_config["inherited_data"] = {}
                child_job_config["inherited_data"]["swarm"] = sibling_payload
        elif str(child_job_config.get("origin") or "") == "swarm_fan_in_rerun_aggregator":
            base_payload = child_job_config.get("swarm_rerun_base_payload") if isinstance(child_job_config.get("swarm_rerun_base_payload"), dict) else {}
            sibling_payload = executor._compose_swarm_rerun_payload(
                base_payload,
                tie_breaker_job=parent_job,
                tie_breaker_source_job_id=str(child_job_config.get("tie_breaker_source_job_id") or ""),
            )
            if "inherited_data" not in child_job_config or not isinstance(child_job_config.get("inherited_data"), dict):
                child_job_config["inherited_data"] = {}
            child_job_config["inherited_data"]["swarm"] = sibling_payload
            child_job_config.pop("swarm_rerun_base_payload", None)

        # Create the child job
        child_job = AgentJob(
            name=child_config.get("name", f"Chained: {parent_job.name}"),
            description=child_config.get("description", f"Chained from job: {parent_job.name}"),
            job_type=child_config.get("job_type", parent_job.job_type),
            goal=child_config.get("goal", parent_job.goal),
            goal_criteria=child_config.get("goal_criteria"),
            config=child_job_config,
            agent_definition_id=child_config.get("agent_definition_id") or parent_job.agent_definition_id,
            user_id=parent_job.user_id,
            status=AgentJobStatus.PENDING.value,
            # Chain hierarchy
            parent_job_id=parent_job.id,
            root_job_id=parent_job.root_job_id or parent_job.id,
            chain_depth=parent_job.chain_depth + 1,
            # Chain config for further chaining
            chain_config=child_config.get("chain_config"),
            # Resource limits - inherit from parent or use child config
            max_iterations=child_config.get("max_iterations", parent_job.max_iterations),
            max_tool_calls=child_config.get("max_tool_calls", parent_job.max_tool_calls),
            max_llm_calls=child_config.get("max_llm_calls", parent_job.max_llm_calls),
            max_runtime_minutes=child_config.get("max_runtime_minutes", parent_job.max_runtime_minutes),
        )

        db.add(child_job)
        await db.flush()  # Get the ID

        # Log the chain creation
        parent_job.add_log_entry({
            "phase": "chain_triggered",
            "child_job_id": str(child_job.id),
            "child_job_name": child_job.name,
            "trigger_event": "complete" if parent_job.status == AgentJobStatus.COMPLETED.value else "fail",
        })

        return child_job

    async def trigger_progress_chain(
        self,
        executor: Any,
        job: AgentJob,
        progress: int,
        findings_count: int,
        db: AsyncSession,
    ) -> List[str]:
        """
        Check and trigger chains based on progress or findings thresholds.

        Called during job execution when progress or findings are updated.

        Args:
            job: The running job
            progress: Current progress percentage
            findings_count: Current findings count
            db: Database session

        Returns:
            List of triggered job IDs
        """
        triggered_jobs = []

        # Check progress-based trigger
        if job.should_trigger_chain("progress", progress):
            triggered = await executor._trigger_chained_jobs(job, "progress", db, progress)
            triggered_jobs.extend(triggered)

        # Check findings-based trigger
        if job.should_trigger_chain("findings", findings_count):
            triggered = await executor._trigger_chained_jobs(job, "findings", db, findings_count)
            triggered_jobs.extend(triggered)

        return triggered_jobs
