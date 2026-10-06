"""Autonomous-job tools: the ``collaboration`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_collaboration_provider(executor: Any) -> FunctionToolProvider:
    """Collaboration tools for AutonomousAgentExecutor."""

    async def _delegate_subtask(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio

        from app.models.agent_job import AgentJob, AgentJobStatus
        from app.services.job_dispatch import enqueue_agent_job

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        chain_depth = int(getattr(job, "chain_depth", 0) or 0)
        if chain_depth >= 3:
            return {"error": "Maximum delegation depth (3) reached"}

        delegated_ids = state.get("delegated_subtask_ids")
        if not isinstance(delegated_ids, list):
            delegated_ids = []
        if len(delegated_ids) >= 5:
            return {"error": "Maximum child job budget (5) reached for this parent"}

        child_name = str(params.get("name") or "Subtask")[:200]
        child_goal = str(params.get("goal") or "").strip()[:2000]
        if not child_goal:
            return {"error": "goal is required"}
        child_type = str(params.get("job_type", "custom")).strip()
        if child_type not in {"research", "analysis", "synthesis", "custom"}:
            child_type = "custom"
        # A copy: the caller's dict is not this handler's to change.
        child_config = (
            dict(params["config"]) if isinstance(params.get("config"), dict) else {}
        )
        share = params.get("share_findings", True)
        if not isinstance(share, bool):
            share = True
        remaining_iters = max(1, (job.max_iterations or 100) - (job.iteration or 0))
        try:
            requested_iters = int(params.get("max_iterations", 30) or 30)
        except (TypeError, ValueError):
            return {"error": "max_iterations must be a number"}
        child_max = max(1, min(requested_iters, remaining_iters))

        if share:
            # Under the key the child's prompt reads. These went to
            # `inherited_findings`, which nothing reads, so a child told
            # "findings shared" started blind.
            child_config.setdefault("inherited_data", {})["parent_findings"] = (
                state.get("findings") or []
            )[-20:]

        try:
            child = AgentJob(
                name=child_name,
                description=f"Subtask delegated from {job.name}: {child_goal[:500]}",
                job_type=child_type,
                goal=child_goal,
                config=child_config,
                status=AgentJobStatus.PENDING.value,
                user_id=job.user_id,
                parent_job_id=job.id,
                chain_depth=chain_depth + 1,
                root_job_id=getattr(job, "root_job_id", None) or job.id,
                max_iterations=child_max,
                max_tool_calls=min(child_max * 5, job.max_tool_calls or 500),
                max_llm_calls=min(child_max * 3, job.max_llm_calls or 200),
                max_runtime_minutes=min(30, job.max_runtime_minutes or 60),
            )
            # A savepoint: this session is the run's, and a refused insert
            # would otherwise leave it unusable for every later tool.
            async with ctx.db.begin_nested():
                ctx.db.add(child)
                await ctx.db.flush()
            delegated_ids.append(str(child.id))
            state["delegated_subtask_ids"] = delegated_ids

            # Committed before it is queued: a worker in another process
            # cannot see a row that has only been flushed, and one that
            # picked the task up first found no job to run.
            await ctx.db.commit()
            enqueue_agent_job(ctx.db, str(child.id), str(job.user_id))

            result = {
                "success": True,
                "data": {
                    "child_job_id": str(child.id),
                    "name": child_name,
                    "status": "pending",
                    "max_iterations": child_max,
                },
            }

            if params.get("wait"):
                timeout = min(int(params.get("timeout_seconds", 60) or 60), 60)
                waited = 0
                while waited < timeout:
                    await asyncio.sleep(3)
                    waited += 3
                    await ctx.db.refresh(child)
                    if child.status in [
                        AgentJobStatus.COMPLETED.value,
                        AgentJobStatus.FAILED.value,
                        AgentJobStatus.CANCELLED.value,
                    ]:
                        result["data"]["status"] = child.status
                        result["data"]["results"] = (
                            child.results if isinstance(child.results, dict) else {}
                        )
                        state.setdefault("delegated_subtask_results", {})[
                            str(child.id)
                        ] = result["data"]["results"]
                        state.setdefault("delegated_subtask_final", {})[
                            str(child.id)
                        ] = {
                            "status": child.status,
                            "results": result["data"]["results"],
                        }
                        break
                else:
                    result["data"]["status"] = child.status
                    result["data"][
                        "note"
                    ] = "Timed out waiting; use wait_for_subtask to check later"

            try:
                await executor._save_checkpoint(job, state, ctx.db)
            except Exception:
                pass
            return result
        except Exception as exc:
            return {"error": f"Failed to create child job: {exc}"}

    async def _wait_for_subtask(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio
        import uuid

        from app.models.agent_job import AgentJob, AgentJobStatus

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        subtask_id = str(params.get("subtask_job_id") or "").strip()
        delegated_ids = state.get("delegated_subtask_ids")
        if not isinstance(delegated_ids, list):
            delegated_ids = []
        if subtask_id not in delegated_ids:
            return {"error": f"Job {subtask_id} is not a delegated subtask of this job"}

        # Only a child that has ended is cached, with the status it ended
        # in. Caching whatever was there replayed a child still running as
        # "completed", with stale results, and never looked at it again.
        cached = (state.get("delegated_subtask_final") or {}).get(subtask_id)
        if isinstance(cached, dict):
            return {
                "success": True,
                "data": {
                    "status": cached.get("status"),
                    "results": cached.get("results") or {},
                    "source": "cache",
                },
            }

        timeout = min(int(params.get("timeout_seconds", 30) or 30), 120)
        try:
            subtask_uuid = uuid.UUID(subtask_id)
            child_query = await ctx.db.execute(
                select(AgentJob).where(
                    AgentJob.id == subtask_uuid, AgentJob.parent_job_id == job.id
                )
            )
            child = child_query.scalar_one_or_none()
            if not child:
                return {"error": f"Child job {subtask_id} not found"}
            waited = 0
            while waited < timeout:
                if child.status in [
                    AgentJobStatus.COMPLETED.value,
                    AgentJobStatus.FAILED.value,
                    AgentJobStatus.CANCELLED.value,
                ]:
                    break
                await asyncio.sleep(3)
                waited += 3
                await ctx.db.refresh(child)
            child_results = child.results if isinstance(child.results, dict) else {}
            if child.status in [
                AgentJobStatus.COMPLETED.value,
                AgentJobStatus.FAILED.value,
                AgentJobStatus.CANCELLED.value,
            ]:
                state.setdefault("delegated_subtask_final", {})[subtask_id] = {
                    "status": child.status,
                    "results": child_results,
                }
            return {
                "success": True,
                "data": {
                    "status": child.status,
                    "progress": child.progress,
                    "results": child_results,
                    "findings_count": len(child_results.get("findings", []))
                    if isinstance(child_results.get("findings"), list)
                    else 0,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to check subtask: {exc}"}

    async def _share_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid
        from datetime import datetime

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.agent_job import AgentJob

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        findings_to_share = params.get("findings") or []
        if not isinstance(findings_to_share, list) or not findings_to_share:
            return {"error": "No findings provided to share"}
        if not getattr(job, "parent_job_id", None):
            return {
                "error": "Cannot share findings: this job has no parent (no siblings)"
            }

        target_ids = params.get("target_job_ids") or []
        if not isinstance(target_ids, list):
            target_ids = []
        try:
            query = select(AgentJob).where(
                AgentJob.parent_job_id == job.parent_job_id,
                AgentJob.id != job.id,
                # The caller's own jobs only, as the other sibling tools do.
                AgentJob.user_id == job.user_id,
            )
            if target_ids:
                target_uuids = []
                for tid in target_ids:
                    try:
                        target_uuids.append(uuid.UUID(str(tid)))
                    except (ValueError, AttributeError):
                        pass
                if not target_uuids:
                    # Addressed to somebody, and none of the addresses could
                    # be read: that is nobody, not everybody.
                    return {
                        "error": "None of target_job_ids is a job id; "
                        "nothing was shared"
                    }
                query = query.where(AgentJob.id.in_(target_uuids))
            siblings_result = await ctx.db.execute(query)
            siblings = siblings_result.scalars().all()
            shared_count = 0
            for sibling in siblings:
                sib_results = (
                    sibling.results if isinstance(sibling.results, dict) else {}
                )
                shared = sib_results.get("shared_findings", [])
                if not isinstance(shared, list):
                    shared = []
                for finding in findings_to_share[:10]:
                    if isinstance(finding, dict):
                        shared.append(
                            {
                                "from_job_id": str(job.id),
                                "title": str(finding.get("title", ""))[:200],
                                "content": str(finding.get("content", ""))[:1000],
                                "category": str(finding.get("category", ""))[:100],
                                "shared_at": datetime.utcnow().isoformat(),
                            }
                        )
                sib_results["shared_findings"] = shared[-50:]
                sibling.results = sib_results
                flag_modified(sibling, "results")
                shared_count += 1
            await ctx.db.flush()
            try:
                await executor._save_checkpoint(job, state, ctx.db)
            except Exception:
                pass
            return {
                "success": True,
                "data": {
                    "siblings_updated": shared_count,
                    "findings_shared": len(findings_to_share[:10]),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to share findings: {exc}"}

    async def _request_review(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        from app.models.agent_job import AgentJob, AgentJobStatus
        from app.services.job_dispatch import enqueue_agent_job

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        review_type = str(params.get("review_type") or "peer_agent").strip()
        content = str(params.get("content_to_review") or "").strip()[:3000]
        if not content:
            return {"error": "content_to_review is required"}
        criteria = [
            str(c)[:200]
            for c in (params.get("review_criteria") or [])
            if isinstance(c, str)
        ][:10]

        review_entry = {
            "type": review_type,
            "content": content[:500],
            "criteria": criteria,
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": int(job.iteration or 0),
        }
        reviews = state.get("review_requests")
        if not isinstance(reviews, list):
            reviews = []
        reviews.append(review_entry)
        state["review_requests"] = reviews[-20:]

        if review_type == "human":
            # This does not pause the run, and used to say it had: it set
            # `approval_checkpoint_pending`, which the executor clears before
            # its next action, and answered "paused_for_human_review". The
            # request is recorded and the owner is told; the run goes on.
            request = {
                "type": "review_request",
                "content_to_review": content,
                "review_criteria": criteria,
                "requested_at": datetime.utcnow().isoformat(),
            }
            notified = False
            try:
                from app.services.notification_service import NotificationService

                async with ctx.db.begin_nested():
                    notification = await NotificationService().create_notification(
                        db=ctx.db,
                        user_id=job.user_id,
                        notification_type="agent_job_alert",
                        title=f"Review requested by {job.name or 'an agent run'}"[:200],
                        message=content[:2000],
                        priority="high",
                        related_entity_type="agent_job",
                        related_entity_id=job.id,
                        data={
                            "source_job_id": str(job.id),
                            "review_criteria": criteria,
                        },
                        commit=False,
                    )
                notified = notification is not None
            except Exception:
                notified = False
            return {
                "success": True,
                "data": {
                    "action": "human_review_requested",
                    "paused": False,
                    "owner_notified": notified,
                    "request": request,
                    "note": (
                        "The request is recorded and the job's owner has been "
                        "notified. The run is NOT paused: continue with work "
                        "that does not depend on the review."
                        if notified
                        else "The request is recorded but the owner could not "
                        "be notified, and the run is NOT paused."
                    ),
                },
            }

        chain_depth = int(getattr(job, "chain_depth", 0) or 0)
        if chain_depth >= 3:
            return {
                "error": "Cannot spawn peer review: maximum delegation depth reached"
            }
        # A reviewer is a child job like any other and counts as one.
        already = state.get("delegated_subtask_ids")
        if isinstance(already, list) and len(already) >= 5:
            return {"error": "Maximum child job budget (5) reached for this parent"}

        try:
            review_goal = f"Review the following content and provide feedback:\n\n{content[:1500]}"
            if criteria:
                review_goal += "\n\nEvaluate against these criteria:\n" + "\n".join(
                    f"- {c}" for c in criteria
                )
            child = AgentJob(
                name=f"Peer review for {job.name}"[:200],
                description="Peer review requested by sibling agent",
                job_type="analysis",
                goal=review_goal,
                config={"review_mode": True},
                status=AgentJobStatus.PENDING.value,
                user_id=job.user_id,
                parent_job_id=job.id,
                chain_depth=chain_depth + 1,
                root_job_id=getattr(job, "root_job_id", None) or job.id,
                max_iterations=10,
                max_tool_calls=30,
                max_llm_calls=15,
                max_runtime_minutes=15,
            )
            async with ctx.db.begin_nested():
                ctx.db.add(child)
                await ctx.db.flush()
            delegated_ids = state.get("delegated_subtask_ids")
            if not isinstance(delegated_ids, list):
                delegated_ids = []
            delegated_ids.append(str(child.id))
            state["delegated_subtask_ids"] = delegated_ids

            # Committed before it is queued: a worker in another process
            # cannot see a row that has only been flushed, and one that
            # picked the task up first found no job to run.
            await ctx.db.commit()
            enqueue_agent_job(ctx.db, str(child.id), str(job.user_id))

            try:
                await executor._save_checkpoint(job, state, ctx.db)
            except Exception:
                pass
            return {
                "success": True,
                "data": {
                    "action": "peer_review_spawned",
                    "review_job_id": str(child.id),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to spawn peer review: {exc}"}

    async def _send_message_to_agent(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime
        from uuid import UUID as _UUID

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.agent_job import AgentJob

        job = ctx.job
        target_job_id_str = str(params.get("target_job_id", "")).strip()
        message_text = str(params.get("message", "")).strip()
        if not target_job_id_str:
            return {"error": "target_job_id is required"}
        if not message_text:
            return {"error": "message is required"}
        try:
            target_job = await ctx.db.get(AgentJob, _UUID(target_job_id_str))
            if not target_job:
                return {"error": f"Target job {target_job_id_str} not found"}
            if str(target_job.user_id) != str(job.user_id):
                return {"error": "Cannot send messages to jobs owned by other users"}
            if str(target_job.id) == str(job.id):
                return {"error": "A job cannot send a message to itself"}
            target_results = (
                target_job.results if isinstance(target_job.results, dict) else {}
            )
            agent_msgs = target_results.get("agent_messages", [])
            if not isinstance(agent_msgs, list):
                agent_msgs = []
            category = str(params.get("category", ""))[:100].strip()
            agent_msgs.append(
                {
                    "from_job_id": str(job.id),
                    "from_job_name": job.name or "unknown",
                    "message": message_text[:2000],
                    "category": category,
                    "sent_at": datetime.utcnow().isoformat(),
                }
            )
            agent_msgs = agent_msgs[-100:]
            target_results["agent_messages"] = agent_msgs
            target_job.results = target_results
            flag_modified(target_job, "results")
            await ctx.db.flush()
            return {
                "success": True,
                "data": {
                    "delivered": True,
                    "target_job_id": target_job_id_str,
                    # Where it is now, after the inbox was trimmed.
                    "message_index": len(agent_msgs) - 1,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to send message: {exc}"}

    async def _read_agent_messages(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        try:
            # Read from the database. Another job's session wrote the message;
            # the copy of this job held in memory since the run began does not
            # have it, so a running job never saw anything it was sent.
            job_results = job.results if isinstance(job.results, dict) else {}
            if ctx.db is not None and getattr(job, "id", None) is not None:
                from app.models.agent_job import AgentJob as _AgentJob

                stored = (
                    await ctx.db.execute(
                        select(_AgentJob.results).where(_AgentJob.id == job.id)
                    )
                ).scalar_one_or_none()
                if isinstance(stored, dict):
                    job_results = stored
            agent_msgs = job_results.get("agent_messages", [])
            if not isinstance(agent_msgs, list):
                agent_msgs = []
            shared = job_results.get("shared_findings", [])
            if not isinstance(shared, list):
                shared = []
            since = max(0, int(params.get("since_index", 0) or 0))
            return {
                "success": True,
                "data": {
                    "messages": agent_msgs[since:],
                    "total": len(agent_msgs),
                    "since_index": since,
                    "shared_findings_count": len(shared),
                    # The findings themselves; the count alone told a job it
                    # had been sent something it had no way to read.
                    "shared_findings": shared[-20:],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to read messages: {exc}"}

    return FunctionToolProvider(
        name="autonomous_collaboration_tools",
        modes={"autonomous"},
        handlers={
            "delegate_subtask": _delegate_subtask,
            "wait_for_subtask": _wait_for_subtask,
            "share_findings": _share_findings,
            "request_review": _request_review,
            "send_message_to_agent": _send_message_to_agent,
            "read_agent_messages": _read_agent_messages,
        },
    )
