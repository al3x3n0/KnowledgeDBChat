"""
Monitoring and maintenance tasks.
"""

import asyncio
import os
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import Document, DocumentSource
from app.models.chat import ChatSession, ChatMessage
from app.models.research_note import ResearchNote
from app.models.synthesis_job import SynthesisJob
from app.models.notification import NotificationType, NotificationPreferences
from app.models.experiment import ExperimentRun, ExperimentPlan
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.notification import Notification
from app.models.research_inbox import ResearchInboxItem
from app.services.vector_store import VectorStoreService, vector_store_service
from app.services.llm_service import LLMService
from app.services.notification_service import notification_service
from app.services.operator_interventions import derive_operator_interventions_with_outcomes
from app.api.endpoints.agent_jobs import _build_checkpoint_queue_items
from app.services.research_monitor_profile_service import research_monitor_profile_service

# Note: cleanup_old_data has been moved to app.tasks.maintenance_tasks


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        return asyncio.run(coro)
    return loop.run_until_complete(coro)


@celery_app.task(name="app.tasks.monitoring_tasks.health_check")
def health_check() -> Dict[str, Any]:
    """Perform comprehensive health check of all services."""
    return _run_async(_async_health_check())


@celery_app.task(name="app.tasks.monitoring_tasks.generate_stats")
def generate_stats() -> Dict[str, Any]:
    """Generate system statistics."""
    return _run_async(_async_generate_stats())

@celery_app.task(name="app.tasks.monitoring_tasks.lint_recent_research_notes_citations")
def lint_recent_research_notes_citations() -> Dict[str, Any]:
    """Periodically lint citations in recently-updated research notes (no LLM)."""
    return _run_async(_async_lint_recent_research_notes_citations())

@celery_app.task(name="app.tasks.monitoring_tasks.sync_experiment_runs")
def sync_experiment_runs() -> Dict[str, Any]:
    """Periodically sync ExperimentRun status/results from linked AgentJobs."""
    return _run_async(_async_sync_experiment_runs())


@celery_app.task(name="app.tasks.monitoring_tasks.emit_queue_urgency_alerts")
def emit_queue_urgency_alerts() -> Dict[str, Any]:
    """Emit notifications for urgent queue items and overdue reminders."""
    return _run_async(_async_emit_queue_urgency_alerts())


def _summarize_experiment_run_notification(
    exp: Dict[str, Any],
    run_status: str,
    *,
    launch_mode: str | None = None,
) -> Dict[str, Any]:
    """Build a compact notification message/data payload for an experiment run update."""
    if not isinstance(exp, dict):
        exp = {}

    failed_commands = exp.get("failed_commands") if isinstance(exp.get("failed_commands"), list) else []
    execution_strategy = (
        exp.get("execution_strategy") if isinstance(exp.get("execution_strategy"), dict) else {}
    )
    execution_graph = (
        execution_strategy.get("execution_graph")
        if isinstance(execution_strategy.get("execution_graph"), dict)
        else {}
    )
    operator_interventions = (
        execution_strategy.get("operator_interventions")
        if isinstance(execution_strategy.get("operator_interventions"), list)
        else []
    )
    graph_health = execution_graph.get("graph_health") if isinstance(execution_graph.get("graph_health"), dict) else {}
    reasons = graph_health.get("reasons") if isinstance(graph_health.get("reasons"), list) else []
    recommended_actions = (
        execution_graph.get("recommended_actions")
        if isinstance(execution_graph.get("recommended_actions"), list)
        else []
    )

    final_phase = str(exp.get("final_phase") or "").strip()
    source_name = str(exp.get("source_name") or "").strip()
    source_id = str(exp.get("source_id") or "").strip()
    fallback_attempted = bool(exp.get("fallback_attempted"))
    fallback_ok = exp.get("fallback_ok")
    bootstrap_attempted = bool(exp.get("bootstrap_attempted"))
    bootstrap_ok = exp.get("bootstrap_ok")
    recovery_open = bool(fallback_attempted and failed_commands and fallback_ok is not True)
    reason = next((str(item).strip() for item in reasons if str(item).strip()), "")
    recommended_action = next((str(item).strip() for item in recommended_actions if str(item).strip()), "")
    first_failed_command = next((str(item).strip() for item in failed_commands if str(item).strip()), "")
    normalized_launch_mode = str(launch_mode or "").strip() or None
    derived_interventions = derive_operator_interventions_with_outcomes(
        [item for item in operator_interventions if isinstance(item, dict)],
        current_status=run_status,
        status_values={
            "completed": AgentJobStatus.COMPLETED.value,
            "failed": AgentJobStatus.FAILED.value,
            "cancelled": AgentJobStatus.CANCELLED.value,
            "pending": AgentJobStatus.PENDING.value,
            "running": AgentJobStatus.RUNNING.value,
            "paused": AgentJobStatus.PAUSED.value,
        },
    )
    latest_operator_intervention = next(
        (item for item in reversed(derived_interventions) if isinstance(item, dict)),
        {},
    )
    latest_operator_action = str(latest_operator_intervention.get("action") or "").strip().lower()
    latest_operator_note = str(latest_operator_intervention.get("note") or "").strip()
    latest_operator_status_before = str(latest_operator_intervention.get("job_status_before") or "").strip().lower()
    latest_operator_status_after = str(latest_operator_intervention.get("job_status_after") or "").strip().lower()
    latest_operator_at = str(latest_operator_intervention.get("at") or "").strip()
    latest_operator_outcome = str(latest_operator_intervention.get("outcome_status") or "").strip().lower()
    latest_operator_outcome_reason = str(latest_operator_intervention.get("outcome_reason") or "").strip()

    message_parts = [str(run_status or "").strip()]
    if final_phase:
        message_parts.append(f"phase {final_phase}")
    if source_name:
        message_parts.append(f"repo {source_name}")
    if recovery_open:
        message_parts.append("recovery open")
    elif fallback_attempted:
        message_parts.append("fallback ok" if fallback_ok is True else "fallback attempted")
    elif bootstrap_attempted:
        message_parts.append("bootstrap ok" if bootstrap_ok is True else "bootstrap attempted")
    if latest_operator_action:
        message_parts.append(f"last operator {latest_operator_action.replace('_', ' ')}")
    if latest_operator_outcome:
        message_parts.append(f"operator {latest_operator_outcome.replace('_', ' ')}")
    if reason:
        message_parts.append(reason)

    return {
        "message_suffix": " · ".join(part for part in message_parts if part),
        "data": {
            "final_phase": final_phase or None,
            "source_name": source_name or None,
            "source_id": source_id or None,
            "bootstrap_attempted": bootstrap_attempted,
            "bootstrap_ok": bootstrap_ok if bootstrap_attempted else None,
            "fallback_attempted": fallback_attempted,
            "fallback_ok": fallback_ok if fallback_attempted else None,
            "failed_command_count": len(failed_commands),
            "first_failed_command": first_failed_command or None,
            "recovery_open": recovery_open,
            "recovery_reason": reason or None,
            "recommended_action": recommended_action or None,
            "launch_mode": normalized_launch_mode,
            "latest_operator_action": latest_operator_action or None,
            "latest_operator_note": latest_operator_note or None,
            "latest_operator_status_before": latest_operator_status_before or None,
            "latest_operator_status_after": latest_operator_status_after or None,
            "latest_operator_at": latest_operator_at or None,
            "latest_operator_outcome": latest_operator_outcome or None,
            "latest_operator_outcome_reason": latest_operator_outcome_reason or None,
        },
    }


def _build_experiment_run_notification_action_url(*, note_id: str | None, agent_job_id: str | None) -> str | None:
    """Prefer the autonomous-agent deep link when a backing agent job exists."""
    if agent_job_id:
        return f"/autonomous-agents?job={agent_job_id}"
    if note_id:
        return f"/research-notes?note={note_id}"
    return None


def _build_queue_urgency_notification_action_url(
    *,
    job_id: str | None,
    item_type: str | None,
    sla_bucket: str | None,
) -> str:
    params = ["tab=queue"]
    if job_id:
        params.append(f"job={job_id}")
    if item_type:
        params.append(f"queue_item_type={item_type}")
    if sla_bucket:
        params.append(f"queue_sla={sla_bucket}")
    return f"/autonomous-agents?{'&'.join(params)}"


def _build_policy_guardrail_notification_action_url(
    *,
    job_id: str | None,
    history_entry_id: str | None,
) -> str:
    params = ["tab=queue", "queue_item_type=policy_review"]
    if job_id:
        params.append(f"job={job_id}")
    if history_entry_id:
        params.append(f"policy_history={history_entry_id}")
    return f"/autonomous-agents?{'&'.join(params)}"


def _build_budget_alert_notification_action_url(*, job_id: str | None) -> str:
    params = ["tab=health"]
    if job_id:
        params.append(f"job={job_id}")
    return f"/autonomous-agents?{'&'.join(params)}"


def _build_customer_budget_alert_notification_action_url(*, customer: str | None) -> str:
    params = ["tab=health"]
    if customer:
        params.append(f"health_customer={customer}")
    return f"/autonomous-agents?{'&'.join(params)}"


def _summarize_queue_urgency_notification(item: Dict[str, Any]) -> Dict[str, Any]:
    title = str(item.get("title") or "Queue item requires attention").strip() or "Queue item requires attention"
    item_type = str(item.get("item_type") or "").strip()
    reason = str(item.get("reason_label") or item.get("reason_code") or "").strip()
    sla_bucket = str(item.get("sla_bucket") or "").strip()
    escalation = str(item.get("escalation_level") or "").strip()
    recommended = str(item.get("recommended_action") or "").strip()
    customer = str(item.get("customer") or "").strip()
    age_minutes = int(item.get("age_minutes") or 0)
    priority_score = float(item.get("priority_score") or 0)
    is_stale = bool(item.get("is_stale"))
    evidence = str(item.get("evidence_summary") or "").strip()
    scheduler_state = item.get("scheduler_state") if isinstance(item.get("scheduler_state"), dict) else None

    message_parts = []
    if item_type:
        message_parts.append(item_type.replace("_", " "))
    if sla_bucket:
        message_parts.append(sla_bucket.replace("_", " "))
    if escalation:
        message_parts.append(f"escalation {escalation}")
    if reason:
        message_parts.append(reason)
    if is_stale:
        message_parts.append("stale")
    if age_minutes > 0:
        message_parts.append(f"age {age_minutes}m")
    if recommended:
        message_parts.append(f"next {recommended.replace('_', ' ')}")
    message = " · ".join(message_parts[:6]) or "Urgent queue item requires operator attention."

    return {
        "title": f"Queue alert: {title[:180]}",
        "message": message,
        "priority": "high" if sla_bucket == "overdue" else "normal",
        "data": {
            "queue_key": str(item.get("queue_key") or ""),
            "queue_item_type": item_type or None,
            "job_id": str(item.get("job_id") or "") or None,
            "sla_bucket": sla_bucket or None,
            "escalation_level": escalation or None,
            "priority_score": priority_score,
            "recommended_action": recommended or None,
            "reason_label": reason or None,
            "customer": customer or None,
            "age_minutes": age_minutes,
            "is_overdue": bool(item.get("is_overdue")),
            "is_stale": is_stale,
            "evidence_summary": evidence or None,
            "scheduler_state": deepcopy(scheduler_state) if isinstance(scheduler_state, dict) else None,
        },
    }


def _summarize_policy_guardrail_notification(item: Dict[str, Any]) -> Dict[str, Any]:
    title = str(item.get("title") or "Monitor policy safeguard recommended").strip() or "Monitor policy safeguard recommended"
    action = str(item.get("policy_guardrail_action") or item.get("recommended_action") or "").strip()
    reasons = [str(reason).strip() for reason in (item.get("policy_guardrail_reasons") or []) if str(reason).strip()]
    customer = str(item.get("customer") or "").strip()
    message_parts = ["degrading policy evaluation"]
    if action:
        message_parts.append(f"suggested {action.replace('_', ' ')}")
    if customer:
        message_parts.append(f"customer {customer}")
    if reasons:
        message_parts.append(reasons[0])
    return {
        "title": f"Policy safeguard: {title[:180]}",
        "message": " · ".join(message_parts[:4]),
        "priority": "high",
        "data": {
            "queue_key": str(item.get("queue_key") or ""),
            "monitor_job_id": str(item.get("job_id") or "") or None,
            "history_entry_id": str(item.get("policy_guardrail_target_history_entry_id") or "") or None,
            "policy_guardrail_action": action or None,
            "policy_guardrail_reasons": reasons[:3],
            "customer": customer or None,
        },
    }


def _summarize_budget_guardrail_notification(item: Dict[str, Any]) -> Dict[str, Any]:
    title = str(item.get("title") or "Monitor autonomy budget throttled").strip() or "Monitor autonomy budget throttled"
    throttle_state = str(item.get("budget_throttle_state") or "").strip()
    reasons = [str(reason).strip() for reason in (item.get("budget_throttle_reasons") or []) if str(reason).strip()]
    customer = str(item.get("customer") or "").strip()
    message_parts = ["autonomy budget throttled"]
    if throttle_state:
        message_parts.append(throttle_state.replace("_", " "))
    if customer:
        message_parts.append(f"customer {customer}")
    if reasons:
        message_parts.append(reasons[0])
    return {
        "title": f"Budget throttle: {title[:180]}",
        "message": " · ".join(message_parts[:4]),
        "priority": "high",
        "data": {
            "queue_key": str(item.get("queue_key") or ""),
            "job_id": str(item.get("job_id") or "") or None,
            "monitor_job_id": str(item.get("job_id") or "") or None,
            "budget_throttle_state": throttle_state or None,
            "budget_throttle_reasons": reasons[:3],
            "customer": customer or None,
        },
    }


def _summarize_customer_budget_notification(item: Dict[str, Any]) -> Dict[str, Any]:
    customer = str(item.get("customer") or "").strip() or "Customer"
    throttle_state = str(item.get("customer_budget_throttle_state") or "").strip()
    reasons = [
        str(reason).strip()
        for reason in (item.get("customer_budget_throttle_reasons") or [])
        if str(reason).strip()
    ]
    message_parts = ["shared customer autonomy budget throttled"]
    if throttle_state:
        message_parts.append(throttle_state.replace("_", " "))
    if reasons:
        message_parts.append(reasons[0])
    return {
        "title": f"Customer budget throttle: {customer[:180]}",
        "message": " · ".join(message_parts[:4]),
        "priority": "high",
        "data": {
            "customer": customer,
            "customer_budget_throttle_state": throttle_state or None,
            "customer_budget_throttle_reasons": reasons[:3],
        },
    }


def _customer_budget_alert_should_emit(
    *,
    item: Dict[str, Any],
    existing_notifications: list[Notification],
) -> bool:
    customer = str(item.get("customer") or "").strip()
    throttle_state = str(item.get("customer_budget_throttle_state") or "").strip().lower()
    if not customer or not throttle_state or throttle_state == "normal":
        return False
    matching = []
    for notification in existing_notifications:
        payload = notification.data if isinstance(notification.data, dict) else {}
        if str(payload.get("customer") or "").strip() == customer:
            matching.append(notification)
    if not matching:
        return True
    matching.sort(key=lambda n: n.created_at or datetime.min, reverse=True)
    latest = matching[0]
    latest_data = latest.data if isinstance(latest.data, dict) else {}
    latest_state = str(latest_data.get("customer_budget_throttle_state") or "").strip().lower()
    if latest_state != throttle_state:
        return True
    return False


def _queue_alert_should_emit(
    *,
    item: Dict[str, Any],
    existing_notifications: list[Notification],
    reminder_cooldown_hours: int,
    now: datetime,
) -> bool:
    queue_key = str(item.get("queue_key") or "").strip()
    sla_bucket = str(item.get("sla_bucket") or "").strip().lower()
    escalation_level = str(item.get("escalation_level") or "").strip().lower()
    item_type = str(item.get("item_type") or "").strip().lower()
    if item_type in {"policy_review", "budget_review"}:
        sla_bucket = sla_bucket or "overdue"
    if not queue_key or sla_bucket not in {"at_risk", "overdue"}:
        return False

    matching = []
    for notification in existing_notifications:
        payload = notification.data if isinstance(notification.data, dict) else {}
        if str(payload.get("queue_key") or "").strip() == queue_key:
            matching.append(notification)
    if not matching:
        return True

    matching.sort(key=lambda n: n.created_at or datetime.min, reverse=True)
    latest = matching[0]
    latest_data = latest.data if isinstance(latest.data, dict) else {}
    latest_sla = str(latest_data.get("sla_bucket") or "").strip().lower()
    latest_escalation = str(latest_data.get("escalation_level") or "").strip().lower()

    if latest_sla != sla_bucket or latest_escalation != escalation_level:
        return True

    if item_type in {"policy_review", "budget_review"}:
        return False

    if sla_bucket == "overdue":
        cooldown = max(1, int(reminder_cooldown_hours or 6))
        created_at = latest.created_at or datetime.min
        if (now - created_at) >= timedelta(hours=cooldown):
            return True

    return False


async def _async_health_check() -> Dict[str, Any]:
    """Async implementation of health check."""
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "services": {}
    }
    
    # Check database connectivity
    try:
        async with create_celery_session()() as db:
            result = await db.execute(select(func.count(DocumentSource.id)))
            source_count = result.scalar()
            
            health_status["services"]["database"] = {
                "status": "healthy",
                "message": f"Connected successfully, {source_count} sources configured"
            }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "unhealthy"
    
    # Check vector store
    try:
        if getattr(vector_store_service, "_initialized", False):
            stats = await vector_store_service.get_collection_stats()
            health_status["services"]["vector_store"] = {
                "status": "healthy",
                "message": f"Vector store operational, {stats.get('total_chunks', 0)} chunks indexed"
            }
        else:
            health_status["services"]["vector_store"] = {
                "status": "degraded",
                "message": "Vector store not initialized yet"
            }
            if health_status["overall_status"] == "healthy":
                health_status["overall_status"] = "degraded"
    except Exception as e:
        health_status["services"]["vector_store"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "unhealthy"
    
    # Check LLM service
    try:
        llm_service = LLMService()
        is_healthy = await llm_service.health_check()
        
        if is_healthy:
            models = await llm_service.list_available_models()
            health_status["services"]["llm"] = {
                "status": "healthy",
                "message": f"Ollama operational, {len(models)} models available"
            }
        else:
            health_status["services"]["llm"] = {
                "status": "unhealthy",
                "message": "Ollama service unavailable"
            }
            health_status["overall_status"] = "degraded"
    except Exception as e:
        health_status["services"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["overall_status"] = "degraded"
    
    # Check disk space
    try:
        import shutil
        data_dir = "./data"
        total, used, free = shutil.disk_usage(data_dir)
        
        # Convert to GB
        total_gb = total // (1024**3)
        used_gb = used // (1024**3)
        free_gb = free // (1024**3)
        usage_percent = (used / total) * 100
        
        status = "healthy"
        if usage_percent > 90:
            status = "critical"
            health_status["overall_status"] = "unhealthy"
        elif usage_percent > 80:
            status = "warning"
            if health_status["overall_status"] == "healthy":
                health_status["overall_status"] = "degraded"
        
        health_status["services"]["disk_space"] = {
            "status": status,
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "usage_percent": round(usage_percent, 2)
        }
    except Exception as e:
        health_status["services"]["disk_space"] = {
            "status": "unknown",
            "error": str(e)
        }
    
    logger.info(f"Health check completed: {health_status['overall_status']}")
    return health_status


async def _async_lint_recent_research_notes_citations() -> Dict[str, Any]:
    """
    Lint citations for recently-updated research notes.

    The linter checks:
    - uncited lines (heuristic)
    - unknown citation keys ([[S99]]) relative to the note's source docs
    - bibliography presence (## Sources)

    Results are stored in `ResearchNote.attribution.lint`.
    """
    import re
    from uuid import UUID

    now = datetime.utcnow()
    window = timedelta(hours=24)
    max_notes = 500
    max_sources = 10
    max_uncited_examples = 10
    default_coverage_threshold = 0.7
    default_notify_cooldown_hours = 12

    processed = 0
    updated = 0
    skipped = 0
    missing_sources = 0
    notified = 0

    async with create_celery_session()() as db:
        stmt = (
            select(ResearchNote)
            .where(ResearchNote.updated_at >= (now - window))
            .order_by(ResearchNote.updated_at.desc())
            .limit(max_notes)
        )
        res = await db.execute(stmt)
        notes = list(res.scalars().all())

        user_ids = list({n.user_id for n in notes if getattr(n, "user_id", None)})
        prefs_by_user: dict[Any, NotificationPreferences] = {}
        if user_ids:
            try:
                pref_res = await db.execute(
                    select(NotificationPreferences).where(NotificationPreferences.user_id.in_(user_ids))
                )
                prefs_by_user = {p.user_id: p for p in pref_res.scalars().all()}
            except Exception as exc:
                logger.warning(f"Failed to load notification preferences for citation lint task: {exc}")

        def _parse_ts(v: str | None) -> datetime | None:
            if not v:
                return None
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                return None

        def _is_line_citable(line: str) -> bool:
            s = (line or "").strip()
            if not s:
                return False
            if s.startswith("#"):
                return False
            if s.startswith("```") or s.startswith(">"):
                return False
            return bool(re.search(r"[A-Za-z0-9]", s))

        for i, note in enumerate(notes, start=1):
            processed += 1
            try:
                prefs = prefs_by_user.get(note.user_id)

                attribution = note.attribution if isinstance(note.attribution, dict) else {}
                lint = attribution.get("lint") if isinstance(attribution.get("lint"), dict) else None
                last_linted_at = _parse_ts(str(lint.get("generated_at")) if lint else None)
                last_notified_at = _parse_ts(str(lint.get("notified_at")) if lint else None) if lint else None
                note_updated_at = note.updated_at.replace(tzinfo=None) if note.updated_at else None

                # Skip if lint is newer than note update.
                if last_linted_at and note_updated_at and last_linted_at >= note_updated_at:
                    skipped += 1
                    continue

                doc_ids: list[UUID] = []
                if isinstance(note.source_document_ids, list) and note.source_document_ids:
                    for x in note.source_document_ids:
                        try:
                            doc_ids.append(UUID(str(x)))
                        except Exception:
                            pass
                if not doc_ids and note.source_synthesis_job_id:
                    job = await db.get(SynthesisJob, note.source_synthesis_job_id)
                    if job and job.user_id == note.user_id and isinstance(job.document_ids, list):
                        for x in job.document_ids:
                            try:
                                doc_ids.append(UUID(str(x)))
                            except Exception:
                                pass

                if not doc_ids:
                    missing_sources += 1
                    continue

                doc_ids = doc_ids[:max_sources]
                doc_res = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
                documents_by_id = {str(d.id): d for d in doc_res.scalars().all()}
                documents: list[Document] = []
                for did in doc_ids:
                    d = documents_by_id.get(str(did))
                    if d:
                        documents.append(d)

                sources_index = [
                    {"key": f"S{i2 + 1}", "doc_id": str(d.id), "title": d.title, "url": d.url}
                    for i2, d in enumerate(documents)
                ]
                max_key_num = len(sources_index)

                markdown = (note.content_markdown or "").strip()
                used_citation_keys: list[str] = []
                unknown_citation_keys: list[str] = []
                seen_keys = set()
                for m in re.finditer(r"\[\[S(\d+)\]\]", markdown):
                    num = int(m.group(1))
                    key = f"S{num}"
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    used_citation_keys.append(key)
                    if num < 1 or num > max_key_num:
                        unknown_citation_keys.append(key)

                total_citable_lines = 0
                cited_citable_lines = 0
                uncited_examples: list[dict[str, Any]] = []
                for line_no, line in enumerate(markdown.splitlines(), start=1):
                    if not _is_line_citable(line):
                        continue
                    total_citable_lines += 1
                    if "[[S" in line:
                        cited_citable_lines += 1
                        continue
                    if max_uncited_examples > 0:
                        uncited_examples.append({"line_no": line_no, "line": line[:500]})
                        if len(uncited_examples) >= max_uncited_examples:
                            break

                bibliography_present = bool(re.search(r"^##\s+Sources\s*$", markdown, flags=re.MULTILINE))

                lint_report = {
                    "generated_at": now.isoformat(),
                    "document_ids_used": [str(x) for x in doc_ids],
                    "sources": sources_index,
                    "bibliography_present": bibliography_present,
                    "used_citation_keys": used_citation_keys,
                    "unknown_citation_keys": unknown_citation_keys,
                    "total_citable_lines": total_citable_lines or None,
                    "cited_citable_lines": cited_citable_lines or None,
                    "line_citation_coverage": (float(cited_citable_lines) / float(total_citable_lines)) if total_citable_lines else None,
                    "uncited_examples": uncited_examples,
                }

                notify_enabled = True if prefs is None else bool(getattr(prefs, "notify_research_note_citation_issues", True))
                coverage_threshold = (
                    float(getattr(prefs, "research_note_citation_coverage_threshold", default_coverage_threshold))
                    if prefs is not None
                    else float(default_coverage_threshold)
                )
                if not (0.0 <= coverage_threshold <= 1.0):
                    coverage_threshold = float(default_coverage_threshold)

                notify_cooldown_hours = (
                    int(getattr(prefs, "research_note_citation_notify_cooldown_hours", default_notify_cooldown_hours))
                    if prefs is not None
                    else int(default_notify_cooldown_hours)
                )
                if notify_cooldown_hours < 0:
                    notify_cooldown_hours = int(default_notify_cooldown_hours)
                notify_cooldown = timedelta(hours=notify_cooldown_hours)

                notify_on_unknown_keys = True if prefs is None else bool(getattr(prefs, "research_note_citation_notify_on_unknown_keys", True))
                notify_on_low_coverage = True if prefs is None else bool(getattr(prefs, "research_note_citation_notify_on_low_coverage", True))
                notify_on_missing_bibliography = True if prefs is None else bool(getattr(prefs, "research_note_citation_notify_on_missing_bibliography", True))

                lint_report["notify_settings"] = {
                    "enabled": notify_enabled,
                    "coverage_threshold": coverage_threshold,
                    "cooldown_hours": notify_cooldown_hours,
                    "notify_on_unknown_keys": notify_on_unknown_keys,
                    "notify_on_low_coverage": notify_on_low_coverage,
                    "notify_on_missing_bibliography": notify_on_missing_bibliography,
                }

                # Notify user if note looks under-cited or has unknown citation keys.
                reasons: list[str] = []
                coverage = lint_report.get("line_citation_coverage")
                if notify_on_low_coverage and isinstance(coverage, (int, float)) and float(coverage) < coverage_threshold:
                    reasons.append(f"low_coverage<{coverage_threshold}")
                if notify_on_unknown_keys and unknown_citation_keys:
                    reasons.append("unknown_citation_keys")
                if notify_on_missing_bibliography and not bibliography_present:
                    reasons.append("missing_bibliography")

                should_notify = bool(reasons) and notify_enabled
                if should_notify and last_notified_at and (now - last_notified_at) < notify_cooldown:
                    should_notify = False

                if should_notify:
                    title = "Research note needs citations"
                    cov_pct = None
                    if isinstance(coverage, (int, float)):
                        cov_pct = int(round(float(coverage) * 100))
                    msg_parts = []
                    if cov_pct is not None:
                        msg_parts.append(f"Cited lines: {cov_pct}%")
                    if notify_on_unknown_keys and unknown_citation_keys:
                        msg_parts.append(f"Unknown keys: {', '.join(unknown_citation_keys[:5])}")
                    if notify_on_missing_bibliography and not bibliography_present:
                        msg_parts.append("Missing bibliography (## Sources)")
                    message = " · ".join(msg_parts) if msg_parts else "Citation issues detected."

                    await notification_service.create_notification(
                        db=db,
                        user_id=note.user_id,
                        notification_type=NotificationType.RESEARCH_NOTE_CITATION_ISSUE,
                        title=title,
                        message=message,
                        priority="high" if ("unknown_citation_keys" in reasons) else "normal",
                        related_entity_type="research_note",
                        related_entity_id=note.id,
                        action_url=f"/research-notes?note={note.id}&action=citation-fix",
                        data={
                            "note_id": str(note.id),
                            "reasons": reasons,
                            "coverage": coverage,
                            "unknown_citation_keys": unknown_citation_keys,
                            "bibliography_present": bibliography_present,
                            "coverage_threshold": coverage_threshold,
                        },
                    )
                    lint_report["notified_at"] = now.isoformat()
                    lint_report["notified_reasons"] = reasons
                    notified += 1

                note.attribution = {**attribution, "lint": lint_report}
                updated += 1
            except Exception as exc:
                logger.warning(f"Failed to lint research note {getattr(note, 'id', None)}: {exc}")

            # Commit in small batches.
            if i % 50 == 0:
                try:
                    await db.commit()
                except Exception:
                    await db.rollback()

        try:
            await db.commit()
        except Exception:
            await db.rollback()

    return {
        "timestamp": now.isoformat(),
        "window_hours": int(window.total_seconds() // 3600),
        "max_notes": max_notes,
        "processed": processed,
        "updated": updated,
        "skipped": skipped,
        "missing_sources": missing_sources,
        "notified": notified,
    }


async def _async_generate_stats() -> Dict[str, Any]:
    """Generate comprehensive system statistics."""
    async with create_celery_session()() as db:
        try:
            stats = {
                "timestamp": datetime.utcnow().isoformat(),
                "documents": {},
                "chat": {},
                "sources": {},
                "processing": {}
            }
            
            # Document statistics
            total_docs_result = await db.execute(select(func.count(Document.id)))
            total_docs = total_docs_result.scalar()
            
            processed_docs_result = await db.execute(
                select(func.count(Document.id)).where(Document.is_processed == True)
            )
            processed_docs = processed_docs_result.scalar()
            
            failed_docs_result = await db.execute(
                select(func.count(Document.id)).where(
                    and_(Document.is_processed == False, Document.processing_error.isnot(None))
                )
            )
            failed_docs = failed_docs_result.scalar()
            
            stats["documents"] = {
                "total": total_docs,
                "processed": processed_docs,
                "failed": failed_docs,
                "pending": total_docs - processed_docs - failed_docs,
                "success_rate": round((processed_docs / total_docs * 100) if total_docs > 0 else 0, 2)
            }

            # Documents without summary (processed but missing or empty summary)
            try:
                without_summary_result = await db.execute(
                    select(func.count(Document.id)).where(
                        and_(
                            Document.is_processed == True,
                            (Document.summary.is_(None)) | (Document.summary == "")
                        )
                    )
                )
                stats["documents"]["without_summary"] = int(without_summary_result.scalar() or 0)
            except Exception:
                stats["documents"]["without_summary"] = 0
            
            # Chat statistics
            total_sessions_result = await db.execute(select(func.count(ChatSession.id)))
            total_sessions = total_sessions_result.scalar()
            
            total_messages_result = await db.execute(select(func.count(ChatMessage.id)))
            total_messages = total_messages_result.scalar()
            
            # Active sessions (with messages in last 24 hours)
            yesterday = datetime.utcnow() - timedelta(days=1)
            active_sessions_result = await db.execute(
                select(func.count(ChatSession.id.distinct())).where(
                    ChatSession.last_message_at >= yesterday
                )
            )
            active_sessions = active_sessions_result.scalar()
            
            stats["chat"] = {
                "total_sessions": total_sessions,
                "active_sessions_24h": active_sessions,
                "total_messages": total_messages,
                "avg_messages_per_session": round(
                    (total_messages / total_sessions) if total_sessions > 0 else 0, 2
                )
            }
            
            # Source statistics
            sources_by_type_result = await db.execute(
                select(DocumentSource.source_type, func.count(DocumentSource.id))
                .group_by(DocumentSource.source_type)
            )
            sources_by_type = dict(sources_by_type_result.all())
            
            active_sources_result = await db.execute(
                select(func.count(DocumentSource.id)).where(DocumentSource.is_active == True)
            )
            active_sources = active_sources_result.scalar()
            
            stats["sources"] = {
                "total": sum(sources_by_type.values()),
                "active": active_sources,
                "by_type": sources_by_type
            }
            
            # Vector store statistics
            try:
                if getattr(vector_store_service, "_initialized", False):
                    vector_stats = await vector_store_service.get_collection_stats()
                    stats["vector_store"] = vector_stats
                else:
                    stats["vector_store"] = {"status": "not_initialized"}
            except Exception as e:
                stats["vector_store"] = {"error": str(e)}
            
            # Processing statistics (documents by date)
            last_week = datetime.utcnow() - timedelta(days=7)
            recent_docs_result = await db.execute(
                select(
                    func.date(Document.created_at).label('date'),
                    func.count(Document.id).label('count')
                )
                .where(Document.created_at >= last_week)
                .group_by(func.date(Document.created_at))
                .order_by(func.date(Document.created_at))
            )
            recent_docs = [{"date": str(row.date), "count": row.count} 
                          for row in recent_docs_result.all()]
            
            stats["processing"] = {
                "documents_last_7_days": recent_docs,
                "total_documents_last_7_days": sum(item["count"] for item in recent_docs)
            }
            
            logger.info("System statistics generated successfully")
            return stats
            
        except Exception as e:
            logger.error(f"Error generating system statistics: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }

async def _async_sync_experiment_runs() -> Dict[str, Any]:
    """Sync ExperimentRun fields from linked AgentJobs."""
    now = datetime.utcnow()
    max_runs = 200

    processed = 0
    updated = 0
    missing_job = 0

    async with create_celery_session()() as db:
        stmt = (
            select(ExperimentRun)
            .where(
                and_(
                    ExperimentRun.agent_job_id.isnot(None),
                    ExperimentRun.status.in_(["running", "planned"]),
                )
            )
            .order_by(ExperimentRun.updated_at.desc())
            .limit(max_runs)
        )
        res = await db.execute(stmt)
        runs = list(res.scalars().all())

        for i, run in enumerate(runs, start=1):
            processed += 1
            try:
                job = await db.get(AgentJob, run.agent_job_id)
                if not job or job.user_id != run.user_id:
                    missing_job += 1
                    continue

                prev_status = run.status
                job_status = str(job.status or "").lower()

                if job_status == AgentJobStatus.COMPLETED.value:
                    run.status = "completed"
                    run.progress = 100
                    run.completed_at = run.completed_at or (job.completed_at or now)
                    run.started_at = run.started_at or job.started_at
                elif job_status == AgentJobStatus.FAILED.value:
                    run.status = "failed"
                    run.progress = int(job.progress or 0)
                    run.completed_at = run.completed_at or (job.completed_at or now)
                    run.started_at = run.started_at or job.started_at
                elif job_status == AgentJobStatus.CANCELLED.value:
                    run.status = "cancelled"
                    run.progress = int(job.progress or 0)
                    run.completed_at = run.completed_at or (job.completed_at or now)
                    run.started_at = run.started_at or job.started_at
                elif job_status in {AgentJobStatus.RUNNING.value, AgentJobStatus.PENDING.value, AgentJobStatus.PAUSED.value}:
                    run.status = "running"
                    run.progress = int(job.progress or 0)
                    run.started_at = run.started_at or job.started_at

                jr = job.results if isinstance(job.results, dict) else {}
                exp = jr.get("experiment_run") if isinstance(jr.get("experiment_run"), dict) else None
                if exp:
                    run.results = exp
                    if not run.summary:
                        note = exp.get("note") or exp.get("summary")
                        if note:
                            run.summary = str(note)[:20000]

                if run.status != prev_status or exp:
                    updated += 1
                terminal = run.status in ["completed", "failed", "cancelled"]
                transitioned = terminal and (run.status != prev_status)
                if transitioned:
                    try:
                        pref_res = await db.execute(
                            select(NotificationPreferences).where(NotificationPreferences.user_id == run.user_id)
                        )
                        prefs = pref_res.scalar_one_or_none()
                        enabled = True if prefs is None else bool(getattr(prefs, "notify_experiment_run_updates", True))
                        if enabled:
                            plan = await db.get(ExperimentPlan, run.experiment_plan_id)
                            note_id = str(plan.research_note_id) if plan else None

                            title = "Experiment run finished"
                            prio = "normal"
                            if run.status == "failed":
                                title = "Experiment run failed"
                                prio = "high"
                            elif run.status == "completed":
                                title = "Experiment run completed"
                            elif run.status == "cancelled":
                                title = "Experiment run cancelled"

                            notification_summary = _summarize_experiment_run_notification(
                                exp or {},
                                run.status,
                                launch_mode=str(getattr(job, "launch_mode", "") or "").strip() or None,
                            )
                            message = f"{run.name} · {notification_summary['message_suffix']}"
                            action_url = _build_experiment_run_notification_action_url(
                                note_id=note_id,
                                agent_job_id=str(run.agent_job_id) if run.agent_job_id else None,
                            )

                            await notification_service.create_notification(
                                db=db,
                                user_id=run.user_id,
                                notification_type=NotificationType.EXPERIMENT_RUN_UPDATE,
                                title=title,
                                message=message,
                                priority=prio,
                                related_entity_type="experiment_run",
                                related_entity_id=run.id,
                                action_url=action_url,
                                data={
                                    "experiment_run_id": str(run.id),
                                    "experiment_plan_id": str(run.experiment_plan_id),
                                    "agent_job_id": str(run.agent_job_id) if run.agent_job_id else None,
                                    "status": run.status,
                                    "note_id": note_id,
                                    **notification_summary["data"],
                                },
                            )
                    except Exception as exc:
                        logger.warning(f"Failed to notify experiment run {getattr(run, 'id', None)}: {exc}")
            except Exception as exc:
                logger.warning(f"Failed to sync experiment run {getattr(run, 'id', None)}: {exc}")

            if i % 50 == 0:
                try:
                    await db.commit()
                except Exception:
                    await db.rollback()

        try:
            await db.commit()
        except Exception:
            await db.rollback()

    return {
        "timestamp": now.isoformat(),
        "processed": processed,
        "updated": updated,
        "missing_job": missing_job,
    }


async def _async_emit_queue_urgency_alerts() -> Dict[str, Any]:
    """Create notifications for urgent queue items using the derived queue projection."""
    now = datetime.utcnow()
    processed_users = 0
    emitted = 0
    skipped = 0

    async with create_celery_session()() as db:
        user_rows = await db.execute(select(NotificationPreferences.user_id))
        user_ids = [row[0] for row in user_rows.fetchall()]

        for user_id in user_ids:
            try:
                prefs_result = await db.execute(
                    select(NotificationPreferences).where(NotificationPreferences.user_id == user_id)
                )
                prefs = prefs_result.scalar_one_or_none()
                if prefs is not None and not any(
                    bool(getattr(prefs, attr, True))
                    for attr in (
                        "notify_queue_urgency_alerts",
                        "notify_policy_guardrail_alerts",
                        "notify_autonomy_budget_alerts",
                        "notify_customer_autonomy_budget_alerts",
                    )
                ):
                    continue

                reminder_cooldown_hours = (
                    int(getattr(prefs, "queue_urgency_alert_reminder_cooldown_hours", 6) or 6)
                    if prefs is not None
                    else 6
                )

                jobs_result = await db.execute(
                    select(AgentJob).where(AgentJob.user_id == user_id).order_by(AgentJob.created_at.desc())
                )
                jobs = list(jobs_result.scalars().all())

                inbox_result = await db.execute(
                    select(ResearchInboxItem)
                    .where(
                        and_(
                            ResearchInboxItem.user_id == user_id,
                            ResearchInboxItem.status == "accepted",
                        )
                    )
                    .order_by(ResearchInboxItem.updated_at.desc(), ResearchInboxItem.discovered_at.desc())
                )
                inbox_items = list(inbox_result.scalars().all())

                analytics = research_monitor_profile_service.build_effectiveness_snapshot(
                    items=inbox_items,
                    jobs_by_id={job.id: job for job in jobs if job.id is not None},
                )
                for customer_row in analytics.get("customers", []) or []:
                    customer_name = str(customer_row.get("customer") or "").strip() or None
                    if not customer_name:
                        continue
                    customer_budget_snapshot = await research_monitor_profile_service.build_customer_budget_snapshot(
                        db=db,
                        user_id=user_id,
                        customer=customer_name,
                        now=now,
                    )
                    customer_row.update(customer_budget_snapshot)
                queue_items = _build_checkpoint_queue_items(
                    jobs,
                    inbox_items,
                    monitor_health_rows=analytics.get("monitors", []),
                )
                candidate_items = [
                    item for item in queue_items
                    if (
                        str(item.item_type or "") in {"approval_checkpoint", "job_recovery"}
                        and str(item.sla_bucket or "") in {"at_risk", "overdue"}
                    )
                    or str(item.item_type or "") in {"policy_review", "budget_review"}
                ]
                customer_budget_items = [
                    customer_row
                    for customer_row in (analytics.get("customers", []) or [])
                    if str(customer_row.get("customer_budget_throttle_state") or "").strip().lower() not in {"", "normal"}
                ]

                existing_result = await db.execute(
                    select(Notification)
                    .where(
                        and_(
                            Notification.user_id == user_id,
                            Notification.notification_type.in_(
                                [
                                    NotificationType.QUEUE_URGENCY_ALERT,
                                    NotificationType.POLICY_GUARDRAIL_ALERT,
                                    NotificationType.AUTONOMY_BUDGET_ALERT,
                                    NotificationType.CUSTOMER_AUTONOMY_BUDGET_ALERT,
                                ]
                            ),
                            Notification.is_dismissed == False,
                        )
                    )
                    .order_by(Notification.created_at.desc())
                    .limit(200)
                )
                existing_notifications = list(existing_result.scalars().all())

                for item in candidate_items:
                    item_payload = item.model_dump(mode="json")
                    item_type = str(item.item_type or "").strip()
                    notification_type = (
                        NotificationType.POLICY_GUARDRAIL_ALERT
                        if item_type == "policy_review"
                        else (
                            NotificationType.AUTONOMY_BUDGET_ALERT
                            if item_type == "budget_review"
                            else NotificationType.QUEUE_URGENCY_ALERT
                        )
                    )
                    if (
                        notification_type == NotificationType.POLICY_GUARDRAIL_ALERT
                        and prefs is not None
                        and not bool(getattr(prefs, "notify_policy_guardrail_alerts", True))
                    ):
                        continue
                    if (
                        notification_type == NotificationType.AUTONOMY_BUDGET_ALERT
                        and prefs is not None
                        and not bool(getattr(prefs, "notify_autonomy_budget_alerts", True))
                    ):
                        continue
                    if not _queue_alert_should_emit(
                        item=item_payload,
                        existing_notifications=[
                            notification
                            for notification in existing_notifications
                            if str(notification.notification_type or "").strip() == notification_type
                        ],
                        reminder_cooldown_hours=reminder_cooldown_hours,
                        now=now,
                    ):
                        skipped += 1
                        continue

                    if notification_type == NotificationType.POLICY_GUARDRAIL_ALERT:
                        summary = _summarize_policy_guardrail_notification(item_payload)
                        action_url = _build_policy_guardrail_notification_action_url(
                            job_id=str(item.job_id) if item.job_id else None,
                            history_entry_id=str(item.policy_guardrail_target_history_entry_id or "").strip() or None,
                        )
                    elif notification_type == NotificationType.AUTONOMY_BUDGET_ALERT:
                        summary = _summarize_budget_guardrail_notification(item_payload)
                        action_url = _build_budget_alert_notification_action_url(
                            job_id=str(item.job_id) if item.job_id else None,
                        )
                    else:
                        summary = _summarize_queue_urgency_notification(item_payload)
                        action_url = _build_queue_urgency_notification_action_url(
                            job_id=str(item.job_id) if item.job_id else None,
                            item_type=str(item.item_type or "").strip() or None,
                            sla_bucket=str(item.sla_bucket or "").strip() or None,
                        )

                    notification = await notification_service.create_notification(
                        db=db,
                        user_id=user_id,
                        notification_type=notification_type,
                        title=summary["title"],
                        message=summary["message"],
                        priority=summary["priority"],
                        related_entity_type="agent_job" if item.job_id else "research_inbox_item",
                        related_entity_id=item.job_id or item.inbox_item_id,
                        action_url=action_url,
                        data=summary["data"],
                    )
                    if notification is not None:
                        emitted += 1
                        existing_notifications.insert(0, notification)

                for customer_item in customer_budget_items:
                    if prefs is not None and not bool(getattr(prefs, "notify_customer_autonomy_budget_alerts", True)):
                        continue
                    if not _customer_budget_alert_should_emit(
                        item=customer_item,
                        existing_notifications=[
                            notification
                            for notification in existing_notifications
                            if str(notification.notification_type or "").strip() == NotificationType.CUSTOMER_AUTONOMY_BUDGET_ALERT
                        ],
                    ):
                        skipped += 1
                        continue
                    summary = _summarize_customer_budget_notification(customer_item)
                    notification = await notification_service.create_notification(
                        db=db,
                        user_id=user_id,
                        notification_type=NotificationType.CUSTOMER_AUTONOMY_BUDGET_ALERT,
                        title=summary["title"],
                        message=summary["message"],
                        priority=summary["priority"],
                        related_entity_type="customer_portfolio",
                        action_url=_build_customer_budget_alert_notification_action_url(
                            customer=str(customer_item.get("customer") or "").strip() or None,
                        ),
                        data=summary["data"],
                    )
                    if notification is not None:
                        emitted += 1
                        existing_notifications.insert(0, notification)
                processed_users += 1
            except Exception as exc:
                logger.warning(f"Failed to emit queue urgency alerts for user {user_id}: {exc}")

    return {
        "timestamp": now.isoformat(),
        "processed_users": processed_users,
        "emitted": emitted,
        "skipped": skipped,
    }
