"""Authenticated evaluation of persisted autonomous R&D job trajectories."""

from collections import defaultdict
from typing import Dict, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.autonomous_rnd_verification_audit_snapshot import (
    AutonomousRndVerificationAuditSnapshot,
)
from app.models.experiment import ExperimentRun
from app.models.user import User
from app.schemas.autonomous_rnd_eval import (
    AutonomousRnDEvalGradeJobsRequest,
    AutonomousRnDEvalGradeJobsResponse,
    AutonomousRnDEvalSuiteListResponse,
    AutonomousRnDJobOutcomeResponse,
    AutonomousRnDVerificationAuditEnvelope,
    AutonomousRnDVerificationAuditKeyListResponse,
    AutonomousRnDVerificationAuditSnapshotListResponse,
    AutonomousRnDVerificationAuditSnapshotRequest,
    AutonomousRnDVerificationAuditVerifyResponse,
    AutonomousRnDVerificationLaunchRequest,
    AutonomousRnDVerificationLaunchResponse,
)
from app.services.autonomous_rnd_eval_service import (
    EvalDefinitionError,
    autonomous_rnd_eval_harness,
)
from app.services.autonomous_rnd_trajectory_service import (
    autonomous_rnd_trajectory_adapter,
)
from app.services.autonomous_rnd_verification_audit_service import (
    autonomous_rnd_verification_audit_service,
)
from app.services.autonomous_rnd_verification_launch_service import (
    VerificationLaunchError,
    autonomous_rnd_verification_launch_service,
)
from app.services.autonomous_rnd_verification_status_service import (
    autonomous_rnd_verification_status_service,
)

router = APIRouter()


@router.get("/suites", response_model=AutonomousRnDEvalSuiteListResponse)
async def list_autonomous_rnd_eval_suites(
    current_user: User = Depends(get_current_active_user),
):
    del current_user
    return {"suites": autonomous_rnd_eval_harness.list_builtin_suites()}


@router.get(
    "/jobs/{job_id}/outcome",
    response_model=AutonomousRnDJobOutcomeResponse,
)
async def get_autonomous_rnd_job_outcome(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    job = (
        await db.execute(
            select(AgentJob).where(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Agent job was not found")
    outcome = autonomous_rnd_trajectory_adapter.build_outcome(job)
    lifecycle = await autonomous_rnd_verification_status_service.build(
        parent_job=job,
        outcome=outcome,
        db=db,
    )
    return {
        "job_id": job.id,
        "job_status": str(job.status or ""),
        "outcome": outcome,
        "verification_lifecycle": lifecycle,
    }


@router.post(
    "/jobs/{job_id}/verification-audit-snapshot",
    response_model=AutonomousRnDVerificationAuditEnvelope,
)
async def create_autonomous_rnd_verification_audit_snapshot(
    job_id: UUID,
    request: AutonomousRnDVerificationAuditSnapshotRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    job = (
        await db.execute(
            select(AgentJob).where(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Agent job was not found")
    outcome = autonomous_rnd_trajectory_adapter.build_outcome(job)
    lifecycle = await autonomous_rnd_verification_status_service.build(
        parent_job=job,
        outcome=outcome,
        db=db,
    )
    registry_id = uuid4()
    envelope = autonomous_rnd_verification_audit_service.build_signed_snapshot(
        registry_id=registry_id,
        parent_job=job,
        lifecycle=lifecycle,
        task_id=request.task_id,
        status=request.status,
    )
    snapshot = envelope["snapshot"]
    integrity = envelope["integrity"]
    conflicting_key = (
        await db.execute(
            select(AutonomousRndVerificationAuditSnapshot.id)
            .where(
                AutonomousRndVerificationAuditSnapshot.key_id == integrity["key_id"],
                AutonomousRndVerificationAuditSnapshot.public_key
                != integrity["public_key"],
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if conflicting_key is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Audit signing key id is already bound to a different public key; "
                "rotate AUTONOMOUS_RND_AUDIT_SIGNING_KEY_ID"
            ),
        )
    db.add(
        AutonomousRndVerificationAuditSnapshot(
            id=registry_id,
            user_id=current_user.id,
            job_id=job.id,
            schema_version=int(snapshot["schema_version"]),
            snapshot=snapshot,
            canonicalization=integrity["canonicalization"],
            sha256=integrity["sha256"],
            signature_algorithm=integrity["signature_algorithm"],
            signature_encoding=integrity["signature_encoding"],
            signature=integrity["signature"],
            key_id=integrity["key_id"],
            public_key=integrity["public_key"],
        )
    )
    await db.commit()
    return envelope


@router.get(
    "/jobs/{job_id}/verification-audit-snapshots",
    response_model=AutonomousRnDVerificationAuditSnapshotListResponse,
)
async def list_autonomous_rnd_verification_audit_snapshots(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    owned_job = (
        await db.execute(
            select(AgentJob.id).where(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if owned_job is None:
        raise HTTPException(status_code=404, detail="Agent job was not found")
    records = (
        (
            await db.execute(
                select(AutonomousRndVerificationAuditSnapshot)
                .where(
                    AutonomousRndVerificationAuditSnapshot.job_id == job_id,
                    AutonomousRndVerificationAuditSnapshot.user_id == current_user.id,
                )
                .order_by(
                    AutonomousRndVerificationAuditSnapshot.created_at.desc(),
                    AutonomousRndVerificationAuditSnapshot.id.desc(),
                )
                .limit(100)
            )
        )
        .scalars()
        .all()
    )
    return {
        "items": [
            {
                "registry_id": record.id,
                "job_id": record.job_id,
                "created_at": record.created_at,
                "filters": dict(record.snapshot.get("filters") or {}),
                "sha256": record.sha256,
                "key_id": record.key_id,
                "signature_algorithm": record.signature_algorithm,
            }
            for record in records
        ]
    }


@router.get(
    "/verification-audit-snapshots/{registry_id}",
    response_model=AutonomousRnDVerificationAuditEnvelope,
)
async def get_autonomous_rnd_verification_audit_snapshot(
    registry_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    record = (
        await db.execute(
            select(AutonomousRndVerificationAuditSnapshot).where(
                AutonomousRndVerificationAuditSnapshot.id == registry_id,
                AutonomousRndVerificationAuditSnapshot.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Audit snapshot was not found")
    return _audit_snapshot_envelope(record)


@router.get(
    "/verification-audit-keys",
    response_model=AutonomousRnDVerificationAuditKeyListResponse,
)
async def list_autonomous_rnd_verification_audit_keys(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    active = autonomous_rnd_verification_audit_service.active_public_key()
    rows = (
        await db.execute(
            select(
                AutonomousRndVerificationAuditSnapshot.key_id,
                AutonomousRndVerificationAuditSnapshot.public_key,
            )
            .where(AutonomousRndVerificationAuditSnapshot.user_id == current_user.id)
            .distinct()
        )
    ).all()
    keys = {(active["key_id"], active["public_key"]): active}
    for key_id, public_key in rows:
        keys.setdefault(
            (key_id, public_key),
            autonomous_rnd_verification_audit_service.public_key_metadata(
                key_id=key_id,
                public_key=public_key,
                status="retired",
            ),
        )
    return {"keys": list(keys.values())}


@router.post(
    "/verification-audit-snapshots/verify",
    response_model=AutonomousRnDVerificationAuditVerifyResponse,
)
async def verify_autonomous_rnd_verification_audit_snapshot(
    request: AutonomousRnDVerificationAuditEnvelope,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    if (
        len(
            autonomous_rnd_verification_audit_service.canonical_bytes(
                request.model_dump()
            )
        )
        > 1_000_000
    ):
        raise HTTPException(status_code=413, detail="Audit snapshot is too large")
    result = autonomous_rnd_verification_audit_service.verify_envelope(
        request.model_dump()
    )
    if not result.get("valid"):
        return result
    try:
        registry_id = UUID(str(result.get("registry_id") or ""))
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Snapshot registry id is invalid"
        ) from exc
    record = (
        await db.execute(
            select(AutonomousRndVerificationAuditSnapshot).where(
                AutonomousRndVerificationAuditSnapshot.id == registry_id,
                AutonomousRndVerificationAuditSnapshot.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Audit snapshot was not found")
    trusted_result = autonomous_rnd_verification_audit_service.verify_envelope(
        request.model_dump(),
        trusted_public_keys={record.key_id: record.public_key},
    )
    if not trusted_result.get("valid"):
        return trusted_result
    if request.model_dump() != _audit_snapshot_envelope(record):
        return {"valid": False, "reason": "registry_mismatch"}
    return trusted_result


def _audit_snapshot_envelope(
    record: AutonomousRndVerificationAuditSnapshot,
) -> Dict[str, Dict]:
    return {
        "snapshot": dict(record.snapshot),
        "integrity": {
            "canonicalization": record.canonicalization,
            "sha256": record.sha256,
            "signature_algorithm": record.signature_algorithm,
            "signature_encoding": record.signature_encoding,
            "signature": record.signature,
            "key_id": record.key_id,
            "public_key": record.public_key,
        },
    }


@router.post(
    "/grade-jobs",
    response_model=AutonomousRnDEvalGradeJobsResponse,
)
async def grade_autonomous_rnd_jobs(
    request: AutonomousRnDEvalGradeJobsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        suite = autonomous_rnd_eval_harness.load_builtin_suite(request.suite_id)
    except EvalDefinitionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    task_ids = {task.id for task in suite.tasks}
    unknown_tasks = sorted(
        binding.task_id for binding in request.trials if binding.task_id not in task_ids
    )
    if unknown_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task ids for suite '{suite.id}': {unknown_tasks}",
        )

    requested_job_ids = {
        job_id for binding in request.trials for job_id in binding.job_ids
    }
    jobs = list(
        (
            await db.execute(
                select(AgentJob).where(
                    AgentJob.id.in_(requested_job_ids),
                    AgentJob.user_id == current_user.id,
                )
            )
        )
        .scalars()
        .all()
    )
    jobs_by_id = {job.id: job for job in jobs}
    if len(jobs_by_id) != len(requested_job_ids):
        raise HTTPException(
            status_code=404,
            detail="One or more agent jobs were not found",
        )

    experiment_runs = list(
        (
            await db.execute(
                select(ExperimentRun).where(
                    ExperimentRun.agent_job_id.in_(requested_job_ids),
                    ExperimentRun.user_id == current_user.id,
                )
            )
        )
        .scalars()
        .all()
    )
    runs_by_job: Dict[object, List[ExperimentRun]] = defaultdict(list)
    for run in experiment_runs:
        runs_by_job[run.agent_job_id].append(run)

    outcomes = {}
    task_bindings = {}
    for binding in request.trials:
        task_bindings[binding.task_id] = [str(job_id) for job_id in binding.job_ids]
        outcomes[binding.task_id] = [
            autonomous_rnd_trajectory_adapter.build_outcome(
                jobs_by_id[job_id],
                experiment_runs=runs_by_job.get(job_id, []),
            )
            for job_id in binding.job_ids
        ]

    report = autonomous_rnd_eval_harness.grade_suite_outcomes(suite, outcomes)
    return {
        "report": report,
        "evaluated_job_count": len(requested_job_ids),
        "task_bindings": task_bindings,
    }


@router.post(
    "/jobs/{job_id}/verification-tasks/{task_id}/launch",
    response_model=AutonomousRnDVerificationLaunchResponse,
)
async def launch_autonomous_rnd_verification_task(
    job_id: UUID,
    task_id: str,
    request: AutonomousRnDVerificationLaunchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    parent_job = (
        await db.execute(
            select(AgentJob).where(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if parent_job is None:
        raise HTTPException(status_code=404, detail="Parent agent job was not found")

    outcome = autonomous_rnd_trajectory_adapter.build_outcome(parent_job)
    plan = (
        outcome.get("verification_plan")
        if isinstance(outcome.get("verification_plan"), dict)
        else {}
    )
    task = next(
        (
            item
            for item in plan.get("tasks", [])
            if isinstance(item, dict) and str(item.get("id") or "") == task_id
        ),
        None,
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Verification task was not found")

    try:
        launch = await autonomous_rnd_verification_launch_service.launch(
            parent_job=parent_job,
            task=task,
            current_user=current_user,
            db=db,
            research_note_id=request.research_note_id,
            source_id=request.source_id,
            sandbox_profile_id=request.sandbox_profile_id,
            commands=request.commands,
            repeat_count=request.repeat_count,
            timeout_seconds=request.timeout_seconds,
            max_runtime_minutes=request.max_runtime_minutes,
            budget_limit=request.budget_limit,
            approval_note=request.approval_note,
        )
    except VerificationLaunchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    queued = False
    if request.start_immediately and not launch.job.celery_task_id:
        from app.tasks.agent_job_tasks import execute_agent_job_task

        task_result = execute_agent_job_task.delay(
            str(launch.job.id), str(current_user.id)
        )
        launch.job.celery_task_id = str(
            getattr(task_result, "id", "") or f"queued:{launch.job.id}"
        )
        launch.run.status = "queued"
        await db.commit()
        queued = True

    return {
        "created": launch.created,
        "queued": queued,
        "experiment_plan_id": launch.plan.id,
        "experiment_run_id": launch.run.id,
        "agent_job_id": launch.job.id,
        "audit_id": launch.audit.id,
        "status": launch.run.status,
        "budget": {
            "repeat_count": request.repeat_count,
            "timeout_seconds": request.timeout_seconds,
            "max_runtime_minutes": request.max_runtime_minutes,
            "budget_limit": request.budget_limit,
        },
    }
