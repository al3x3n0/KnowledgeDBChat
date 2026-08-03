"""HTTP composition for autonomous-job quick-start workflows."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.document import Document, DocumentSource
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobFromTemplate,
    AgentJobQuickStartBugTriageSwarmRequest,
    AgentJobQuickStartBuildBreakSwarmRequest,
    AgentJobQuickStartClaudeBackendRequest,
    AgentJobQuickStartDomainResearchRequest,
    AgentJobQuickStartFrontendRegressionSwarmRequest,
    AgentJobQuickStartRepoBugTriageRequest,
    AgentJobQuickStartRoleWorkflowRequest,
    AgentJobResponse,
)
from app.services.agent_coding_swarm_launch_service import (
    AgentCodingSwarmLaunchError,
    agent_coding_swarm_launch_service,
)
from app.services.agent_job_creation_service import (
    AgentJobSpec,
    agent_job_creation_service,
)
from app.services.agent_job_templates import (
    CLAUDE_CODE_BACKEND_TEMPLATE_ID,
    DOMAIN_RESEARCH_TEMPLATE_ID,
    REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
)

JobSerializer = Callable[..., AgentJobResponse]
TemplateLauncher = Callable[..., Awaitable[AgentJobResponse]]
RequestBuilder = Callable[..., Any]


@dataclass(frozen=True)
class QuickStartBuilders:
    """Legacy application builders injected during incremental extraction."""

    claude_backend_config: RequestBuilder
    domain_research_config: RequestBuilder
    domain_research_goal: RequestBuilder
    repo_bug_triage_config: RequestBuilder
    repo_bug_triage_goal: RequestBuilder
    role_workflow_config: RequestBuilder


@dataclass(frozen=True)
class QuickStartApi:
    """Composed router plus compatibility-callable endpoint handlers."""

    router: APIRouter
    quick_start_claude_backend_job: Callable[..., Any]
    quick_start_domain_research_job: Callable[..., Any]
    quick_start_repo_bug_triage_job: Callable[..., Any]
    quick_start_bug_triage_swarm_job: Callable[..., Any]
    quick_start_build_break_swarm_job: Callable[..., Any]
    quick_start_frontend_regression_swarm_job: Callable[..., Any]
    create_quick_start_coding_swarm_job: Callable[..., Any]
    quick_start_role_workflow_job: Callable[..., Any]


async def _load_repository_source(
    *,
    source_id: Any,
    db: AsyncSession,
    current_user: User,
) -> tuple[DocumentSource, str]:
    source = await db.get(DocumentSource, source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document source not found",
        )
    source_type = str(source.source_type or "").strip().lower()
    if source_type not in {"github", "gitlab"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Quick start requires a github/gitlab document source",
        )
    if (
        not current_user.is_admin()
        and not agent_coding_swarm_launch_service.is_source_owned_by_user(
            source, current_user
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for this source",
        )
    document_count = int(
        (
            await db.execute(
                select(func.count()).where(Document.source_id == source.id)
            )
        ).scalar()
        or 0
    )
    if document_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Source has no documents; ingest/sync the repository first",
        )
    return source, source_type


def _reject_unsafe_commands(commands: Any) -> None:
    unsafe_commands = agent_job_creation_service.find_unsafe_commands(commands)
    if unsafe_commands:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Quick start rejected potentially destructive command(s)",
                "blocked_commands": unsafe_commands,
            },
        )


def build_quick_start_api(
    *,
    builders: QuickStartBuilders,
    create_job_from_template: TemplateLauncher,
    job_serializer: JobSerializer,
    execute_job_task: Any,
) -> QuickStartApi:
    """Build quick-start routes without importing the legacy endpoint module."""
    router = APIRouter()

    @router.post(
        "/quick-start/claude-backend",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def quick_start_claude_backend_job(
        request: AgentJobQuickStartClaudeBackendRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        source, source_type = await _load_repository_source(
            source_id=request.source_id,
            db=db,
            current_user=current_user,
        )
        _reject_unsafe_commands(request.commands)
        merged_config = builders.claude_backend_config(
            request,
            source_name=str(source.name or ""),
            source_type=source_type,
        )
        job_name = str(request.name or "").strip() or (
            f"Claude Backend Loop - {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        template_request = AgentJobFromTemplate(
            template_id=CLAUDE_CODE_BACKEND_TEMPLATE_ID,
            name=job_name,
            goal=request.goal,
            config=merged_config,
            start_immediately=bool(request.start_immediately),
        )
        return await create_job_from_template(template_request, db, current_user)

    @router.post(
        "/quick-start/domain-research",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def quick_start_domain_research_job(
        request: AgentJobQuickStartDomainResearchRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        job_name = str(request.name or "").strip() or (
            f"Domain Research - {str(request.domain or '').strip()[:80]}"
        )
        template_request = AgentJobFromTemplate(
            template_id=DOMAIN_RESEARCH_TEMPLATE_ID,
            name=job_name,
            goal=builders.domain_research_goal(request),
            config=builders.domain_research_config(request),
            start_immediately=bool(request.start_immediately),
        )
        return await create_job_from_template(template_request, db, current_user)

    @router.post(
        "/quick-start/repo-bug-triage",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def quick_start_repo_bug_triage_job(
        request: AgentJobQuickStartRepoBugTriageRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        source, source_type = await _load_repository_source(
            source_id=request.source_id,
            db=db,
            current_user=current_user,
        )
        _reject_unsafe_commands(request.commands)
        merged_config = builders.repo_bug_triage_config(
            request,
            source_name=str(source.name or ""),
            source_type=source_type,
        )
        job_name = str(request.name or "").strip() or (
            f"Repo Bug Triage - {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        template_request = AgentJobFromTemplate(
            template_id=REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
            name=job_name,
            goal=builders.repo_bug_triage_goal(request),
            config=merged_config,
            start_immediately=bool(request.start_immediately),
        )
        return await create_job_from_template(template_request, db, current_user)

    async def create_quick_start_coding_swarm_job(
        *,
        request: AgentJobQuickStartBugTriageSwarmRequest
        | AgentJobQuickStartBuildBreakSwarmRequest
        | AgentJobQuickStartFrontendRegressionSwarmRequest,
        db: AsyncSession,
        current_user: User,
        preset_key: str,
    ):
        try:
            job = await agent_coding_swarm_launch_service.launch(
                request=request,
                db=db,
                current_user=current_user,
                preset_key=preset_key,
            )
        except AgentCodingSwarmLaunchError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.detail,
            ) from error
        if request.start_immediately:
            execute_job_task.delay(str(job.id), str(current_user.id))
        return job_serializer(job)

    @router.post(
        "/quick-start/bug-triage-swarm",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def quick_start_bug_triage_swarm_job(
        request: AgentJobQuickStartBugTriageSwarmRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        return await create_quick_start_coding_swarm_job(
            request=request,
            db=db,
            current_user=current_user,
            preset_key="bug_triage_swarm",
        )

    @router.post(
        "/quick-start/build-break-swarm",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def quick_start_build_break_swarm_job(
        request: AgentJobQuickStartBuildBreakSwarmRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        return await create_quick_start_coding_swarm_job(
            request=request,
            db=db,
            current_user=current_user,
            preset_key="build_break_swarm",
        )

    @router.post(
        "/quick-start/frontend-regression-swarm",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def quick_start_frontend_regression_swarm_job(
        request: AgentJobQuickStartFrontendRegressionSwarmRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        return await create_quick_start_coding_swarm_job(
            request=request,
            db=db,
            current_user=current_user,
            preset_key="frontend_regression_swarm",
        )

    @router.post(
        "/quick-start/role-workflow",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def quick_start_role_workflow_job(
        request: AgentJobQuickStartRoleWorkflowRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        job_name = str(request.name or "").strip() or (
            f"Role Workflow - {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        job = await agent_job_creation_service.create(
            spec=AgentJobSpec(
                name=job_name,
                description=(
                    "Role-based multi-agent plan-and-execute workflow (quick start)."
                ),
                job_type="research",
                goal=str(request.goal or "").strip(),
                config=builders.role_workflow_config(request),
                default_enable_memory=True,
                max_iterations=120,
                max_tool_calls=700,
                max_llm_calls=260,
                max_runtime_minutes=120,
            ),
            user_id=current_user.id,
            db=db,
        )
        if request.start_immediately:
            execute_job_task.delay(str(job.id), str(current_user.id))
        return job_serializer(job)

    return QuickStartApi(
        router=router,
        quick_start_claude_backend_job=quick_start_claude_backend_job,
        quick_start_domain_research_job=quick_start_domain_research_job,
        quick_start_repo_bug_triage_job=quick_start_repo_bug_triage_job,
        quick_start_bug_triage_swarm_job=quick_start_bug_triage_swarm_job,
        quick_start_build_break_swarm_job=quick_start_build_break_swarm_job,
        quick_start_frontend_regression_swarm_job=(
            quick_start_frontend_regression_swarm_job
        ),
        create_quick_start_coding_swarm_job=create_quick_start_coding_swarm_job,
        quick_start_role_workflow_job=quick_start_role_workflow_job,
    )
