"""Registry for executor deterministic job runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Protocol


class DeterministicJobRunner(Protocol):
    @property
    def name(self) -> str: ...

    async def run(
        self,
        *,
        job: Any,
        db: Any,
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]: ...


@dataclass(slots=True)
class FunctionDeterministicJobRunner:
    name: str
    handler: Callable[..., Awaitable[Dict[str, Any]]]

    async def run(
        self,
        *,
        job: Any,
        db: Any,
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        return await self.handler(job=job, db=db, progress_callback=progress_callback)


class DeterministicRunnerRegistry:
    """Resolves deterministic job runners by config key."""

    def __init__(self, runners: Optional[Iterable[DeterministicJobRunner]] = None) -> None:
        self._runners: Dict[str, DeterministicJobRunner] = {}
        for runner in runners or []:
            self.register(runner)

    def register(self, runner: DeterministicJobRunner) -> None:
        self._runners[runner.name] = runner

    def resolve(self, name: Optional[str]) -> Optional[DeterministicJobRunner]:
        if not name:
            return None
        return self._runners.get(str(name).strip())

    async def try_execute(
        self,
        name: Optional[str],
        *,
        job: Any,
        db: Any,
        progress_callback: Optional[Callable[..., Any]] = None,
    ) -> tuple[bool, Optional[Dict[str, Any]]]:
        runner = self.resolve(name)
        if runner is None:
            return False, None
        return True, await runner.run(job=job, db=db, progress_callback=progress_callback)


def _resolve_handler(executor: Any, service_attr: str, service_method: str, fallback_method: str):
    service = getattr(executor, service_attr, None)
    method = getattr(service, service_method, None) if service is not None else None
    if callable(method):
        async def _handler(*, job: Any, db: Any, progress_callback: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
            return await method(executor, job=job, db=db, progress_callback=progress_callback)

        return _handler
    return getattr(executor, fallback_method)


def _build_research_orchestration_runners(executor: Any) -> list[DeterministicJobRunner]:
    return [
        FunctionDeterministicJobRunner(
            "ai_hub_scientist",
            _resolve_handler(executor, "research_runner_service", "run_ai_hub_scientist", "_run_ai_hub_scientist"),
        ),
        FunctionDeterministicJobRunner(
            "research_inbox_monitor",
            _resolve_handler(executor, "research_runner_service", "run_research_inbox_monitor", "_run_research_inbox_monitor"),
        ),
        FunctionDeterministicJobRunner(
            "research_engineer_scientist",
            _resolve_handler(executor, "research_runner_service", "run_research_engineer_scientist", "_run_research_engineer_scientist"),
        ),
        FunctionDeterministicJobRunner(
            "domain_research_orchestrator",
            _resolve_handler(executor, "research_runner_service", "run_domain_research_orchestrator", "_run_domain_research_orchestrator"),
        ),
        FunctionDeterministicJobRunner(
            "research_fleet_orchestrator",
            _resolve_handler(executor, "research_runner_service", "run_research_fleet_orchestrator", "_run_research_fleet_orchestrator"),
        ),
        FunctionDeterministicJobRunner(
            "research_engineer_paper_update",
            _resolve_handler(executor, "research_runner_service", "run_research_engineer_paper_update", "_run_research_engineer_paper_update"),
        ),
        FunctionDeterministicJobRunner(
            "swarm_fan_in_aggregate",
            _resolve_handler(executor, "research_runner_service", "run_swarm_fan_in_aggregate", "_run_swarm_fan_in_aggregate"),
        ),
    ]


def _build_latex_runners(executor: Any) -> list[DeterministicJobRunner]:
    return [
        FunctionDeterministicJobRunner(
            "latex_citation_sync",
            _resolve_handler(executor, "latex_runner_service", "run_latex_citation_sync", "_run_latex_citation_sync"),
        ),
        FunctionDeterministicJobRunner(
            "latex_reviewer_critic",
            _resolve_handler(executor, "latex_runner_service", "run_latex_reviewer_critic", "_run_latex_reviewer_critic"),
        ),
        FunctionDeterministicJobRunner(
            "latex_compile_project",
            _resolve_handler(executor, "latex_runner_service", "run_latex_compile_project", "_run_latex_compile_project"),
        ),
        FunctionDeterministicJobRunner(
            "latex_publish_project",
            _resolve_handler(executor, "latex_runner_service", "run_latex_publish_project", "_run_latex_publish_project"),
        ),
        FunctionDeterministicJobRunner(
            "latex_apply_unified_diff",
            _resolve_handler(executor, "latex_runner_service", "run_latex_apply_unified_diff", "_run_latex_apply_unified_diff"),
        ),
    ]


def _build_experiment_runners(executor: Any) -> list[DeterministicJobRunner]:
    return [
        FunctionDeterministicJobRunner(
            "experiment_loop_seed",
            _resolve_handler(executor, "experiment_runner_service", "run_experiment_loop_seed", "_run_experiment_loop_seed"),
        ),
        FunctionDeterministicJobRunner(
            "experiment_plan_generate",
            _resolve_handler(executor, "experiment_runner_service", "run_experiment_plan_generate", "_run_experiment_plan_generate"),
        ),
        FunctionDeterministicJobRunner(
            "experiment_decide_next",
            _resolve_handler(executor, "experiment_runner_service", "run_experiment_decide_next", "_run_experiment_decide_next"),
        ),
        FunctionDeterministicJobRunner(
            "experiment_runner",
            _resolve_handler(executor, "experiment_runner_service", "run_experiment_runner", "_run_experiment_runner"),
        ),
        FunctionDeterministicJobRunner(
            "experiment_persist_results",
            _resolve_handler(executor, "experiment_runner_service", "run_experiment_persist_results", "_run_experiment_persist_results"),
        ),
    ]


def _build_coding_runners(executor: Any) -> list[DeterministicJobRunner]:
    return [
        FunctionDeterministicJobRunner(
            "code_patch_proposer",
            _resolve_handler(executor, "coding_runner_service", "run_code_patch_proposer", "_run_code_patch_proposer"),
        ),
        FunctionDeterministicJobRunner(
            "code_patch_apply_to_kb",
            _resolve_handler(executor, "coding_runner_service", "run_code_patch_apply_to_kb", "_run_code_patch_apply_to_kb"),
        ),
        FunctionDeterministicJobRunner(
            "coding_backlog_orchestrator",
            _resolve_handler(executor, "coding_runner_service", "run_coding_backlog_orchestrator", "_run_coding_backlog_orchestrator"),
        ),
    ]


def _build_ingestion_demo_runners(executor: Any) -> list[DeterministicJobRunner]:
    return [
        FunctionDeterministicJobRunner(
            "arxiv_inbox_extract_repos",
            _resolve_handler(executor, "ingestion_demo_runner_service", "run_arxiv_inbox_extract_repos", "_run_arxiv_inbox_extract_repos"),
        ),
        FunctionDeterministicJobRunner(
            "git_repo_ingest_wait",
            _resolve_handler(executor, "ingestion_demo_runner_service", "run_git_repo_ingest_wait", "_run_git_repo_ingest_wait"),
        ),
        FunctionDeterministicJobRunner(
            "paper_algorithm_project",
            _resolve_handler(executor, "ingestion_demo_runner_service", "run_paper_algorithm_project", "_run_paper_algorithm_project"),
        ),
        FunctionDeterministicJobRunner(
            "generated_project_demo_check",
            _resolve_handler(executor, "ingestion_demo_runner_service", "run_generated_project_demo_check", "_run_generated_project_demo_check"),
        ),
    ]


def build_deterministic_runner_registry(executor: Any) -> DeterministicRunnerRegistry:
    """Build the deterministic runner registry for AutonomousAgentExecutor."""
    return DeterministicRunnerRegistry(
        [
            *_build_research_orchestration_runners(executor),
            *_build_latex_runners(executor),
            *_build_experiment_runners(executor),
            *_build_coding_runners(executor),
            *_build_ingestion_demo_runners(executor),
        ]
    )
