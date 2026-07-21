"""Tests for deterministic job runner registry."""

from app.services.agent_deterministic_runner_registry import (
    DeterministicRunnerRegistry,
    FunctionDeterministicJobRunner,
    build_deterministic_runner_registry,
)


async def _noop_runner(*, job, db, progress_callback=None):
    return {"status": "completed", "job": getattr(job, "id", None)}


def test_registry_resolves_known_runner():
    registry = DeterministicRunnerRegistry(
        [FunctionDeterministicJobRunner(name="known", handler=_noop_runner)]
    )
    assert registry.resolve("known") is not None
    assert registry.resolve("missing") is None


def test_build_registry_contains_known_executor_runner():
    class _Executor:
        async def _run_ai_hub_scientist(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_research_inbox_monitor(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_research_engineer_scientist(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_domain_research_orchestrator(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_research_fleet_orchestrator(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_research_engineer_paper_update(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_swarm_fan_in_aggregate(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_latex_citation_sync(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_latex_reviewer_critic(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_latex_compile_project(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_latex_publish_project(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_latex_apply_unified_diff(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_experiment_loop_seed(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_experiment_plan_generate(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_experiment_decide_next(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_experiment_runner(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_experiment_persist_results(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_code_patch_proposer(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_code_patch_apply_to_kb(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_coding_backlog_orchestrator(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_arxiv_inbox_extract_repos(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_git_repo_ingest_wait(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_paper_algorithm_project(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

        async def _run_generated_project_demo_check(self, *, job, db, progress_callback=None):
            return {"status": "completed"}

    registry = build_deterministic_runner_registry(_Executor())
    assert registry.resolve("ai_hub_scientist") is not None
    assert registry.resolve("experiment_runner") is not None
    assert registry.resolve("generated_project_demo_check") is not None
