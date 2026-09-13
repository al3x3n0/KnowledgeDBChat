from uuid import UUID

from app.schemas.domain_research_profile import DomainResearchProfileCreate


def test_domain_research_profile_create_normalizes_scope_and_queries():
    payload = DomainResearchProfileCreate(
        title=" Retrieval Monitor ",
        domain=" Retrieval ",
        objective=" Track new evidence ",
        source_scope="repo",
        track_type=" microarch ",
        research_mode=" literature to hypothesis ",
        monitor_queries="retrieval latency\nretrieval evals",
        repo_source_ids="11111111-1111-1111-1111-111111111111\n22222222-2222-2222-2222-222222222222",
        benchmark_queries="ipc stall\nbranch miss",
        sandbox_profile_id=" scientific-microarchitecture-sandbox ",
        validation_policy={
            "confidence_threshold": 0.81,
            "max_auto_follow_up_launches": 3,
        },
    )

    assert payload.title == "Retrieval Monitor"
    assert payload.domain == "Retrieval"
    assert payload.objective == "Track new evidence"
    assert payload.source_scope == "kb_plus_arxiv_plus_repo"
    assert payload.track_type == "microarchitecture"
    assert payload.research_mode == "literature_to_hypothesis"
    assert payload.monitor_queries == ["retrieval latency", "retrieval evals"]
    assert payload.repo_source_ids == [
        UUID("11111111-1111-1111-1111-111111111111"),
        UUID("22222222-2222-2222-2222-222222222222"),
    ]
    assert payload.benchmark_queries == ["ipc stall", "branch miss"]
    assert payload.sandbox_profile_id == "scientific-microarchitecture-sandbox"
    assert payload.validation_policy == {
        "confidence_threshold": 0.81,
        "experiment_readiness_threshold": 0.8,
        "max_auto_follow_up_launches": 3,
        "auto_create_experiment_plans": True,
        "auto_launch_follow_up": True,
        "auto_execute_validation_runs": False,
        "max_concurrent_validation_runs": 1,
        "max_validation_runtime_minutes": 20,
        "max_validation_budget_per_run": 25.0,
        "validation_backoff_policy": {
            "max_consecutive_failures": 2,
            "cooldown_minutes": 180,
        },
    }
