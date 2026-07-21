from datetime import datetime
from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.job_results_exporter import JobResultsExporter


def _make_job(*, results=None) -> AgentJob:
    return AgentJob(
        id=uuid4(),
        name="Exporter Memory Test",
        goal="Validate export metadata contract",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=5,
        max_iterations=10,
        tool_calls_used=11,
        max_tool_calls=50,
        llm_calls_used=7,
        max_llm_calls=30,
        created_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        results=results or {},
    )


def test_get_memory_extraction_summary_reads_execution_strategy_payload():
    exporter = JobResultsExporter()
    root_id = str(uuid4())
    job = _make_job(
        results={
            "execution_strategy": {
                "memory_persistence": {
                    "extraction": {
                        "status": "completed",
                        "created_count": 3,
                        "skipped_duplicates": 2,
                        "parsed_count": 5,
                        "candidate_count": 4,
                        "is_relaunch_chain": True,
                        "relaunch_root_job_id": root_id,
                    }
                }
            }
        }
    )

    summary = exporter._get_memory_extraction_summary(job)

    assert summary["status"] == "completed"
    assert summary["created_count"] == 3
    assert summary["skipped_duplicates"] == 2
    assert summary["parsed_count"] == 5
    assert summary["candidate_count"] == 4
    assert summary["is_relaunch_chain"] is True
    assert summary["relaunch_root_job_id"] == root_id


def test_build_document_content_includes_memory_extraction_metadata_rows():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "summary": "Completed successfully",
            "execution_strategy": {
                "memory_persistence": {
                    "extraction": {
                        "status": "completed",
                        "created_count": 2,
                        "skipped_duplicates": 1,
                        "parsed_count": 3,
                    }
                }
            },
        }
    )

    content = exporter._build_document_content(job, include_log=False, include_metadata=True)
    tables = [item for item in content if item.get("type") == "table"]
    assert tables
    rows = tables[0].get("rows") or []
    row_map = {str(r[0]): str(r[1]) for r in rows if isinstance(r, list) and len(r) >= 2}
    assert row_map.get("Memory Extraction") == "COMPLETED"
    assert row_map.get("Memories Created") == "2"
    assert row_map.get("Duplicates Skipped") == "1"


def test_build_presentation_outline_includes_memory_extraction_stats_line():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "execution_strategy": {
                "memory_persistence": {
                    "extraction": {
                        "status": "completed",
                        "created_count": 4,
                        "skipped_duplicates": 3,
                    }
                }
            }
        }
    )

    outline = exporter._build_presentation_outline(job, include_log=False, include_metadata=True)
    stats_slide = next((slide for slide in outline.slides if slide.title == "Job Statistics"), None)
    assert stats_slide is not None
    assert any("Memory extraction: COMPLETED" in str(line) for line in (stats_slide.content or []))


def test_get_operator_intervention_summary_reads_execution_strategy_payload():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "pause",
                        "actor_user_id": "user-1",
                        "at": "2026-03-10T00:00:00Z",
                        "job_status_before": "running",
                        "job_status_after": "paused",
                    },
                    {
                        "action": "restart",
                        "actor_user_id": "user-1",
                        "at": "2026-03-10T01:00:00Z",
                        "note": "Retry after fallback failure",
                        "job_status_before": "failed",
                        "job_status_after": "pending",
                    },
                ]
            }
        }
    )

    summary = exporter._get_operator_intervention_summary(job)

    assert summary["count"] == 2
    assert summary["latest_action"] == "restart"
    assert summary["latest_status_before"] == "failed"
    assert summary["latest_status_after"] == "pending"
    assert summary["latest_note"] == "Retry after fallback failure"
    assert summary["latest_outcome"] == "resolved"
    assert summary["latest_outcome_reason"] == "Job completed after intervention"
    assert summary["latest_actor_user_id"] == "user-1"
    assert summary["latest_at"] == "2026-03-10T01:00:00Z"
    assert summary["recent_items"] == [
        "pause (running -> paused) [superseded]",
        "restart (failed -> pending): Retry after fallback failure [resolved]",
    ]


def test_get_experiment_run_summary_reads_recovery_guidance():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "experiment_run": {
                "final_phase": "fallback",
                "bootstrap_attempted": True,
                "bootstrap_ok": True,
                "fallback_attempted": True,
                "fallback_ok": False,
                "source_name": "Knowledge Repo",
                "source_id": "repo-7",
                "verification_commands": ["pytest -q"],
                "failed_commands": ["pytest -q"],
            },
            "execution_strategy": {
                "execution_graph": {
                    "graph_health": {"reasons": ["fallback verification still failing"]},
                    "recommended_actions": ["Inspect failing fallback output"],
                }
            },
        }
    )

    summary = exporter._get_experiment_run_summary(job)

    assert summary["final_phase"] == "fallback"
    assert summary["source_name"] == "Knowledge Repo"
    assert summary["source_id"] == "repo-7"
    assert summary["failed_command_count"] == 1
    assert summary["verification_command_count"] == 1
    assert summary["recovery_open"] is True
    assert summary["reason"] == "fallback verification still failing"
    assert summary["recommended_action"] == "Inspect failing fallback output"


def test_build_document_content_includes_experiment_recovery_metadata_and_stats():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "summary": "Fallback recovery remains open.",
            "experiment_run": {
                "final_phase": "fallback",
                "bootstrap_attempted": True,
                "bootstrap_ok": True,
                "fallback_attempted": True,
                "fallback_ok": False,
                "source_name": "Knowledge Repo",
                "source_id": "repo-8",
                "verification_commands": ["pytest -q"],
                "failed_commands": ["pytest -q"],
            },
            "execution_strategy": {
                "execution_graph": {
                    "graph_health": {"reasons": ["fallback verification still failing"]},
                    "recommended_actions": ["Inspect failing fallback output"],
                }
            },
        }
    )

    content = exporter._build_document_content(job, include_log=False, include_metadata=True)
    tables = [item for item in content if item.get("type") == "table"]
    assert tables
    rows = tables[0].get("rows") or []
    row_map = {str(r[0]): str(r[1]) for r in rows if isinstance(r, list) and len(r) >= 2}
    assert row_map.get("Experiment Final Phase") == "fallback"
    assert row_map.get("Experiment Source") == "Knowledge Repo"
    assert row_map.get("Experiment Source ID") == "repo-8"
    assert row_map.get("Experiment Bootstrap") == "OK"
    assert row_map.get("Experiment Fallback") == "ATTEMPTED"
    assert row_map.get("Experiment Recovery") == "OPEN"
    assert row_map.get("Recovery Reason") == "fallback verification still failing"
    assert row_map.get("Recovery Next Action") == "Inspect failing fallback output"

    bullet_lists = [item for item in content if item.get("type") == "bullet_list"]
    stats_items = [str(line) for item in bullet_lists for line in (item.get("items") or [])]
    assert any("Experiment final phase: fallback" in line for line in stats_items)
    assert any("Experiment recovery: OPEN" in line for line in stats_items)
    assert any("Recovery reason: fallback verification still failing" in line for line in stats_items)
    assert any("Recovery next action: Inspect failing fallback output" in line for line in stats_items)


def test_build_document_content_includes_operator_intervention_metadata_and_stats():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "summary": "Operator restarted the job after failure.",
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "restart",
                        "actor_user_id": "user-1",
                        "at": "2026-03-10T01:00:00Z",
                        "note": "Retry after fallback failure",
                        "job_status_before": "failed",
                        "job_status_after": "pending",
                    }
                ]
            },
        }
    )

    content = exporter._build_document_content(job, include_log=False, include_metadata=True)
    tables = [item for item in content if item.get("type") == "table"]
    assert tables
    rows = tables[0].get("rows") or []
    row_map = {str(r[0]): str(r[1]) for r in rows if isinstance(r, list) and len(r) >= 2}
    assert row_map.get("Operator Interventions") == "1"
    assert row_map.get("Latest Intervention") == "restart (failed -> pending)"
    assert row_map.get("Latest Intervention Outcome") == "RESOLVED"
    assert row_map.get("Latest Intervention Outcome Reason") == "Job completed after intervention"
    assert row_map.get("Latest Intervention Note") == "Retry after fallback failure"

    bullet_lists = [item for item in content if item.get("type") == "bullet_list"]
    stats_items = [str(line) for item in bullet_lists for line in (item.get("items") or [])]
    assert any("Operator interventions: 1" in line for line in stats_items)
    assert any("Latest intervention: restart (failed -> pending)" in line for line in stats_items)
    assert any("Latest intervention outcome: RESOLVED" in line for line in stats_items)
    assert any("Intervention outcome reason: Job completed after intervention" in line for line in stats_items)
    assert any("Recent Operator Interventions" == str(item.get("text")) for item in content if item.get("type") == "heading")
    assert any("restart (failed -> pending): Retry after fallback failure [resolved]" in line for line in stats_items)


def test_build_presentation_outline_includes_experiment_recovery_stats_line():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "experiment_run": {
                "final_phase": "fallback",
                "fallback_attempted": True,
                "fallback_ok": False,
                "failed_commands": ["pytest -q"],
            },
            "execution_strategy": {
                "execution_graph": {
                    "graph_health": {"reasons": ["fallback verification still failing"]},
                    "recommended_actions": ["Inspect failing fallback output"],
                }
            },
        }
    )

    outline = exporter._build_presentation_outline(job, include_log=False, include_metadata=True)
    stats_slide = next((slide for slide in outline.slides if slide.title == "Job Statistics"), None)
    assert stats_slide is not None
    assert any("Experiment final phase: fallback" in str(line) for line in (stats_slide.content or []))
    assert any("Experiment recovery: OPEN" in str(line) for line in (stats_slide.content or []))
    assert any("Recovery reason: fallback verification still failing" in str(line) for line in (stats_slide.content or []))
    assert any("Recovery next action: Inspect failing fallback output" in str(line) for line in (stats_slide.content or []))


def test_build_presentation_outline_includes_operator_intervention_stats_line():
    exporter = JobResultsExporter()
    job = _make_job(
        results={
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "restart",
                        "actor_user_id": "user-1",
                        "at": "2026-03-10T01:00:00Z",
                        "job_status_before": "failed",
                        "job_status_after": "pending",
                    }
                ]
            },
        }
    )

    outline = exporter._build_presentation_outline(job, include_log=False, include_metadata=True)
    stats_slide = next((slide for slide in outline.slides if slide.title == "Job Statistics"), None)
    assert stats_slide is not None
    assert any("Operator interventions: 1" in str(line) for line in (stats_slide.content or []))
    assert any("Latest intervention: restart (failed -> pending)" in str(line) for line in (stats_slide.content or []))
    assert any("Latest intervention outcome: RESOLVED" in str(line) for line in (stats_slide.content or []))
    assert any("Intervention outcome reason: Job completed after intervention" in str(line) for line in (stats_slide.content or []))
    assert any(
        "Intervention timeline: restart (failed -> pending) [resolved]" in str(line)
        for line in (stats_slide.content or [])
    )
