from app.services.autonomous_rnd_verification_planner_service import (
    AutonomousRnDVerificationPlanner,
)


def test_unverified_claim_evidence_gets_critical_bounded_local_plan():
    outcome = {
        "claims": [
            {
                "id": "claim-1",
                "evidence_ids": ["external-agent:request-1"],
            }
        ],
        "evidence": [
            {
                "id": "external-agent:request-1",
                "kind": "external_agent_response",
                "verification_status": "unverified",
                "capability": "compiler.review",
            }
        ],
        "verification_links": [],
        "experiment": {},
    }

    planned = AutonomousRnDVerificationPlanner().build_plan(outcome)

    assert planned["verification_plan"]["task_count"] == 1
    task = planned["verification_plan"]["tasks"][0]
    assert task["priority"] == "critical"
    assert task["required_checks"] == [
        "collect_independent_local_evidence",
        "capture_replayable_artifacts",
        "run_repeated_controlled_experiment",
    ]
    assert task["experiment_spec"]["repeat_count"] == 2
    assert task["experiment_spec"]["external_agents_allowed"] is False
    assert task["autolaunch_eligible"] is False
    assert task["approval_required"] is True


def test_corroborated_evidence_plans_only_missing_verification_requirements():
    outcome = {
        "claims": [],
        "evidence": [
            {
                "id": "external-agent:request-2",
                "kind": "external_agent_response",
                "verification_status": "corroborated",
                "verified_by_artifact_ids": ["artifact-1"],
            }
        ],
        "verification_links": [
            {
                "external_evidence_id": "external-agent:request-2",
                "min_repeat_count": 3,
            }
        ],
        "experiment": {
            "ran": True,
            "all_commands_ok": True,
            "repeat_count": 3,
        },
    }

    planned = AutonomousRnDVerificationPlanner().build_plan(outcome)

    task = planned["verification_plan"]["tasks"][0]
    assert task["required_checks"] == ["collect_independent_local_evidence"]
    assert task["experiment_spec"]["repeat_count"] == 3


def test_verified_and_rejected_evidence_do_not_create_tasks():
    outcome = {
        "claims": [],
        "evidence": [
            {
                "id": "external-agent:verified",
                "kind": "external_agent_response",
                "verification_status": "verified",
            },
            {
                "id": "external-agent:rejected",
                "kind": "external_agent_response",
                "verification_status": "rejected",
            },
        ],
    }

    planned = AutonomousRnDVerificationPlanner().build_plan(outcome)

    assert planned["verification_plan"]["task_count"] == 0
    assert planned["verification_plan"]["tasks"] == []


def test_failed_linked_run_cannot_borrow_unrelated_experiment_success():
    outcome = {
        "claims": [],
        "evidence": [
            {
                "id": "external-agent:request-3",
                "kind": "external_agent_response",
                "verification_status": "corroborated",
                "verified_by_evidence_ids": ["local-1"],
                "verified_by_artifact_ids": ["artifact-1"],
            }
        ],
        "verification_links": [
            {
                "external_evidence_id": "external-agent:request-3",
                "experiment_run_id": "run-failed",
            }
        ],
        "verification_experiments": [
            {
                "run_id": "run-failed",
                "ran": True,
                "all_commands_ok": False,
                "repeat_count": 3,
            }
        ],
        "experiment": {
            "ran": True,
            "all_commands_ok": True,
            "repeat_count": 10,
        },
    }

    planned = AutonomousRnDVerificationPlanner().build_plan(outcome)

    assert planned["verification_plan"]["tasks"][0]["required_checks"] == [
        "run_repeated_controlled_experiment"
    ]
