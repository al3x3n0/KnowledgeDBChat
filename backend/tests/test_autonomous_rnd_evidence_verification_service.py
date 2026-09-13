from app.services.autonomous_rnd_evidence_verification_service import (
    AutonomousRnDEvidenceVerifier,
)


def _outcome(*, links, experiment=None):
    return {
        "evidence": [
            {
                "id": "external-agent:request-1",
                "kind": "external_agent_response",
                "verification_status": "verified",
            },
            {
                "id": "local-benchmark-1",
                "kind": "benchmark_output",
                "record_origin": "runtime_finding",
            },
        ],
        "artifacts": [{"id": "artifact-1", "kind": "compiler_logs"}],
        "verification_links": links,
        "experiment": experiment or {},
    }


def test_external_evidence_cannot_self_assert_verified_status():
    verified = AutonomousRnDEvidenceVerifier().verify(_outcome(links=[]))

    external = verified["evidence"][0]
    assert external["verification_status"] == "unverified"
    assert verified["evidence_verification"]["status_counts"]["unverified"] == 1


def test_local_support_without_repeatable_experiment_is_corroborated():
    verified = AutonomousRnDEvidenceVerifier().verify(
        _outcome(
            links=[
                {
                    "external_evidence_id": "external-agent:request-1",
                    "verdict": "supports",
                    "local_evidence_ids": ["local-benchmark-1"],
                    "artifact_ids": ["artifact-1"],
                }
            ],
            experiment={
                "ran": True,
                "all_commands_ok": True,
                "repeat_count": 1,
            },
        )
    )

    assert verified["evidence"][0]["verification_status"] == "corroborated"


def test_independent_evidence_artifact_and_repeated_experiment_verify():
    verified = AutonomousRnDEvidenceVerifier().verify(
        _outcome(
            links=[
                {
                    "external_evidence_id": "external-agent:request-1",
                    "verdict": "supports",
                    "local_evidence_ids": ["local-benchmark-1"],
                    "artifact_kinds": ["compiler_logs"],
                }
            ],
            experiment={
                "ran": True,
                "all_commands_ok": True,
                "repeat_count": 2,
            },
        )
    )

    external = verified["evidence"][0]
    assert external["verification_status"] == "verified"
    assert external["verified_by_evidence_ids"] == ["local-benchmark-1"]
    assert external["verified_by_artifact_kinds"] == ["compiler_logs"]


def test_grounded_contradiction_rejects_external_evidence():
    verified = AutonomousRnDEvidenceVerifier().verify(
        _outcome(
            links=[
                {
                    "external_evidence_id": "external-agent:request-1",
                    "verdict": "contradicts",
                    "local_evidence_ids": ["local-benchmark-1"],
                }
            ]
        )
    )

    assert verified["evidence"][0]["verification_status"] == "rejected"


def test_agent_authored_evidence_cannot_promote_external_evidence():
    outcome = _outcome(
        links=[
            {
                "external_evidence_id": "external-agent:request-1",
                "verdict": "supports",
                "local_evidence_ids": ["local-benchmark-1"],
                "artifact_ids": ["artifact-1"],
            }
        ],
        experiment={
            "ran": True,
            "all_commands_ok": True,
            "repeat_count": 3,
        },
    )
    outcome["evidence"][1]["record_origin"] = "structured_output"

    verified = AutonomousRnDEvidenceVerifier().verify(outcome)

    assert verified["evidence"][0]["verification_status"] == "corroborated"
    assert "verified_by_evidence_ids" not in verified["evidence"][0]


def test_linked_experiment_cannot_borrow_success_from_another_run():
    outcome = _outcome(
        links=[
            {
                "external_evidence_id": "external-agent:request-1",
                "verdict": "supports",
                "local_evidence_ids": ["local-benchmark-1"],
                "artifact_ids": ["artifact-1"],
                "experiment_run_id": "run-failed",
            }
        ],
        experiment={
            "ran": True,
            "all_commands_ok": True,
            "repeat_count": 5,
        },
    )
    outcome["verification_experiments"] = [
        {
            "run_id": "run-failed",
            "ran": True,
            "all_commands_ok": False,
            "repeat_count": 3,
        }
    ]

    verified = AutonomousRnDEvidenceVerifier().verify(outcome)

    assert verified["evidence"][0]["verification_status"] == "corroborated"
