import asyncio
from types import SimpleNamespace
from uuid import UUID

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_note import ResearchNote
from app.models.research_portfolio import ResearchPortfolio
from app.services.research_opportunity_reprioritization_service import (
    project_note_reevaluation_to_autonomous_opportunities,
)


def test_project_note_reevaluation_updates_profile_and_auto_launches_follow_up(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Reprioritized note",
        content_markdown="# Hypothesis Re-evaluation",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "reprioritization_summary": "New evidence improved the top idea.",
            "hypotheses": [
                {
                    "id": "profile-opp-1",
                    "title": "Profile opportunity",
                    "claim": "Better branch locality improves throughput.",
                    "novelty_score": 0.63,
                    "evidence_score": 0.88,
                    "overall_score": 0.86,
                    "recommended_next_step": "Expand the benchmark matrix.",
                    "supporting_sources": [
                        {"id": "paper-1", "title": "Branch locality"}
                    ],
                    "experiment_evidence": [
                        {
                            "run_id": "run-profile-1",
                            "summary": "Profile run improved throughput by 9%.",
                            "result_highlights": ["Runtime: 91ms"],
                            "supporting_sources": [
                                {"id": "run-profile-doc", "title": "Profile run report"}
                            ],
                            "autonomous_origin": {
                                "source_kind": "profile",
                                "source_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                                "opportunity_id": "profile-opp-1",
                                "evidence_revision_at_launch": "seed-profile",
                            },
                        }
                    ],
                }
            ],
            "priority_deltas": [
                {
                    "hypothesis_id": "profile-opp-1",
                    "reason": "Positive evidence reinforced priority.",
                },
            ],
        },
    )
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Profile Parent",
        goal="Track profile ideas",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    profile = DomainResearchProfile(
        id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        user_id=test_user.id,
        title="Profile",
        domain="Compiler",
        objective="Track profile ideas",
        status="active",
        latest_run_job_id=parent_job.id,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "profile-opp-1",
                    "title": "Old profile opportunity",
                    "hypothesis": "Old claim",
                    "decision_state": "accepted",
                    "stage": "accepted",
                    "confidence": 0.52,
                    "readiness": 0.55,
                    "novelty": 0.4,
                    "source_note_ids": [],
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                        "opportunity_id": "profile-opp-1",
                        "evidence_revision_at_launch": "seed-profile",
                    },
                }
            ]
        },
    )
    created_jobs: list[str] = []
    queued_jobs: list[tuple[str, str]] = []

    async def _fake_create_follow_up(self, **kwargs):
        created_jobs.append(str(kwargs.get("top_idea", {}).get("opportunity_id") or ""))
        return SimpleNamespace(id="child-profile-1")

    async def _fake_record_event(*args, **kwargs):
        return SimpleNamespace(id="evt-1")

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._create_domain_research_follow_up_job",
        _fake_create_follow_up,
    )
    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )
    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.record_autonomy_decision_event",
        _fake_record_event,
    )

    async def _run():
        db_session.add(note)
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_run_job_id = parent_job.id
        db_session.add(profile)
        await db_session.commit()
        result = await project_note_reevaluation_to_autonomous_opportunities(
            db=db_session, note=note
        )
        await db_session.commit()
        await db_session.refresh(profile)
        return result, profile

    result, profile = asyncio.get_event_loop().run_until_complete(_run())

    assert result == {
        "profiles_updated": 1,
        "portfolios_updated": 0,
        "opportunities_updated": 1,
    }
    opportunity = profile.latest_summary["opportunities"][0]
    assert opportunity["title"] == "Profile opportunity"
    assert opportunity["confidence"] == 0.88
    assert opportunity["readiness"] == 0.86
    assert opportunity["prior_confidence"] == 0.52
    assert opportunity["reprioritization_source_run_ids"] == ["run-profile-1"]
    assert opportunity["follow_up_review_status"] == "approved_launch"
    assert opportunity["child_job_ids"] == ["child-profile-1"]
    assert created_jobs == ["profile-opp-1"]
    assert queued_jobs == [("child-profile-1", str(test_user.id))]


def test_project_note_reevaluation_queues_portfolio_follow_up_for_approval(
    db_session, test_user, monkeypatch
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Queued note",
        content_markdown="# Hypothesis Re-evaluation",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "hypotheses": [
                {
                    "id": "portfolio-opp-2",
                    "title": "Queue me",
                    "claim": "Good claim",
                    "novelty_score": 0.58,
                    "evidence_score": 0.87,
                    "overall_score": 0.85,
                    "recommended_next_step": "Operator approval should be required.",
                    "experiment_evidence": [
                        {
                            "run_id": "run-queue-1",
                            "summary": "Improved evidence",
                            "autonomous_origin": {
                                "source_kind": "portfolio",
                                "source_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
                                "opportunity_id": "portfolio-opp-2",
                                "evidence_revision_at_launch": "seed-queue",
                            },
                        }
                    ],
                }
            ],
        },
    )
    portfolio = ResearchPortfolio(
        id=UUID("dddddddd-dddd-dddd-dddd-dddddddddddd"),
        user_id=test_user.id,
        title="Queue Portfolio",
        objective="Track approval-gated opportunities",
        status="active",
        automation_policy={
            "auto_launch_follow_up": True,
            "follow_up_review_mode": "queue_for_approval",
            "confidence_threshold": 0.72,
            "experiment_readiness_threshold": 0.8,
        },
        opportunities=[
            {
                "opportunity_id": "portfolio-opp-2",
                "title": "Queue me",
                "hypothesis": "Old claim",
                "decision_state": "accepted",
                "stage": "accepted",
                "confidence": 0.61,
                "readiness": 0.7,
                "novelty": 0.55,
                "autonomous_origin": {
                    "source_kind": "portfolio",
                    "source_id": "dddddddd-dddd-dddd-dddd-dddddddddddd",
                    "opportunity_id": "portfolio-opp-2",
                    "evidence_revision_at_launch": "seed-queue",
                },
            }
        ],
    )

    async def _fake_record_event(*args, **kwargs):
        return SimpleNamespace(id="evt-queue")

    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.record_autonomy_decision_event",
        _fake_record_event,
    )

    async def _run():
        db_session.add(note)
        db_session.add(portfolio)
        await db_session.commit()
        result = await project_note_reevaluation_to_autonomous_opportunities(
            db=db_session, note=note
        )
        await db_session.commit()
        await db_session.refresh(portfolio)
        return result, portfolio

    result, portfolio = asyncio.get_event_loop().run_until_complete(_run())

    assert result == {
        "profiles_updated": 0,
        "portfolios_updated": 1,
        "opportunities_updated": 1,
    }
    opportunity = portfolio.opportunities[0]
    assert opportunity["follow_up_review_status"] == "pending_approval"
    assert (
        opportunity["follow_up_review_evidence_revision"]
        == opportunity["evidence_revision"]
    )
    assert opportunity["child_job_ids"] == []
    assert opportunity["stage"] == "accepted"


def test_project_note_reevaluation_is_idempotent(db_session, test_user, monkeypatch):
    note = ResearchNote(
        user_id=test_user.id,
        title="Idempotent note",
        content_markdown="# Hypothesis Re-evaluation",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "hypotheses": [
                {
                    "id": "opp-1",
                    "title": "Stable opportunity",
                    "claim": "Claim",
                    "novelty_score": 0.6,
                    "evidence_score": 0.84,
                    "overall_score": 0.82,
                    "recommended_next_step": "Keep validating.",
                    "experiment_evidence": [
                        {
                            "run_id": "run-1",
                            "summary": "Stable evidence",
                            "autonomous_origin": {
                                "source_kind": "profile",
                                "source_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                                "opportunity_id": "opp-1",
                                "evidence_revision_at_launch": "seed-1",
                            },
                        }
                    ],
                }
            ],
        },
    )
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Stable Parent",
        goal="Track stable ideas",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    profile = DomainResearchProfile(
        id=UUID("cccccccc-cccc-cccc-cccc-cccccccccccc"),
        user_id=test_user.id,
        title="Stable profile",
        domain="Compiler",
        objective="Track stable ideas",
        status="active",
        latest_run_job_id=parent_job.id,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp-1",
                    "title": "Stable opportunity",
                    "hypothesis": "Claim",
                    "decision_state": "accepted",
                    "stage": "accepted",
                    "confidence": 0.8,
                    "readiness": 0.8,
                    "novelty": 0.6,
                    "supporting_evidence": [],
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                        "opportunity_id": "opp-1",
                        "evidence_revision_at_launch": "seed-1",
                    },
                }
            ]
        },
    )
    created_jobs: list[str] = []
    queued_jobs: list[tuple[str, str]] = []

    async def _fake_create_follow_up(self, **kwargs):
        created_jobs.append(str(kwargs.get("top_idea", {}).get("opportunity_id") or ""))
        return SimpleNamespace(id="child-stable-1")

    async def _fake_record_event(*args, **kwargs):
        return SimpleNamespace(id="evt-stable")

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._create_domain_research_follow_up_job",
        _fake_create_follow_up,
    )
    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )
    monkeypatch.setattr(
        "app.services.research_opportunity_reprioritization_service.record_autonomy_decision_event",
        _fake_record_event,
    )

    async def _run():
        db_session.add(note)
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_run_job_id = parent_job.id
        db_session.add(profile)
        await db_session.commit()
        first = await project_note_reevaluation_to_autonomous_opportunities(
            db=db_session, note=note
        )
        await db_session.commit()
        await db_session.refresh(profile)
        first_stamp = profile.latest_summary["opportunities"][0]["reprioritized_at"]
        second = await project_note_reevaluation_to_autonomous_opportunities(
            db=db_session, note=note
        )
        await db_session.commit()
        await db_session.refresh(profile)
        return (
            first,
            second,
            first_stamp,
            profile.latest_summary["opportunities"][0]["reprioritized_at"],
            profile,
        )

    (
        first,
        second,
        first_stamp,
        second_stamp,
        profile,
    ) = asyncio.get_event_loop().run_until_complete(_run())

    assert first["opportunities_updated"] == 1
    assert second == {
        "profiles_updated": 0,
        "portfolios_updated": 0,
        "opportunities_updated": 0,
    }
    assert first_stamp == second_stamp
    assert created_jobs == ["opp-1"]
    assert queued_jobs == [("child-stable-1", str(test_user.id))]
    assert profile.latest_summary["opportunities"][0]["child_job_ids"] == [
        "child-stable-1"
    ]
