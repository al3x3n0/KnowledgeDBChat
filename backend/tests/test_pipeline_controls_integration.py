from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.endpoints import agent_pipelines
from app.models.agent_job import AgentJob
from app.schemas.agent_pipeline import PipelineInsertStageRequest
from app.services import agent_pipeline_binding as binding
from app.services import agent_pipeline_restart as restart
from app.services import agent_pipeline_spec as spec
from app.services.agent_chain_orchestration_service import (
    AgentChainOrchestrationService,
)
from app.services.agent_pipeline_tool_bindings import apply_bindings


async def parent_job(db, user):
    pipeline = spec.normalize(
        {
            "name": "study",
            "stages": [
                {
                    "id": "produce",
                    "goal": "Produce",
                    "tools": ["search_documents"],
                    "tool_defaults": {"search_documents": {"query": "parent default"}},
                    "contract": {"required_finding_types": ["algorithm_spec"]},
                },
                {
                    "id": "consume",
                    "goal": "Check",
                    "depends_on": ["produce"],
                    "contract": {"required_finding_types": ["implementation_verified"]},
                    "tool_bindings": {
                        "check_implementation": {
                            "code": {
                                "stage": "produce",
                                "path": "/structured_output/code",
                            }
                        }
                    },
                },
            ],
        }
    )
    root = binding.bind(pipeline).roots[0]
    parent = AgentJob(
        id=uuid4(),
        name="Produce",
        goal="Produce",
        job_type="research",
        user_id=user.id,
        status="completed",
        chain_depth=0,
        config=root["config"],
        chain_config=root["chain_config"],
        results={
            "goal_contract": {"enabled": True, "satisfied": True},
            "structured_output": {"code": "source code"},
        },
    )
    db.add(parent)
    await db.flush()
    return parent


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_insertion_isolates_controls_and_keeps_explicit_downstream_bindings(
    monkeypatch, db_session, test_user, configured
):
    parent = await parent_job(db_session, test_user)

    async def load(*args):
        return [restart.StageJob("produce", parent)]

    monkeypatch.setattr(restart, "load_run", load)
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay", lambda *args: None
    )
    executor = SimpleNamespace(
        chain_orchestration_service=AgentChainOrchestrationService()
    )
    stage = {
        "id": "extra",
        "goal": "Check another way",
        "contract": {"required_finding_types": ["implementation_verified"]},
    }
    if configured:
        stage.update(
            tools=["search_web"],
            tool_defaults={"search_web": {"query": "extra default"}},
            tool_bindings={
                "search_web": {
                    "query": {"stage": "produce", "path": "/structured_output/code"}
                }
            },
        )
    inserted = await restart.insert_stage_after(
        root_job_id=parent.id,
        after_stage="produce",
        stage=stage,
        executor=executor,
        db=db_session,
    )
    assert inserted.config["pipeline_tools"] == (["search_web"] if configured else None)
    assert inserted.config["pipeline_tool_defaults"] == (
        stage.get("tool_defaults") or {}
    )
    assert inserted.config["pipeline_tool_bindings"] == (
        stage.get("tool_bindings") or {}
    )
    assert inserted.config["pipeline_depends_on"] == ["produce"]
    inserted.results = {"structured_output": {"code": "different output"}}
    consumer = await executor.chain_orchestration_service.create_chained_job(
        executor, inserted, inserted.chain_config["child_jobs"][0], db_session
    )
    assert consumer.config["pipeline_tools"] is None
    assert consumer.config["pipeline_tool_defaults"] == {}
    action = apply_bindings(
        {"tool": "check_implementation", "params": {"code": "model guess"}}, consumer
    )
    assert action["params"]["code"] == "source code"
    assert "_pipeline_binding_error" not in action


@pytest.mark.asyncio
async def test_repeated_insertions_preserve_frozen_downstream_binding_inputs(
    monkeypatch, db_session, test_user
):
    parent = await parent_job(db_session, test_user)
    stages = [restart.StageJob("produce", parent)]

    async def load(*args):
        return stages

    monkeypatch.setattr(restart, "load_run", load)
    monkeypatch.setattr(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay", lambda *args: None
    )
    executor = SimpleNamespace(
        chain_orchestration_service=AgentChainOrchestrationService()
    )
    after = "produce"
    for stage_id in ("extra_1", "extra_2", "extra_3"):
        inserted = await restart.insert_stage_after(
            root_job_id=parent.id,
            after_stage=after,
            stage={
                "id": stage_id,
                "goal": "Check another way",
                "contract": {"required_finding_types": ["implementation_verified"]},
            },
            executor=executor,
            db=db_session,
        )
        # Inputs available to this stage's tools still match its own dependencies.
        assert set(inserted.config["pipeline_input_results"]) == {after}
        inserted.status = "completed"
        inserted.results = {
            "goal_contract": {"enabled": True, "satisfied": True},
            "structured_output": {"code": f"output of {stage_id}"},
        }
        stages.append(restart.StageJob(stage_id, inserted))
        after = stage_id

    # Forward the snapshot, even if the original job changes after insertion.
    parent.results = {"structured_output": {"code": "later change"}}
    consumer = await executor.chain_orchestration_service.create_chained_job(
        executor, inserted, inserted.chain_config["child_jobs"][0], db_session
    )
    action = apply_bindings(
        {"tool": "check_implementation", "params": {"code": "model guess"}}, consumer
    )
    assert "_pipeline_binding_error" not in action, action
    assert action["params"]["code"] == "source code"
    assert set(consumer.config["pipeline_input_results"]) == {"produce"}
    assert consumer.config["pipeline_forward_results"] == {}
    assert consumer.config["pipeline_forward_dependencies"] == []

    # A forwarded result is not a new input the inserted stage may bind.
    undeclared = apply_bindings(
        {"tool": "check_implementation", "params": {}},
        SimpleNamespace(
            config={
                **inserted.config,
                "pipeline_tool_bindings": consumer.config["pipeline_tool_bindings"],
            }
        ),
    )
    assert "declared dependency" in undeclared["_pipeline_binding_error"]

    restarted = await restart.restart_from_stage(
        root_job_id=parent.id,
        stage_id="extra_3",
        executor=executor,
        db=db_session,
    )
    assert (
        restarted.config["pipeline_forward_results"]
        == inserted.config["pipeline_forward_results"]
    )
    restarted.config["pipeline_forward_results"]["produce"]["structured_output"][
        "code"
    ] = "another attempt"
    assert (
        inserted.config["pipeline_forward_results"]["produce"]["structured_output"][
            "code"
        ]
        == "source code"
    )


@pytest.mark.asyncio
async def test_insertion_refuses_a_shape_the_fan_in_runner_cannot_express(
    monkeypatch, db_session, test_user
):
    parent = await parent_job(db_session, test_user)
    parent.chain_config["child_jobs"][0]["config"]["origin"] = "swarm_fan_in_aggregator"

    async def load(*args):
        return [restart.StageJob("produce", parent)]

    monkeypatch.setattr(restart, "load_run", load)
    create = AsyncMock()
    with pytest.raises(restart.PipelineRestartError, match="fan-in branch"):
        await restart.insert_stage_after(
            root_job_id=parent.id,
            after_stage="produce",
            stage={
                "id": "extra",
                "goal": "Check",
                "contract": {"required_finding_types": ["implementation_verified"]},
            },
            executor=SimpleNamespace(
                chain_orchestration_service=SimpleNamespace(create_chained_job=create)
            ),
            db=db_session,
        )
    create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("as_admin", [False, True])
async def test_insert_endpoint_checks_capabilities_for_owner_before_creation(
    monkeypatch, db_session, test_user, admin_user, as_admin
):
    parent = await parent_job(db_session, test_user)
    parent.job_type = "coding"

    async def load(*args):
        return [restart.StageJob("produce", parent)]

    monkeypatch.setattr(restart, "load_run", load)
    check = AsyncMock(return_value=["extra: tool is unavailable to the owner"])
    monkeypatch.setattr(agent_pipelines, "_capability_problems", check)
    create = AsyncMock()
    monkeypatch.setattr(restart, "insert_stage_after", create)
    with pytest.raises(HTTPException) as error:
        await agent_pipelines.insert_pipeline_stage(
            root_job_id=parent.id,
            payload=PipelineInsertStageRequest(
                after="produce",
                stage={
                    "id": "extra",
                    "goal": "Use a tool",
                    "tools": ["owner_tool"],
                    "contract": {"required_finding_types": ["algorithm_spec"]},
                },
            ),
            current_user=admin_user if as_admin else test_user,
            db=db_session,
        )
    assert error.value.status_code == 422
    checked_db, owner, pipeline = check.await_args.args
    assert checked_db is db_session
    assert owner.id == test_user.id
    assert pipeline.stages[0].job_type == "coding"
    assert list(pipeline.stages[0].tools) == ["owner_tool"]
    create.assert_not_called()
