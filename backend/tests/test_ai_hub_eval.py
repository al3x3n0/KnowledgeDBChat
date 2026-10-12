"""Running an AI Hub eval through its route, with Ollama answering.

* A judge reply that was not a score was entered as a 3 and averaged in, so a
  judge that never answered in JSON reported 3.00 / 5.
* The "last eval" summary was written into the dict the JSON column already
  held, which SQLAlchemy does not see as a change: it was computed, returned
  and not saved.
* Evaluating someone's public adapter wrote into their record.
* Ollama being away was a 500.
"""

import json
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.api.endpoints import ai_hub_eval as eval_endpoints
from app.models.model_registry import AdapterStatus, ModelAdapter
from app.schemas.ai_hub_eval import RunEvalRequest
from app.services import ai_hub_eval_service as eval_module
from app.services.ai_hub_eval_service import EvalTemplate

pytestmark = pytest.mark.unit

TEMPLATE = EvalTemplate(
    id="support",
    name="Support",
    description="",
    version=2,
    judge_preamble="",
    rubric={"criteria": ["accurate"]},
    cases=[
        {"id": "a", "prompt": "First question"},
        {"id": "b", "prompt": "Second question"},
        {"id": "c", "prompt": "Third question"},
    ],
)


class _Ollama:
    """Answers generation; judges by the question it is shown."""

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.reachable = True

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not self.reachable:
            raise httpx.ConnectError("refused", request=request)
        body = json.loads(request.content)
        prompt = body["prompt"]
        if "CANDIDATE ANSWER" in prompt:
            for question, verdict in self.verdicts.items():
                if question in prompt:
                    return httpx.Response(200, json={"response": verdict})
        return httpx.Response(200, json={"response": f"answer from {body['model']}"})


@pytest.fixture
def ollama(monkeypatch):
    fake = _Ollama(
        {
            "First question": '{"score": 5, "notes": "good"}',
            "Second question": "I think the candidate did fine.",
            "Third question": '{"score": 9, "notes": "off the scale"}',
        }
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        eval_module.httpx,
        "AsyncClient",
        lambda *a, **k: real(transport=httpx.MockTransport(fake.handle)),
    )
    monkeypatch.setattr(
        eval_endpoints.ai_hub_eval_service, "get_template", lambda _id: TEMPLATE
    )
    return fake


async def _deployed_adapter(db, user, **fields):
    values = {
        "name": f"adapter-{uuid4().hex[:8]}",
        "display_name": "Support tone",
        "base_model": "llama3.2:3b",
        "adapter_path": "adapters/x",
        "user_id": user.id,
        "status": AdapterStatus.DEPLOYED.value,
        "is_deployed": True,
        "deployment_config": {"ollama_model_name": "support-tone"},
        "training_metrics": {"loss": 0.12},
    }
    values.update(fields)
    adapter = ModelAdapter(**values)
    db.add(adapter)
    await db.commit()
    await db.refresh(adapter)
    return adapter


async def _run(db, user, adapter):
    return await eval_endpoints.run_eval(
        RunEvalRequest(adapter_id=str(adapter.id), template_id="support"),
        db=db,
        current_user=user,
    )


async def test_only_judged_cases_are_averaged(db_session, test_user, ollama):
    adapter = await _deployed_adapter(db_session, test_user)

    report = await _run(db_session, test_user, adapter)

    assert report.num_cases == 3
    assert report.num_judged == 1
    assert report.avg_score == 5.0
    by_case = {r["case_id"]: r for r in report.results}
    assert by_case["a"]["score"] == 5 and by_case["a"]["judged"] is True
    assert by_case["b"]["score"] is None and by_case["b"]["judged"] is False
    assert by_case["c"]["score"] is None
    assert by_case["a"]["candidate_response"] == "answer from support-tone"
    assert by_case["a"]["base_response"] == "answer from llama3.2:3b"


async def test_the_run_is_saved_on_the_adapter(db_session, test_user, ollama):
    adapter = await _deployed_adapter(db_session, test_user)

    await _run(db_session, test_user, adapter)
    await _run(db_session, test_user, adapter)

    # Read back from the database, not from the object the route edited.
    adapter_id = adapter.id
    db_session.expire_all()
    stored = await db_session.get(ModelAdapter, adapter_id)
    assert stored.training_metrics["loss"] == 0.12
    runs = stored.training_metrics["eval_runs"]
    assert len(runs) == 2
    assert runs[-1]["avg_score"] == 5.0
    assert runs[-1]["num_judged"] == 1 and runs[-1]["num_cases"] == 3


async def test_evaluating_a_shared_adapter_leaves_its_record_alone(
    db_session, test_user, admin_user, ollama
):
    shared = await _deployed_adapter(db_session, admin_user, is_public=True)

    report = await _run(db_session, test_user, shared)

    assert report.num_cases == 3
    shared_id = shared.id
    db_session.expire_all()
    stored = await db_session.get(ModelAdapter, shared_id)
    assert "eval_runs" not in stored.training_metrics


async def test_ollama_being_away_is_not_this_servers_error(
    db_session, test_user, ollama
):
    adapter = await _deployed_adapter(db_session, test_user)
    ollama.reachable = False

    with pytest.raises(HTTPException) as refused:
        await _run(db_session, test_user, adapter)

    assert refused.value.status_code == 503


async def test_an_adapter_that_is_not_deployed_is_refused(
    db_session, test_user, ollama
):
    adapter = await _deployed_adapter(
        db_session, test_user, is_deployed=False, deployment_config=None
    )

    with pytest.raises(HTTPException) as refused:
        await _run(db_session, test_user, adapter)

    assert refused.value.status_code == 400
