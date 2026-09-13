"""JSON columns on a job must survive the commit that is supposed to save them.

`Column(JSON)` holds an ordinary dict or list, and SQLAlchemy sees a change
only when the attribute is reassigned. Every writer in the agent runtime does
the natural thing instead -- `job.execution_log.append(...)`,
`job.results["summary"] = ...` -- which mutates the object the session already
has and commits nothing. The append succeeds and the value is right for the
rest of the process, so nothing looks wrong until the row is read back in
another one.

Every test here therefore commits, drops the object from the session, and loads
it again. Asserting on the in-memory object is what hid this for so long: those
assertions pass against the bug.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.mutable_json import NestedMutableDict, NestedMutableList


async def _job(db):
    job = AgentJob(
        name="a job",
        goal="measure something",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=5,
        max_tool_calls=5,
        max_llm_calls=5,
        max_runtime_minutes=5,
    )
    db.add(job)
    await db.commit()
    return job


async def _reload(db, job_id):
    """Read the row back as another process would see it."""
    db.expunge_all()
    return await db.get(AgentJob, job_id)


@pytest.mark.asyncio
async def test_a_log_entry_survives_the_commit(db_session):
    job = await _job(db_session)
    job_id = job.id

    job.add_log_entry({"phase": "thinking", "thought": "consider the chain"})
    await db_session.commit()

    reloaded = await _reload(db_session, job_id)
    assert len(reloaded.execution_log) == 1
    assert reloaded.execution_log[0]["thought"] == "consider the chain"


@pytest.mark.asyncio
async def test_entries_accumulate_across_separate_commits(db_session):
    """An iteration record is built one append at a time over a long run."""
    job = await _job(db_session)
    job_id = job.id

    for index in range(4):
        job = await _reload(db_session, job_id)
        job.add_log_entry({"phase": "acting", "step": index})
        await db_session.commit()

    reloaded = await _reload(db_session, job_id)
    assert [e["step"] for e in reloaded.execution_log] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_the_first_entry_on_a_null_column_survives(db_session):
    """This one passed even before the fix, and that is the point.

    add_log_entry *assigns* a fresh list when the column is NULL, and an
    assignment is exactly what SQLAlchemy does detect, so the first entry was
    always written and every later one was dropped. A job's log therefore held
    precisely one entry, which reads as a quiet run rather than as breakage.
    Pinned so that the shape of the old bug stays on record."""
    job = await _job(db_session)
    job_id = job.id
    assert job.execution_log is None

    job.add_log_entry({"phase": "started"})
    await db_session.commit()

    reloaded = await _reload(db_session, job_id)
    assert len(reloaded.execution_log) == 1
    assert reloaded.execution_log[0]["phase"] == "started"


@pytest.mark.asyncio
async def test_a_result_written_by_key_survives(db_session):
    job = await _job(db_session)
    job_id = job.id
    job.results = {}
    await db_session.commit()

    job = await _reload(db_session, job_id)
    job.results["summary"] = "it went well"
    await db_session.commit()

    assert (await _reload(db_session, job_id)).results["summary"] == "it went well"


@pytest.mark.asyncio
async def test_a_result_written_one_level_down_survives(db_session):
    """The case a plain MutableDict would still lose:
    job.results["research"]["brief_document_id"] = ... mutates an ordinary
    dict that merely happens to live inside the tracked one."""
    job = await _job(db_session)
    job_id = job.id
    job.results = {"research": {"created_documents": []}}
    await db_session.commit()

    job = await _reload(db_session, job_id)
    job.results["research"]["brief_document_id"] = "doc-1"
    job.results["research"]["created_documents"].append("doc-1")
    await db_session.commit()

    reloaded = await _reload(db_session, job_id)
    assert reloaded.results["research"]["brief_document_id"] == "doc-1"
    assert reloaded.results["research"]["created_documents"] == ["doc-1"]


@pytest.mark.asyncio
async def test_an_appended_artifact_survives(db_session):
    job = await _job(db_session)
    job_id = job.id
    job.output_artifacts = []
    await db_session.commit()

    job = await _reload(db_session, job_id)
    job.output_artifacts.append({"kind": "manifest", "id": "m-1"})
    await db_session.commit()

    assert (await _reload(db_session, job_id)).output_artifacts[0]["id"] == "m-1"


# --- the wrappers themselves -------------------------------------------------


def test_a_nested_dict_reports_upward():
    root = NestedMutableDict({"a": {"b": {}}})
    seen = []
    root.changed = lambda: seen.append(True)  # type: ignore[method-assign]

    root["a"]["b"]["c"] = 1

    assert seen, "a change three levels down must reach the tracked object"


def test_a_list_inside_a_dict_reports_upward():
    root = NestedMutableDict({"items": []})
    seen = []
    root.changed = lambda: seen.append(True)  # type: ignore[method-assign]

    root["items"].append("x")

    assert seen


def test_a_dict_inside_a_list_reports_upward():
    root = NestedMutableList([{"k": "v"}])
    seen = []
    root.changed = lambda: seen.append(True)  # type: ignore[method-assign]

    root[0]["k"] = "w"

    assert seen


def test_wrapping_preserves_the_plain_value():
    root = NestedMutableDict({"a": [1, {"b": 2}]})

    assert root == {"a": [1, {"b": 2}]}
    assert root["a"][1]["b"] == 2


def test_a_deep_copy_does_not_drag_the_parent_along():
    from copy import deepcopy

    root = NestedMutableDict({"a": {"b": 1}})

    copied = deepcopy(root)

    assert copied == {"a": {"b": 1}}
    assert copied["a"]._parent is not root


def test_the_wrappers_serialise_as_ordinary_json():
    import json

    root = NestedMutableDict({"a": [1, 2], "b": {"c": 3}})

    assert json.loads(json.dumps(root)) == {"a": [1, 2], "b": {"c": 3}}
