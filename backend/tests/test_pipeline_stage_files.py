"""Files between pipeline stages: fan-in, and how long they are kept.

Two limitations, both lifted here:

* a fan-in stage is created as a child of whichever stage it waits on finished
  last, so it inherited that one stage's files and none of the others';
* directories were pruned by age alone, so a finished stage's files were
  deleted while the next stage waited more than a day on a checkpoint.
"""

import os
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob
from app.services import agent_pipeline_binding, agent_sandbox_skill_tools
from app.services import sandbox_skill_runtime as runtime
from app.services.agent_pipeline_spec import normalize

pytestmark = pytest.mark.unit

CONTRACT = {"required_finding_types": ["algorithm_spec"]}


@pytest.fixture
def skills_root(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    return tmp_path


# ------------------------------------------------------------------- fan-in


def test_a_fan_in_stage_starts_with_every_stage_it_waits_on(skills_root):
    left = runtime.run_dir("left-job")
    (left / "left.o").write_text("from left")
    right = runtime.run_dir("right-job")
    (right / "right.o").write_text("from right")

    merged = runtime.run_dir(
        "merge-job", "right-job", parent_label="right", also_from=[("left", "left-job")]
    )

    assert (merged / "left.o").read_text() == "from left"
    assert (merged / "right.o").read_text() == "from right"
    note = runtime.inheritance_note(merged)
    assert "several earlier stages" in note
    assert "stage left" in note and "stage right" in note


def test_a_name_two_stages_left_is_kept_from_both(skills_root):
    """Neither copy silently wins: the second goes under from-<stage>/."""
    left = runtime.run_dir("left-job")
    (left / "prog").write_text("left build")
    right = runtime.run_dir("right-job")
    (right / "prog").write_text("right build")

    merged = runtime.run_dir(
        "merge-job", "right-job", parent_label="right", also_from=[("left", "left-job")]
    )

    assert (merged / "prog").read_text() == "right build"
    assert (merged / "from-left" / "prog").read_text() == "left build"
    assert "from-left/" in runtime.inheritance_note(merged)


def test_one_budget_covers_every_stage(skills_root, monkeypatch):
    monkeypatch.setattr(runtime, "MAX_INHERIT_BYTES", 150)
    left = runtime.run_dir("left-job")
    (left / "a").write_text("x" * 100)
    right = runtime.run_dir("right-job")
    (right / "b").write_text("y" * 100)

    merged = runtime.run_dir(
        "merge-job", "right-job", parent_label="right", also_from=[("left", "left-job")]
    )

    assert (merged / "b").exists()
    assert not (merged / "a").exists()
    assert "more than remained" in runtime.inheritance_note(merged)


def test_the_binding_names_what_a_fan_in_stage_waits_on():
    pipeline = normalize(
        {
            "name": "diamond",
            "stages": [
                {"id": "start", "goal": "g", "contract": CONTRACT},
                {
                    "id": "left",
                    "goal": "g",
                    "depends_on": ["start"],
                    "contract": CONTRACT,
                },
                {
                    "id": "right",
                    "goal": "g",
                    "depends_on": ["start"],
                    "contract": CONTRACT,
                },
                {
                    "id": "merge",
                    "goal": "g",
                    "depends_on": ["left", "right"],
                    "contract": CONTRACT,
                },
            ],
        }
    )
    bound = agent_pipeline_binding.bind(pipeline)

    configs = {}

    def walk(node):
        cfg = node.get("config") or {}
        configs[cfg.get("pipeline_stage")] = cfg
        for child in (node.get("chain_config") or {}).get("child_jobs") or []:
            walk(child)

    for root in bound.roots:
        walk(root)
    assert configs["merge"]["pipeline_depends_on"] == ["left", "right"]
    assert "pipeline_depends_on" not in configs["left"]


async def _job(db, user, stage, *, parent=None, status="completed", depends_on=None):
    config = {"pipeline_stage": stage}
    if depends_on:
        config["pipeline_depends_on"] = depends_on
    job = AgentJob(
        name=stage,
        goal="g",
        job_type="research",
        user_id=user.id,
        status=status,
        config=config,
        parent_job_id=parent.id if parent else None,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _diamond(db, user, *, merge_status="running"):
    start = await _job(db, user, "start")
    left = await _job(db, user, "left", parent=start)
    right = await _job(db, user, "right", parent=start)
    merge = await _job(
        db,
        user,
        "merge",
        parent=right,
        status=merge_status,
        depends_on=["left", "right"],
    )
    return start, left, right, merge


async def test_a_fan_in_job_finds_the_stages_it_was_not_created_under(
    db_session, test_user
):
    _, left, right, merge = await _diamond(db_session, test_user)

    label, others = await agent_sandbox_skill_tools._stage_sources(
        SimpleNamespace(db=db_session), merge
    )

    assert label == "right"
    assert others == [("left", str(left.id))]


async def test_a_restarted_stage_contributes_its_newest_job(db_session, test_user):
    start, left, right, merge = await _diamond(db_session, test_user)
    rerun = await _job(db_session, test_user, "left", parent=start)

    _, others = await agent_sandbox_skill_tools._stage_sources(
        SimpleNamespace(db=db_session), merge
    )

    assert others == [("left", str(rerun.id))]


async def test_the_skill_tools_seed_a_fan_in_from_both_branches(
    db_session, test_user, skills_root
):
    _, left, right, merge = await _diamond(db_session, test_user)
    (runtime.run_dir(str(left.id)) / "left.o").write_text("L")
    (runtime.run_dir(str(right.id)) / "right.o").write_text("R")

    workdir = await agent_sandbox_skill_tools._workdir(
        SimpleNamespace(db=db_session, job=merge, extra={})
    )

    assert sorted(runtime.list_files(workdir)["files"]) == ["left.o", "right.o"]


# ------------------------------------------------------------------ pruning


def _age(path, days):
    then = time.time() - days * 86400
    os.utime(path, (then, then))


def test_a_kept_directory_survives_any_age(skills_root):
    old = runtime.run_dir("finished-stage")
    abandoned = runtime.run_dir("abandoned")
    _age(old, 5)
    _age(abandoned, 5)

    removed = runtime.prune_stale(keep=["finished-stage"])

    assert removed == 1
    assert old.exists()
    assert not abandoned.exists()


async def test_a_run_with_a_live_stage_keeps_every_stage(db_session, test_user):
    # The merge waits on a checkpoint (paused); every earlier stage finished.
    start, left, right, merge = await _diamond(
        db_session, test_user, merge_status="paused"
    )
    finished = await _job(db_session, test_user, "other-run")

    keep = await agent_sandbox_skill_tools._live_run_keys(db_session)

    assert {str(j.id) for j in (start, left, right, merge)} <= keep
    assert str(finished.id) not in keep


async def test_a_finished_run_keeps_nothing(db_session, test_user):
    await _diamond(db_session, test_user, merge_status="completed")
    assert await agent_sandbox_skill_tools._live_run_keys(db_session) == set()


async def test_pruning_spares_a_live_runs_old_stages(
    db_session, test_user, skills_root
):
    _, left, _, _ = await _diamond(db_session, test_user, merge_status="paused")
    kept = runtime.run_dir(str(left.id))
    stray = runtime.run_dir(str(uuid4()))
    _age(kept, 3)
    _age(stray, 3)

    await agent_sandbox_skill_tools._prune(SimpleNamespace(db=db_session))

    assert kept.exists()
    assert not stray.exists()


async def test_pruning_does_nothing_when_it_cannot_tell_what_is_live(skills_root):
    stray = runtime.run_dir("stray")
    _age(stray, 3)

    class Broken:
        async def execute(self, *args, **kwargs):
            raise RuntimeError("database unavailable")

    await agent_sandbox_skill_tools._prune(SimpleNamespace(db=Broken()))

    assert stray.exists()
