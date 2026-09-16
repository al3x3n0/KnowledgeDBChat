"""Drafting a manifest, and the two things that make the draft worth having.

The interesting behaviour is not the happy path -- it is that a wrong draft is
handed its own refusal and tries again, and that a draft whose view reads a
path its tool does not produce is caught by *running* the tool rather than by
reading the manifest.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from app.services import plugin_author_service as author


class ScriptedLLM:
    """Serves queued replies, and records what it was asked."""

    def __init__(self, replies: List[Any]):
        self._replies = list(replies)
        self.prompts: List[str] = []

    async def generate_structured(self, **kwargs):
        self.prompts.append(str(kwargs.get("user_message") or ""))
        reply = self._replies.pop(0) if self._replies else {}
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            return SimpleNamespace(structured=None, text=reply)
        return SimpleNamespace(structured=reply, text="")


@pytest.fixture
def scripted(monkeypatch):
    def install(replies):
        llm = ScriptedLLM(replies)
        monkeypatch.setattr(author, "LLMService", lambda: llm, raising=False)
        monkeypatch.setattr(
            "app.services.llm_service.LLMService", lambda: llm, raising=False
        )
        return llm

    return install


def _manifest(**over) -> Dict[str, Any]:
    base = {
        "id": "bench",
        "name": "Benchmarks",
        "version": "0.1.0",
        "contributes": {
            "tools": [
                {
                    "name": "recent",
                    "description": "Recent runs",
                    "tool_type": "transform",
                    "parameters_schema": {"type": "object", "properties": {}},
                    "config": {"template": '{"items":[{"name":"a","ms":1}]}'},
                    "job_types": ["research"],
                }
            ],
            "views": {
                "board": {
                    "kind": "table",
                    "title": "Runs",
                    "source": {"tool": "recent"},
                    "path": "items",
                    "columns": ["name", "ms"],
                }
            },
        },
    }
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_a_valid_draft_is_returned_on_the_first_attempt(scripted):
    scripted([_manifest()])

    result = await author.draft_manifest("a run board")

    assert result["manifest"]["id"] == "bench"
    assert result["attempts"] == 1
    assert result["notes"] == []


@pytest.mark.asyncio
async def test_a_refusal_is_handed_back_and_the_next_attempt_uses_it(scripted):
    """The validator's messages were written to be actionable for a human
    author, which is what makes them a usable repair signal for a model."""
    broken = _manifest(id="Bench-Marks")
    llm = scripted([broken, _manifest()])

    result = await author.draft_manifest("a run board")

    assert result["manifest"]["id"] == "bench"
    assert result["attempts"] == 2
    assert "lowercase" in result["notes"][0]
    # The second prompt must actually carry the reason, or the retry is a coin
    # flip rather than a repair.
    assert "lowercase" in llm.prompts[1]


@pytest.mark.asyncio
async def test_giving_up_returns_no_manifest_but_says_why(scripted):
    scripted([_manifest(id="Nope!")] * author.MAX_ATTEMPTS)

    result = await author.draft_manifest("a run board")

    assert result["manifest"] is None
    assert len(result["notes"]) == author.MAX_ATTEMPTS
    assert all("lowercase" in note for note in result["notes"])


@pytest.mark.asyncio
async def test_a_reply_that_is_not_json_is_challenged(scripted):
    llm = scripted(["I would suggest a plugin that…", _manifest()])

    result = await author.draft_manifest("a run board")

    assert result["manifest"]["id"] == "bench"
    assert "not JSON" in result["notes"][0]
    assert "JSON object" in llm.prompts[1]


@pytest.mark.asyncio
async def test_fenced_json_is_accepted(scripted):
    scripted(["```json\n" + json.dumps(_manifest()) + "\n```"])

    result = await author.draft_manifest("a run board")

    assert result["manifest"]["id"] == "bench"


@pytest.mark.asyncio
async def test_an_unreachable_model_is_reported_not_swallowed(scripted):
    scripted([RuntimeError("provider down")])

    result = await author.draft_manifest("a run board")

    assert result["manifest"] is None
    assert "provider down" in result["notes"][0]


@pytest.mark.asyncio
async def test_an_empty_description_asks_for_one(scripted):
    result = await author.draft_manifest("   ")

    assert result["manifest"] is None
    assert result["attempts"] == 0


# --------------------------------------------------------------------------
# The vocabulary is read, never restated
# --------------------------------------------------------------------------


def test_the_prompt_is_built_from_the_live_vocabularies():
    """Restating these in the prompt would be another copy to keep in step, and
    the first to drift would produce drafts refused for a reason the author
    cannot act on."""
    from app.services.plugin_ui import ICONS, NAV_DOORS, PANEL_SLOTS, VIEW_KINDS

    prompt = author._system_prompt()

    for door in NAV_DOORS:
        assert door in prompt
    for kind in VIEW_KINDS:
        assert kind in prompt
    for slot in PANEL_SLOTS:
        assert slot in prompt
    assert ICONS[0] in prompt


def test_only_read_only_executors_are_offered_for_a_view():
    vocab = author.vocabulary()

    assert "webhook" not in vocab["read_only_tool_types"]
    assert "transform" in vocab["read_only_tool_types"]


def test_only_inert_executors_are_dry_run():
    """Running an *unreviewed* draft must cost nothing: a webhook would reach
    the network and an llm_prompt would spend a model call, neither of which
    should happen because somebody typed a sentence into a box."""
    assert author.DRY_RUN_TYPES == {"transform"}


# --------------------------------------------------------------------------
# Path resolution, mirrored from the renderer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data,path,found,stopped",
    [
        ({"items": []}, "items", True, ""),
        ({"items": []}, "result.items", False, "(root)"),
        ({"a": {"b": 1}}, "a.b", True, ""),
        ({"a": 5}, "a.b", False, "a"),
        ({}, "", True, ""),
    ],
)
def test_the_path_walker_matches_the_renderer(data, path, found, stopped):
    ok, where = author._resolve(data, path)

    assert ok is found
    if not found:
        assert where == stopped


def test_the_shape_summary_is_readable():
    assert author._shape({"items": [{"a": 1}]}) == "{items: [{a: int}]}"
    assert author._shape([]) == "[]"


# --------------------------------------------------------------------------
# Progress, and running it off the request thread
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_each_attempt_is_reported_as_it_happens(scripted):
    """The repair loop's reasons are the interesting part of the wait. A caller
    that only learns them at the end has watched a spinner for two minutes."""
    scripted([_manifest(id="Bad-Id"), _manifest()])
    seen: List[Any] = []

    await author.draft_manifest(
        "a run board", on_progress=lambda *args: seen.append(args)
    )

    stages = [stage for stage, _attempt, _notes in seen]
    assert stages[0] == "drafting"
    assert stages[-1] == "done"
    # The second attempt must carry the reason the first was refused, or the
    # progress says "still working" and nothing more.
    second = next(notes for stage, attempt, notes in seen if attempt == 2)
    assert second and "lowercase" in second[0]


@pytest.mark.asyncio
async def test_a_broken_progress_callback_does_not_lose_the_draft(scripted):
    """Progress is a courtesy. A caller whose reporting breaks must not take
    the draft down with it."""

    def boom(*_args):
        raise RuntimeError("reporting is broken")

    scripted([_manifest()])

    result = await author.draft_manifest("a run board", on_progress=boom)

    assert result["manifest"]["id"] == "bench"


def test_the_task_is_registered_and_bounded():
    """A wedged provider must not hold a worker for ever."""
    from app.core.celery import celery_app
    from app.tasks import plugin_tasks

    assert "app.tasks.plugin_tasks.draft_plugin_manifest" in celery_app.tasks
    assert plugin_tasks.HARD_LIMIT_SECONDS > plugin_tasks.SOFT_LIMIT_SECONDS
    # Generous against the two minutes observed, finite so it cannot hang.
    assert plugin_tasks.SOFT_LIMIT_SECONDS >= 180


def test_the_task_carries_its_owner_in_every_state():
    """A task id is unguessable, which is not the same as checked: the polling
    endpoint compares this against the caller."""
    import inspect

    from app.tasks import plugin_tasks

    source = inspect.getsource(plugin_tasks.draft_plugin_manifest)

    assert '"user_id": str(user_id)' in source
    assert 'state="PROGRESS"' in source
