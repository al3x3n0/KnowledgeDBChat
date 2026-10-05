"""A user's LLM preferences are loaded in one place.

Ten modules each had a `_load_user_settings` and another twenty sites wrote
the lookup inline. They agreed on the result and differed in what a caller can
trip on: UUID or string, session first or user first, a logged failure or a
swallowed one, None or an empty settings object when there are no preferences.
"""

import ast
from pathlib import Path
from uuid import uuid4

import pytest

from app.models.memory import UserPreferences
from app.services.llm_service import UserLLMSettings, load_user_llm_settings

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"


async def test_no_preferences_is_none(db_session, test_user):
    assert await load_user_llm_settings(db_session, test_user.id) is None


async def test_preferences_come_back_as_settings(db_session, test_user):
    db_session.add(
        UserPreferences(
            user_id=test_user.id, llm_provider="deepseek", llm_model="deepseek-v4-pro"
        )
    )
    await db_session.commit()

    for key in (test_user.id, str(test_user.id)):
        settings = await load_user_llm_settings(db_session, key)
        assert isinstance(settings, UserLLMSettings)
        assert (settings.provider, settings.model) == ("deepseek", "deepseek-v4-pro")


async def test_nothing_to_look_up_is_none_without_a_query(db_session):
    class Exploding:
        async def execute(self, *_a, **_k):
            raise AssertionError("no query should be made")

    assert await load_user_llm_settings(Exploding(), None) is None
    assert await load_user_llm_settings(Exploding(), "") is None
    assert await load_user_llm_settings(None, uuid4()) is None


async def test_a_failure_is_none_not_an_exception():
    """Preferences are a refinement: a turn that cannot read them runs on the
    defaults rather than failing."""

    class Broken:
        async def execute(self, *_a, **_k):
            raise RuntimeError("connection reset")

    assert await load_user_llm_settings(Broken(), uuid4()) is None
    assert await load_user_llm_settings(Broken(), "not-a-uuid") is None


def test_every_named_loader_delegates():
    """A module may keep its `_load_user_settings` for its callers' sake; it
    may not keep its own copy of the query."""
    own_copies = []
    for path in sorted(APP.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_load_user_settings"
                and "load_user_llm_settings("
                not in ast.get_source_segment(source, node)
            ):
                own_copies.append(f"{path.relative_to(APP)}:{node.lineno}")
    assert not own_copies, f"these query preferences themselves: {own_copies}"


#: Sites that still build settings from a preferences row inline. Each has a
#: shape of its own (a different default, a row already in hand). The number
#: may only go down.
INLINE_SITES_REMAINING = 5


def test_inline_lookups_do_not_grow_back():
    count = sum(
        path.read_text(encoding="utf-8").count("UserLLMSettings.from_preferences(")
        for path in APP.rglob("*.py")
        if path.name != "llm_service.py"
    )
    assert (
        count <= INLINE_SITES_REMAINING
    ), f"{count} inline lookups; call load_user_llm_settings(db, user_id) instead"
