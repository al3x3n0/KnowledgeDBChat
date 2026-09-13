"""Tests for follow-up learning-profile loading and normalization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.modules.autonomy.application.follow_up_learning_profiles import (
    PROFILE_FIELDS,
    empty_follow_up_learning_profile,
    load_follow_up_learning_profile,
)


def test_empty_learning_profiles_are_complete_and_independent():
    first = empty_follow_up_learning_profile()
    second = empty_follow_up_learning_profile()

    assert tuple(first) == PROFILE_FIELDS
    assert all(value == {} for value in first.values())
    first["token_scores"]["compiler"] = 2
    assert second["token_scores"] == {}


@pytest.mark.asyncio
async def test_loader_returns_empty_contract_when_profile_is_missing():
    loader = AsyncMock(return_value=None)
    user_id = uuid4()
    db = object()

    profile = await load_follow_up_learning_profile(
        db=db,
        user_id=user_id,
        customer="Acme",
        profile_loader=loader,
    )

    loader.assert_awaited_once_with(db=db, user_id=user_id, customer="Acme")
    assert profile == empty_follow_up_learning_profile()


@pytest.mark.asyncio
async def test_loader_tolerates_profile_lookup_failure():
    loader = AsyncMock(side_effect=RuntimeError("profile store unavailable"))

    profile = await load_follow_up_learning_profile(
        db=object(),
        user_id=uuid4(),
        customer=None,
        profile_loader=loader,
    )

    assert profile == empty_follow_up_learning_profile()


@pytest.mark.asyncio
async def test_loader_preserves_only_dictionary_score_fields():
    token_scores = {"compiler": 3}
    profile_row = SimpleNamespace(
        token_scores=token_scores,
        phrase_scores=None,
        recommendation_scores={"deep_dive_chain": 2},
        source_type_scores=["invalid"],
        outcome_counters={"completed_follow_up": 4},
    )

    profile = await load_follow_up_learning_profile(
        db=object(),
        user_id=uuid4(),
        customer="Acme",
        profile_loader=AsyncMock(return_value=profile_row),
    )

    assert profile == {
        "token_scores": token_scores,
        "phrase_scores": {},
        "recommendation_scores": {"deep_dive_chain": 2},
        "source_type_scores": {},
        "outcome_counters": {"completed_follow_up": 4},
    }
