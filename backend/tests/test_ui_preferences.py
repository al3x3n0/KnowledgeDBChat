"""Bounds on the one field a client composes freely.

`user_preferences.ui` is the only place in this application where a user's own
document becomes configuration the application then acts on. Everything else on
that model is a typed scalar Pydantic has already bounded. So the normalizer is
what stands between a malformed or hostile preferences document and a row, and
these are its edges rather than its happy path.
"""

from __future__ import annotations

import pytest

from app.services.ui_preferences import (
    MAX_ENTRIES,
    MAX_KEY_LENGTH,
    MAX_LABEL_LENGTH,
    normalize,
)

# --------------------------------------------------------------------------
# Nothing customized is None, not {}
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "empty", [None, {}, {"nav": {}}, {"nav": None}, "nonsense", 42, []]
)
def test_nothing_customized_stores_null(empty):
    """Null means "never customized"; an empty object would mean "customized
    to exactly today's defaults" and would pin that person to them for ever."""
    assert normalize(empty) is None


def test_unknown_top_level_keys_are_dropped():
    assert normalize({"nav": {"hidden": ["/papers"]}, "evil": {"a": 1}}) == {
        "nav": {"hidden": ["/papers"]}
    }


def test_unknown_nav_keys_are_dropped():
    assert normalize({"nav": {"hidden": ["/papers"], "whatever": [1, 2]}}) == {
        "nav": {"hidden": ["/papers"]}
    }


# --------------------------------------------------------------------------
# Lists
# --------------------------------------------------------------------------


def test_order_is_preserved_and_duplicates_collapse():
    out = normalize({"nav": {"doorOrder": ["rnd", "chat", "rnd", "library"]}})

    assert out["nav"]["doorOrder"] == ["rnd", "chat", "library"]


def test_a_list_is_capped():
    out = normalize({"nav": {"hidden": [f"/p{i}" for i in range(MAX_ENTRIES + 50)]}})

    assert len(out["nav"]["hidden"]) == MAX_ENTRIES


def test_an_overlong_key_is_dropped_not_truncated():
    """Truncating would produce a key that matches a *different* destination."""
    out = normalize({"nav": {"hidden": ["/ok", "/" + "x" * (MAX_KEY_LENGTH + 1)]}})

    assert out["nav"]["hidden"] == ["/ok"]


def test_blank_and_non_string_entries_are_dropped():
    out = normalize({"nav": {"pinned": ["/a", "", "   ", None, 7, "/b"]}})

    assert out["nav"]["pinned"] == ["/a", "/b"]


def test_a_list_that_is_not_a_list_is_ignored():
    assert normalize({"nav": {"hidden": "/papers"}}) is None


# --------------------------------------------------------------------------
# Renames
# --------------------------------------------------------------------------


def test_a_label_is_trimmed_to_a_length():
    out = normalize({"nav": {"renamed": {"/papers": "x" * (MAX_LABEL_LENGTH + 40)}}})

    assert len(out["nav"]["renamed"]["/papers"]) == MAX_LABEL_LENGTH


def test_an_empty_rename_is_a_removal_not_a_blank_label():
    """A nav entry with no name is unreachable by anything but position."""
    out = normalize({"nav": {"renamed": {"/papers": "   ", "/memory": "Notes"}}})

    assert out["nav"]["renamed"] == {"/memory": "Notes"}


def test_renames_that_are_not_a_mapping_are_ignored():
    assert normalize({"nav": {"renamed": ["/papers", "Nope"]}}) is None


# --------------------------------------------------------------------------
# Landing
# --------------------------------------------------------------------------


def test_the_landing_page_is_kept_as_given():
    out = normalize({"nav": {"landing": "/agent-control-plane"}})

    assert out["nav"]["landing"] == "/agent-control-plane"


def test_a_blank_landing_page_is_dropped():
    assert normalize({"nav": {"landing": "  "}}) is None


def test_a_route_that_does_not_exist_is_kept_and_left_inert():
    """The catalog decides what exists, not this. A person who hid a page that
    was later renamed should not have their whole navigation rejected."""
    out = normalize({"nav": {"hidden": ["/a-page-that-was-removed"]}})

    assert out["nav"]["hidden"] == ["/a-page-that-was-removed"]


# --------------------------------------------------------------------------
# A realistic document survives intact
# --------------------------------------------------------------------------


def test_a_full_document_round_trips():
    document = {
        "nav": {
            "doorOrder": ["rnd", "chat", "library", "synthesis"],
            "hidden": ["/latex", "/presentations"],
            "pinned": ["/agent-control-plane"],
            "renamed": {"/autonomous-agents": "My Runs"},
            "landing": "/agent-control-plane",
        }
    }

    assert normalize(document) == document
