"""A campaign should decide what to do next, not merely what is oldest.

Working the backlog in creation order is the same as having no opinion: with a
job budget in the tens and jobs taking tens of minutes, order is most of what a
campaign decides. These are the judgements it is allowed to make and, more
importantly, the ones it is not.
"""

from __future__ import annotations

from app.services import agent_campaign_priority as priority

TARGETS = ["fusion_candidate"]


def _view(origin="discovered", generation=1, outcome=None, siblings=0):
    return {
        "origin": origin,
        "generation": generation,
        "parent_outcome": outcome,
        "siblings_launched": siblings,
    }


def _outcome(satisfied=True, counts=None):
    return {
        "contract_satisfied": satisfied,
        "finding_counts": counts if counts is not None else {"fusion_candidate": 2},
    }


def test_work_from_a_productive_job_outranks_a_guess_in_the_seed_list():
    productive = _view(outcome=_outcome())
    seed = _view(origin="seed", generation=0)

    assert priority.score(productive, TARGETS)[0] > priority.score(seed, TARGETS)[0]


def test_work_from_a_job_that_produced_nothing_ranks_below_the_seed_list():
    barren = _view(outcome=_outcome(satisfied=False, counts={"document": 4}))
    seed = _view(origin="seed", generation=0)

    assert priority.score(barren, TARGETS)[0] <= priority.score(seed, TARGETS)[0]


def test_an_unsettled_parent_is_not_treated_as_evidence():
    """Absence of an outcome is not a good one."""
    unsettled = _view(outcome=None)
    productive = _view(outcome=_outcome())

    assert (
        priority.score(unsettled, TARGETS)[0] < priority.score(productive, TARGETS)[0]
    )
    assert "not settled" in priority.score(unsettled, TARGETS)[1]


def test_findings_of_the_wrong_type_do_not_count_as_yield():
    """Ten documents are not a result for a campaign hunting fusable pairs."""
    documents = _outcome(satisfied=False, counts={"document": 10})

    assert priority.parent_yield(documents, TARGETS) == 0.0


def test_a_met_contract_counts_even_without_target_findings():
    """Either signal alone is weak, so each is worth half."""
    assert priority.parent_yield(_outcome(satisfied=True, counts={}), TARGETS) == 0.5


def test_depth_is_discounted():
    """A campaign spawning from its own spawn drifts from the goal it was
    given, and nothing else pulls it back."""
    near = _view(generation=1, outcome=_outcome())
    far = _view(generation=4, outcome=_outcome())

    assert priority.score(near, TARGETS)[0] > priority.score(far, TARGETS)[0]
    assert "generation 4" in priority.score(far, TARGETS)[1]


def test_siblings_already_run_discount_the_next_one():
    first = _view(outcome=_outcome(), siblings=0)
    fifth = _view(outcome=_outcome(), siblings=4)

    assert priority.score(first, TARGETS)[0] > priority.score(fifth, TARGETS)[0]


def test_a_campaign_that_never_said_what_it_wanted_counts_any_finding():
    assert priority.parent_yield(_outcome(satisfied=False), []) == 0.5


def test_one_barren_ancestor_is_not_a_cold_line():
    """One bad job is a bad job."""
    barren = _outcome(satisfied=False, counts={})

    assert priority.is_cold([barren], TARGETS) is False


def test_two_barren_ancestors_are_a_cold_line():
    barren = _outcome(satisfied=False, counts={})

    assert priority.is_cold([barren, barren], TARGETS) is True


def test_a_productive_ancestor_breaks_the_run():
    barren = _outcome(satisfied=False, counts={})

    assert priority.is_cold([barren, _outcome(), barren], TARGETS) is False


def test_an_unsettled_ancestor_stops_the_walk_rather_than_condemning_the_line():
    """A line is not cold because its parent has not finished."""
    barren = _outcome(satisfied=False, counts={})

    assert priority.is_cold([None, barren, barren], TARGETS) is False


def test_choose_prefers_the_better_item_over_the_older_one():
    pending = [
        _view(origin="seed", generation=0),
        _view(outcome=_outcome()),
    ]

    assert priority.choose(pending, target_types=TARGETS)["index"] == 1


def test_choose_falls_back_to_creation_order_when_nothing_separates_them():
    """A campaign with nothing to go on behaves exactly as it did before."""
    pending = [_view(origin="seed", generation=0) for _ in range(3)]

    assert priority.choose(pending, target_types=TARGETS)["index"] == 0


def test_choose_says_why():
    pending = [_view(outcome=_outcome())]

    assert (
        "met its contract" in priority.choose(pending, target_types=TARGETS)["reason"]
    )


def test_choose_on_an_empty_backlog_is_no_choice():
    assert priority.choose([], target_types=TARGETS) is None


def test_self_spawned_work_cannot_starve_the_seed_list_forever():
    """A campaign whose jobs spawn their own successors can chase its own tail
    while the work a person actually asked for never starts."""
    pending = [
        _view(outcome=_outcome()),  # would win on score
        _view(origin="seed", generation=0),
    ]
    recent = ["discovered"] * priority.MAX_CONSECUTIVE_DISCOVERED

    chosen = priority.choose(pending, target_types=TARGETS, recent_origins=recent)

    assert chosen["index"] == 1
    assert chosen["starved"] is True


def test_a_short_run_of_self_spawned_work_does_not_trigger_the_guard():
    pending = [_view(outcome=_outcome()), _view(origin="seed", generation=0)]
    recent = ["discovered"] * (priority.MAX_CONSECUTIVE_DISCOVERED - 1)

    assert (
        priority.choose(pending, target_types=TARGETS, recent_origins=recent)["index"]
        == 0
    )


def test_a_seed_run_resets_the_starvation_guard():
    pending = [_view(outcome=_outcome()), _view(origin="seed", generation=0)]
    recent = ["discovered", "discovered", "discovered", "seed"]

    assert (
        priority.choose(pending, target_types=TARGETS, recent_origins=recent)["index"]
        == 0
    )


def test_the_guard_does_nothing_when_no_seed_is_waiting():
    pending = [_view(outcome=_outcome())]
    recent = ["discovered"] * 5

    assert (
        priority.choose(pending, target_types=TARGETS, recent_origins=recent)["index"]
        == 0
    )
