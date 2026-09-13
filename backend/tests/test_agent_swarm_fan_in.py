"""Direct tests for the extracted swarm fan-in helpers.

These are pure functions over plain dicts, so they are exercised here without
constructing an executor.
"""

from app.services.agent_swarm_fan_in import (
    build_swarm_fan_in_result,
    normalize_role_token,
)


def test_normalize_role_token_maps_aliases_to_canonical_roles():
    assert normalize_role_token("researcher_documents") == "researcher"
    assert normalize_role_token("Knowledge Researcher") == "researcher"
    assert normalize_role_token("Swarm Agent 2: Analyst") == "critic"
    assert normalize_role_token("monitor") == "verifier"
    assert normalize_role_token("synth") == "synthesizer"
    assert normalize_role_token("patcher") == "coder"


def test_normalize_role_token_is_empty_for_blank_input():
    assert normalize_role_token(None) == ""
    assert normalize_role_token("   ") == ""
    assert normalize_role_token("!!!") == ""


def test_normalize_role_token_passes_through_unknown_roles():
    assert normalize_role_token("Domain Expert") == "domain_expert"


FINDING = "Retry budget is exhausted early"


def _sibling(job_id, role, *, title, status="completed"):
    return {
        "job_id": job_id,
        "role": role,
        "status": status,
        "results": {"findings": [{"title": title}]},
    }


def _without_timestamp(result):
    return {k: v for k, v in result.items() if k != "generated_at"}


def test_fan_in_is_deterministic_apart_from_its_timestamp():
    payload = {
        "expected_siblings": 2,
        "terminal_siblings": 2,
        "sibling_jobs": [
            _sibling("a", "Researcher", title=FINDING),
            _sibling("b", "Analyst", title="Backoff resets on every attempt"),
        ],
    }

    first = build_swarm_fan_in_result(payload, fan_in_group_id="group-1")
    second = build_swarm_fan_in_result(payload, fan_in_group_id="group-1")

    assert _without_timestamp(first) == _without_timestamp(second)
    assert first["fan_in_group_id"] == "group-1"


def test_agreed_findings_become_consensus_with_sorted_supporting_roles():
    payload = {
        "expected_siblings": 2,
        "terminal_siblings": 2,
        "sibling_jobs": [
            _sibling("a", "Researcher", title=FINDING),
            _sibling("b", "Analyst", title=FINDING),
        ],
    }

    result = build_swarm_fan_in_result(payload)

    assert result["consensus_findings"] == [
        {
            "finding": FINDING,
            "support_count": 2,
            "supporting_roles": ["Analyst", "Researcher"],
        }
    ]
    assert result["confidence"]["agreement"] == 1.0


def test_merged_conclusions_do_not_depend_on_sibling_order():
    """Consensus and confidence are conclusions, so order must not move them.

    The per-sibling listings (roles, role_summaries, sibling_status) do mirror
    the input order by design, and are excluded here.
    """
    a = _sibling("a", "Researcher", title=FINDING)
    b = _sibling("b", "Analyst", title=FINDING)

    def _run(siblings):
        return build_swarm_fan_in_result(
            {
                "expected_siblings": 2,
                "terminal_siblings": 2,
                "sibling_jobs": siblings,
            }
        )

    forward, reverse = _run([a, b]), _run([b, a])

    assert forward["consensus_findings"] == reverse["consensus_findings"]
    assert forward["confidence"] == reverse["confidence"]
    assert sorted(forward["roles"]) == sorted(reverse["roles"])


def test_disagreeing_siblings_are_reported_as_a_conflict():
    payload = {
        "expected_siblings": 2,
        "terminal_siblings": 2,
        "sibling_jobs": [
            _sibling("a", "Researcher", title=FINDING),
            _sibling("b", "Analyst", title="Something entirely unrelated"),
        ],
    }

    result = build_swarm_fan_in_result(payload)

    assert result["consensus_findings"] == []
    assert [c["type"] for c in result["conflicts"]] == ["low_alignment"]
    assert result["confidence"]["agreement"] == 0.0


def test_fan_in_tolerates_missing_and_malformed_siblings():
    payload = {
        "expected_siblings": 3,
        "terminal_siblings": 1,
        "sibling_jobs": [
            _sibling("a", "Researcher", title=FINDING),
            {"job_id": "b"},  # no role, no results
            "not-a-dict",
        ],
    }

    result = build_swarm_fan_in_result(payload)

    assert result["received_siblings"] == 3
    assert result["expected_siblings"] == 3
    # Current contract: coverage counts entries received, not entries that
    # carried usable results, so unusable siblings still score a full 1.0.
    assert result["confidence"]["coverage"] == 1.0
    assert result["consensus_findings"] == []


def test_fan_in_does_not_mutate_the_payload():
    payload = {
        "expected_siblings": 1,
        "terminal_siblings": 1,
        "sibling_jobs": [_sibling("a", "Researcher", title=FINDING)],
    }
    before = repr(payload)

    build_swarm_fan_in_result(payload)

    assert repr(payload) == before
