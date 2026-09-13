"""How a run records a conclusion, and how often.

Two defects a live autonomous study exposed. Every derived finding appeared
twice, because the tool wrote it into state and then returned it for the
executor to write again. And a conclusion could carry neither a type nor a
number, so the validity predicates -- which bound named fields on typed
findings -- could police the measurements a tool emitted but never the claim
the run drew from them.
"""

import pytest

from app.services import agent_measurement_validity as validity
from app.services.agent_tool_dispatch import build_autonomous_research_provider


class _Job:
    id = "job-1"


class _Ctx:
    def __init__(self, state):
        self.job = _Job()
        self.state = state
        self.mode = "autonomous"
        self.db = None
        self.service = None
        self.user_id = None


class _Executor:
    def __init__(self):
        self._job_findings = {}


async def _save(params, state):
    provider = build_autonomous_research_provider(_Executor())
    handler = provider._handlers["save_research_finding"]
    return await handler(params, _Ctx(state))


BASE = {"title": "t", "content": "c", "category": "result"}


class TestRecordedOnce:
    @pytest.mark.asyncio
    async def test_the_tool_returns_the_finding_without_also_writing_state(self):
        """The executor extends state from what a tool returns, for every tool.
        Doing both here is what produced identical pairs."""
        state = {"findings": []}
        result = await _save(BASE, state)

        assert len(result["findings"]) == 1
        assert state["findings"] == [], "the executor records it, not the tool"

    @pytest.mark.asyncio
    async def test_recording_through_the_executor_path_yields_one(self):
        state = {"findings": []}
        result = await _save(BASE, state)
        state["findings"].extend(result["findings"])  # what the executor does

        assert len(state["findings"]) == 1


class TestTypedConclusions:
    @pytest.mark.asyncio
    async def test_a_finding_can_declare_its_type(self):
        result = await _save({**BASE, "finding_type": "latency_measurement"}, {})
        assert result["findings"][0]["type"] == "latency_measurement"

    @pytest.mark.asyncio
    async def test_a_finding_without_a_type_says_none_rather_than_guessing(self):
        result = await _save(BASE, {})
        assert result["findings"][0]["type"] is None

    @pytest.mark.asyncio
    async def test_numbers_are_carried_where_a_contract_can_read_them(self):
        result = await _save(
            {
                **BASE,
                "finding_type": "latency_measurement",
                "metrics": {"cycles_per_multiply": 6.0},
            },
            {},
        )
        finding = result["findings"][0]
        assert validity._find_number(finding, "cycles_per_multiply") == 6.0

    @pytest.mark.asyncio
    async def test_a_bound_can_now_refuse_a_wrong_conclusion(self):
        """The gap this closes. An autonomous run reported 13.517 cycles per
        multiply -- the true figure is 6 -- and no predicate could see it,
        because the claim carried no type and no readable number."""
        contract = {
            "validity": {
                "bounds": {
                    "latency_measurement": {
                        "field": "cycles_per_multiply",
                        "min": 1,
                        "max": 10,
                    }
                }
            }
        }
        wrong = await _save(
            {
                **BASE,
                "finding_type": "latency_measurement",
                "metrics": {"cycles_per_multiply": 13.517},
            },
            {},
        )
        right = await _save(
            {
                **BASE,
                "finding_type": "latency_measurement",
                "metrics": {"cycles_per_multiply": 6.0},
            },
            {},
        )

        assert validity.evaluate(contract, {"findings": wrong["findings"]})["missing"]
        assert not validity.evaluate(contract, {"findings": right["findings"]})[
            "missing"
        ]

    @pytest.mark.asyncio
    async def test_metrics_that_are_not_a_mapping_are_ignored_not_stored(self):
        result = await _save({**BASE, "metrics": "six"}, {})
        assert result["findings"][0]["metrics"] == {}
