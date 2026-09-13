"""Carrying a pipeline stage's assumed evidence into the job that runs it.

A stage declares `assumes: [...]`, the binding writes it to the job config as
`pipeline_assumes`, and nothing read it -- the same shape of bug as
`loop_until` before it: a documented option that silently did nothing.

Each stage runs as its own job with its own empty state, so evidence produced
upstream was invisible downstream. Observed end to end: a `compare` stage that
assumed an implementation had been verified found no such finding in its own
run and returned `incomparable`, refusing to score the measurement. The
refusal was right; the situation it was put in was not.

The rule that makes this safe is the one the planner already uses. Perishable
evidence -- a passing test, a correctness check -- describes a tree as it stood
and is NOT inherited: a stage that edits or rebuilds the code has to establish
it again. Durable evidence -- an ingested paper, an extracted specification --
crosses the boundary.
"""

import uuid

import pytest

from app.services.autonomous_agent_executor import AutonomousAgentExecutor

pytestmark = pytest.mark.unit


class _Job:
    def __init__(self, config=None, parent=None, findings=None):
        self.id = uuid.uuid4()
        self.config = config or {}
        self.parent_job_id = parent.id if parent else None
        self.results = {"findings": findings or []}
        self.execution_log = []

    def add_log_entry(self, entry):
        self.execution_log.append(entry)


class _Result:
    def __init__(self, obj):
        self._obj = obj

    def scalar_one_or_none(self):
        return self._obj


class _Db:
    """Resolves ancestors by id, the way the real query does."""

    def __init__(self, jobs):
        self._by_id = {str(j.id): j for j in jobs}
        self.queries = 0

    async def execute(self, statement):
        # The statement is `select(AgentJob).where(AgentJob.id == <uuid>)`;
        # read the bound value rather than re-implementing SQLAlchemy.
        self.queries += 1
        wanted = None
        for param in statement.compile().params.values():
            wanted = str(param)
        return _Result(self._by_id.get(wanted))


async def _inherit(job, db):
    state = {"findings": []}
    await AutonomousAgentExecutor._inherit_assumed_findings(
        AutonomousAgentExecutor.__new__(AutonomousAgentExecutor), job, state, db
    )
    return state


class TestDurableEvidenceCrossesTheBoundary:
    @pytest.mark.asyncio
    async def test_an_assumed_specification_is_inherited(self):
        parent = _Job(
            findings=[
                {"type": "algorithm_spec", "algorithm_name": "fastmod"},
                {"type": "document", "id": "noise"},
            ]
        )
        child = _Job(config={"pipeline_assumes": ["algorithm_spec"]}, parent=parent)
        state = await _inherit(child, _Db([parent, child]))

        assert [f["type"] for f in state["findings"]] == ["algorithm_spec"]
        assert state["findings"][0]["algorithm_name"] == "fastmod"

    @pytest.mark.asyncio
    async def test_it_is_marked_as_someone_else_s_work(self):
        # "This run measured it" and "an earlier stage did" are different
        # claims, and a run must not be able to report the second as the first.
        parent = _Job(findings=[{"type": "algorithm_spec"}])
        child = _Job(config={"pipeline_assumes": ["algorithm_spec"]}, parent=parent)
        state = await _inherit(child, _Db([parent, child]))

        carried = state["findings"][0]
        assert carried["inherited"] is True
        assert carried["inherited_from_job_id"] == str(parent.id)

    @pytest.mark.asyncio
    async def test_it_reaches_back_past_the_immediate_parent(self):
        # A stage may assume evidence from further back than one stage.
        grandparent = _Job(findings=[{"type": "papers_ingested", "n": 1}])
        parent = _Job(parent=grandparent, findings=[{"type": "document"}])
        child = _Job(config={"pipeline_assumes": ["papers_ingested"]}, parent=parent)
        state = await _inherit(child, _Db([grandparent, parent, child]))

        assert [f["type"] for f in state["findings"]] == ["papers_ingested"]


class TestPerishableEvidenceMustBeReEarned:
    @pytest.mark.asyncio
    async def test_a_correctness_check_is_not_inherited(self):
        """The rule that keeps this honest.

        A stage that rebuilds or edits the code cannot lean on a check that
        passed against the code as it was two stages ago.
        """
        parent = _Job(
            findings=[
                {"type": "implementation_verified", "verified": True},
                {"type": "algorithm_spec"},
            ]
        )
        child = _Job(
            config={"pipeline_assumes": ["implementation_verified", "algorithm_spec"]},
            parent=parent,
        )
        state = await _inherit(child, _Db([parent, child]))

        types = [f["type"] for f in state["findings"]]
        assert "algorithm_spec" in types
        assert "implementation_verified" not in types

    @pytest.mark.asyncio
    async def test_the_run_is_told_what_it_must_re_establish(self):
        # Silence here would leave a stage wondering why the evidence it was
        # promised is missing.
        parent = _Job(findings=[{"type": "implementation_verified"}])
        child = _Job(
            config={"pipeline_assumes": ["implementation_verified"]}, parent=parent
        )
        await _inherit(child, _Db([parent, child]))

        entries = [
            e
            for e in child.execution_log
            if e.get("phase") == "assumed_evidence_not_inherited"
        ]
        assert entries and entries[0]["types"] == ["implementation_verified"]


class TestItDoesNothingWhenThereIsNothingToDo:
    @pytest.mark.asyncio
    async def test_a_job_with_no_assumptions(self):
        parent = _Job(findings=[{"type": "algorithm_spec"}])
        child = _Job(parent=parent)
        db = _Db([parent, child])
        state = await _inherit(child, db)
        assert state["findings"] == []
        assert db.queries == 0, "it queried the chain for nothing"

    @pytest.mark.asyncio
    async def test_a_root_job_has_no_chain_to_inherit_from(self):
        root = _Job(config={"pipeline_assumes": ["algorithm_spec"]})
        state = await _inherit(root, _Db([root]))
        assert state["findings"] == []

    @pytest.mark.asyncio
    async def test_an_assumption_the_chain_never_produced(self):
        # Not an error: the stage will have to obtain it, and its contract is
        # what decides whether that mattered.
        parent = _Job(findings=[{"type": "document"}])
        child = _Job(config={"pipeline_assumes": ["algorithm_spec"]}, parent=parent)
        state = await _inherit(child, _Db([parent, child]))
        assert state["findings"] == []
