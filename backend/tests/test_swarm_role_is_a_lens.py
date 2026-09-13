"""A swarm role is a perspective on the task, not a different task.

The child goal used to read:

    Swarm role: Researcher
    Objective: Gather high-signal evidence from papers and internal knowledge…
    Parent goal: <what was actually asked>

The role's boilerplate was the instruction and the real task was filed as
background. Measured: a swarm told to benchmark one C kernel exactly once came
back with ten knowledge-base documents -- the `researcher` template's own
objective, and its own `max_documents: 10`.

That also breaks the premise of the whole feature. Agreement between roles is
only evidence if the roles were answering the same question.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob
from app.services.agent_swarm_chain_config import ensure_swarm_chain_config

pytestmark = pytest.mark.unit


def _parent(job_type="analysis", roles=("researcher", "verifier")):
    job = AgentJob(
        id=uuid.uuid4(),
        name="swarm parent",
        goal='Call benchmark_c_snippet EXACTLY ONCE with label="dotprod".',
        job_type=job_type,
        status="running",
        iteration=1,
        max_iterations=4,
        config={
            "swarm_child_jobs_enabled": True,
            "swarm_roles": list(roles),
            "swarm_max_agents": len(roles),
        },
        results={},
        execution_log=[],
    )
    job.user_id = uuid.uuid4()
    return job


def _children(job):
    state = {}
    ensure_swarm_chain_config(job, state, append_step_event=lambda *a, **k: None)
    chain = job.chain_config or {}
    return [c for c in (chain.get("child_jobs") or []) if isinstance(c, dict)]


class TestTheTaskLeads:
    def test_the_parent_goal_comes_first(self):
        children = _children(_parent())

        assert children, "the swarm must plan children at all"
        for child in children:
            goal = str(child.get("goal") or "")
            assert goal.startswith(
                "Call benchmark_c_snippet"
            ), "the task is the instruction; the role is the lens"

    def test_the_role_is_present_but_subordinate(self):
        children = _children(_parent())
        goal = str(children[0].get("goal") or "")

        assert "one of several agents" in goal.lower()
        assert "do not substitute" in goal.lower()

    def test_the_goal_asks_for_comparable_labels(self):
        """Consensus groups on subject. If the roles label the same thing
        differently there is nothing to compare, which is what happened: one
        agent invented four subject names in a single run."""
        goal = str(_children(_parent())[0].get("goal") or "")

        assert "exactly those" in goal.lower()


class TestARoleDoesNotChangeWhatTheAgentCanDo:
    def test_an_analysis_swarm_stays_analysis(self):
        """Job type decides which tools a run can see. Every role template
        declares `research`, so an analysis swarm silently became a research
        swarm and lost the tools its goal depended on."""
        children = _children(_parent(job_type="analysis"))

        assert {str(c.get("job_type")) for c in children} == {"analysis"}

    def test_a_coding_swarm_stays_coding(self):
        children = _children(_parent(job_type="coding"))

        assert {str(c.get("job_type")) for c in children} == {"coding"}

    def test_a_research_parent_still_lets_roles_specialise(self):
        """When the parent never chose a specialisation, the template's own
        job type is the best information available."""
        children = _children(_parent(job_type="research"))

        assert children
        assert all(str(c.get("job_type")) for c in children)


class TestGeneralRolesDoNotPrescribeLiteratureWork:
    """The role objectives were research TASKS, not perspectives.

    `researcher` read "Gather high-signal evidence from papers and internal
    knowledge sources". Applied to "benchmark this kernel exactly once", that
    is an instruction to go and find documents -- and the agent obeyed it,
    returning eight `document` findings and no measurement, twice.

    Reordering the prompt did not help, because the sentence itself named the
    wrong work. A general role has to describe a STANCE that applies to
    whatever the goal is. The source-specific roles are exempt: being about
    documents or arXiv is their whole point.
    """

    GENERAL = ("researcher", "critic", "verifier", "synthesizer")
    LITERATURE_WORDS = ("paper", "document", "knowledge source", "knowledge-base")

    def _goal_for(self, role):
        job = _parent(job_type="analysis", roles=(role,))
        children = _children(job)
        assert children, f"no child planned for role {role}"
        return str(children[0].get("goal") or "").lower()

    @pytest.mark.parametrize("role", GENERAL)
    def test_a_general_role_does_not_send_the_agent_to_the_library(self, role):
        goal = self._goal_for(role)

        offenders = [w for w in self.LITERATURE_WORDS if w in goal]
        assert not offenders, (
            f"role {role!r} steers toward {offenders}; a general role must "
            "describe how to approach the goal, not replace it with reading"
        )

    @pytest.mark.parametrize("role", GENERAL)
    def test_the_task_still_leads(self, role):
        assert self._goal_for(role).startswith("call benchmark_c_snippet")

    def test_a_literature_role_is_still_allowed_to_be_one(self):
        """`researcher_arxiv` exists precisely to go to the literature."""
        goal = self._goal_for("researcher_arxiv")

        assert "paper" in goal
