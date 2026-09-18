"""A contract may only require evidence some tool can actually produce.

A stage validated, planned and started with a contract nothing it could call
would ever satisfy. `literature_review` names two producers, both reachable only
from chat and MCP, and the pipeline check read their empty `job_types` tuple as
"no restriction" rather than "no autonomous job type" -- the two cases that tuple
distinguishes. The stage could not have completed under any job type.

The last class is the one that matters most: it asks the real vocabulary and the
real catalog whether every evidence type is producible at all, so a tool being
withdrawn from autonomous jobs shows up here rather than in a stuck run.
"""

from app.agent_core import tool_specs
from app.services import agent_pipeline_spec as spec_module
from app.services import agent_pipeline_vocabulary as vocabulary


def _stage(finding_type: str, job_type: str = "synthesis") -> dict:
    return {
        "name": "p",
        "stages": [
            {
                "id": "report",
                "goal": "Write it up",
                "job_type": job_type,
                "contract": {"required_finding_type_counts": {finding_type: 1}},
            }
        ],
    }


def _problems(finding_type: str, job_type: str = "synthesis"):
    return spec_module.validate(spec_module.normalize(_stage(finding_type, job_type)))


class TestAContractNothingCanSatisfy:
    def test_it_is_refused(self):
        assert _problems(
            "literature_review"
        ), "a stage requiring evidence no callable tool produces must not validate"

    def test_the_message_names_the_evidence(self):
        (problem,) = _problems("literature_review")
        assert "literature_review" in problem

    def test_the_message_says_no_job_type_would_help(self):
        # The author's next move differs entirely: a wrong job_type is a one
        # word fix, while evidence nothing can produce needs a new contract.
        (problem,) = _problems("literature_review")
        assert "no job type can" in problem


class TestWhatItMustNotRefuse:
    def test_evidence_with_one_reachable_producer_is_fine(self):
        # papers_ingested names ingest_arxiv_papers, which no autonomous job may
        # call, and ingest_paper_by_id, which research may. One reachable
        # producer is enough; flagging the barred one would refuse a contract
        # that works.
        assert _problems("papers_ingested", "research") == []

    def test_ordinary_evidence_still_validates(self):
        assert _problems("synthesis_document") == []


class TestTheEmptyTupleIsNotNoRestriction:
    """`job_types is None` means every job type; `()` means none."""

    def test_the_catalog_distinguishes_them(self):
        catalog = tool_specs.STATIC_CATALOG
        chat_only = catalog.spec_for("literature_review_arxiv")
        assert chat_only.job_types == ()
        for job_type in vocabulary.job_types():
            assert "literature_review_arxiv" not in catalog.tools_for_job_type(job_type)


class TestEveryEvidenceTypeIsProducibleBySomething:
    """Asked of the real vocabulary and the real catalog.

    A tool withdrawn from autonomous jobs silently strands every contract that
    required what it produced. This is where that shows up.
    """

    def test_no_evidence_type_is_unreachable_from_every_job_type(self):
        catalog = tool_specs.STATIC_CATALOG
        reachable = {
            job_type: set(catalog.tools_for_job_type(job_type))
            for job_type in vocabulary.job_types()
        }
        stranded = []
        for evidence in vocabulary.evidence_types():
            if not evidence.producers:
                continue
            if not any(
                producer in tools
                for tools in reachable.values()
                for producer in evidence.producers
            ):
                stranded.append((evidence.name, list(evidence.producers)))

        assert stranded == [
            (
                "literature_review",
                ["literature_review_arxiv", "generate_literature_review_for_source"],
            )
        ], (
            "A contract requiring these can never be satisfied by an autonomous "
            "job. Either give a producer a job_types allowance, or stop "
            "advertising the evidence. literature_review is the known case and "
            "is refused by validate(); anything else here is new."
        )


class TestTheRunIsOnlyToldAboutToolsItCanCall:
    """The evidence guidance in the thinking prompt is advice a run follows.

    A stage requiring papers_ingested was told "ingest_arxiv_papers (or
    ingest_paper_by_id) yields papers_ingested" while its job type could call
    only the second. The recommended tool, named first, would have been refused.
    That run called search_arxiv once, found 18 papers, and spent its remaining
    rounds on web search and progress reports without ingesting one.
    """

    def test_it_leads_with_a_tool_the_job_type_may_call(self):
        from app.services import agent_evidence_map

        (line,) = agent_evidence_map.describe_chain(
            ["papers_ingested"], job_type="research"
        )
        assert line.startswith("ingest_paper_by_id")
        assert "ingest_arxiv_papers" not in line

    def test_it_describes_the_tool_it_actually_recommends(self):
        # The inputs differ: one takes a query or a list, the other one id. The
        # run has to satisfy the inputs of the tool it is being sent to.
        from app.services import agent_evidence_map

        (line,) = agent_evidence_map.describe_chain(
            ["papers_ingested"], job_type="research"
        )
        assert "One arXiv id" in line

    def test_it_says_nothing_when_nothing_can_produce_it(self):
        # validate() refuses such a contract, so this is the belt to that
        # braces. Silence beats advertising a door that does not open.
        from app.services import agent_evidence_map

        assert (
            agent_evidence_map.describe_chain(["papers_ingested"], job_type="coding")
            == []
        )

    def test_unfiltered_still_names_every_route(self):
        # Callers without a job type in hand keep the old behaviour.
        from app.services import agent_evidence_map

        (line,) = agent_evidence_map.describe_chain(["papers_ingested"])
        assert "ingest_arxiv_papers" in line and "ingest_paper_by_id" in line
