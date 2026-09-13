"""Tests for the run conclusion.

A run that measured nine kernels ended as a table of numbers with nobody
saying what they meant. The conclusion answers the goal from the findings
already recorded, and its failure must cost nothing.
"""

import pytest

from app.services import agent_run_synthesis_service as synthesis


class _Job:
    id = "job-1"
    goal = "Determine whether clang vectorizes a float reduction at -O3"
    iteration = 4


def _state(findings):
    return {"findings": findings}


FINDINGS = [
    {
        "type": "codegen_measurement",
        "title": "float reduction @ clang -O3: 0 vector ops",
    },
    {
        "type": "codegen_measurement",
        "title": "float reduction @ clang -O3 -ffast-math: 17 vector ops",
    },
]


def test_findings_are_rendered_with_their_kind_and_title():
    rows = synthesis.summarize_findings_for_prompt(FINDINGS)

    assert rows[0].startswith("[codegen_measurement] float reduction")


def test_findings_without_a_title_are_skipped_not_rendered_blank():
    rows = synthesis.summarize_findings_for_prompt([{"type": "x"}, "not-a-dict", None])

    assert rows == []


def test_prompt_carries_the_goal_and_the_findings():
    prompt = synthesis.build_conclusion_prompt(_Job.goal, FINDINGS)

    assert "clang vectorizes a float reduction" in prompt
    assert "17 vector ops" in prompt
    assert "FINDINGS RECORDED (2)" in prompt


def test_the_system_prompt_states_the_json_shape():
    """The schema-constrained path is not always available: a provider answered
    a schema request with HTTP 400, and the prompted fallback returned markdown
    that JSON extraction discarded."""
    assert "single JSON object" in synthesis.SYSTEM_PROMPT
    for key in ("answer", "confidence", "evidence", "gaps"):
        assert f'"{key}"' in synthesis.SYSTEM_PROMPT


class _Executor:
    def __init__(self, payload=None, raises=False):
        self.llm_service = object()
        self._payload = payload
        self._raises = raises


@pytest.fixture
def patched(monkeypatch):
    """Patch the real llm_structured module.

    synthesize_conclusion imports it inside the function, so replacing the
    module object in sys.modules leaks into other tests and makes the result
    depend on import order. Patching the attribute is restored automatically.
    """
    from app.services import llm_structured

    def _install(payload=None, raises=False):
        async def _ask(*args, **kwargs):
            if raises:
                raise RuntimeError("provider exploded")
            return payload

        monkeypatch.setattr(llm_structured, "ask_for_json", _ask)

    return _install


@pytest.mark.asyncio
async def test_a_run_with_no_findings_says_so_rather_than_guessing(patched):
    patched(payload={"answer": "should never be used", "confidence": "high"})

    result = await synthesis.synthesize_conclusion(_Executor(), _Job(), _state([]))

    assert result["answer"] is None
    assert result["generated_by"] == "no_evidence"
    assert "no findings" in result["gaps"][0].lower()


@pytest.mark.asyncio
async def test_a_conclusion_is_recorded_with_its_evidence(patched):
    patched(
        payload={
            "answer": "clang does not vectorize the float reduction at -O3.",
            "confidence": "high",
            "evidence": ["float reduction @ clang -O3: 0 vector ops"],
            "gaps": ["Only one kernel was tested."],
        }
    )

    result = await synthesis.synthesize_conclusion(
        _Executor(), _Job(), _state(FINDINGS)
    )

    assert result["generated_by"] == "llm"
    assert result["confidence"] == "high"
    assert result["evidence"] == ["float reduction @ clang -O3: 0 vector ops"]
    assert result["findings_considered"] == 2


@pytest.mark.asyncio
async def test_a_provider_failure_is_recorded_not_raised(patched):
    patched(raises=True)

    result = await synthesis.synthesize_conclusion(
        _Executor(), _Job(), _state(FINDINGS)
    )

    assert result["answer"] is None
    assert result["generated_by"] == "error"
    assert "provider exploded" in result["gaps"][0]


@pytest.mark.asyncio
async def test_an_empty_model_reply_is_not_dressed_up_as_an_answer(patched):
    patched(payload={"answer": "   ", "confidence": "high"})

    result = await synthesis.synthesize_conclusion(
        _Executor(), _Job(), _state(FINDINGS)
    )

    assert result["answer"] is None
    assert result["generated_by"] == "no_evidence"


def test_conclusion_line_reads_cleanly_either_way():
    assert "confidence: high" in synthesis.conclusion_line(
        {"answer": "It does not vectorize.", "confidence": "high"}
    )
    assert synthesis.conclusion_line(
        {"answer": None, "gaps": ["The run recorded no findings."]}
    ).startswith("No conclusion:")


def test_the_numbers_a_finding_carries_reach_the_prompt():
    """Only titles reached the conclusion prompt, so measurements vanished.

    A run concluded "GFLOP/s values were not recorded in the findings" about a
    finding carrying five of them.
    """
    from app.services.agent_run_synthesis_service import summarize_findings_for_prompt

    rows = summarize_findings_for_prompt(
        [
            {
                "type": "benchmark_measurement",
                "title": "dotprod @ clang -O2: fastest 11 ms of 5 trials",
                "fastest_ms": 11,
                "reported_metrics": {
                    "gflops": [0.98, 1.34, 1.53],
                    "runtime_sec": [0.00137, 0.0022],
                },
            }
        ]
    )

    assert "gflops=0.98..1.53 (n=3)" in rows[0]
    assert "fastest_ms=11" in rows[0]
    assert rows[0].startswith("[benchmark_measurement] dotprod @ clang -O2")


def test_a_finding_without_numbers_is_unchanged():
    from app.services.agent_run_synthesis_service import summarize_findings_for_prompt

    rows = summarize_findings_for_prompt(
        [{"type": "paper", "title": "VecTrans", "arxiv_id": "2401.00001"}]
    )

    assert rows == ["[paper] VecTrans"]


def test_metric_rendering_is_bounded_and_skips_non_numbers():
    from app.services.agent_run_synthesis_service import (
        MAX_METRICS_PER_FINDING,
        summarize_finding_metrics,
    )

    finding = {"flag": True, "label": "text", "empty_series": ["a", "b"]}
    finding.update({f"m{i}": i for i in range(20)})

    rendered = summarize_finding_metrics(finding)

    assert "flag" not in rendered and "label" not in rendered
    assert "empty_series" not in rendered
    assert len(rendered.split(", ")) <= MAX_METRICS_PER_FINDING
