"""Guards on the AXIS tools.

The docker-backed paths are exercised through their preflight and parsing; the
container itself is not run here.
"""

import pytest

from app.agent_core import tool_specs
from app.services import agent_axis_sandbox as axis


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(axis.agent_sandbox_runtime, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        axis.agent_sandbox_runtime, "allowed_images", lambda: [axis.DEFAULT_IMAGE]
    )


@pytest.mark.asyncio
async def test_an_empty_description_is_rejected_before_a_container_starts(enabled):
    assert "source is required" in (await axis.check_description(source=" "))["error"]


@pytest.mark.asyncio
async def test_execution_must_be_enabled(monkeypatch):
    monkeypatch.setattr(axis.agent_sandbox_runtime, "execution_enabled", lambda: False)

    result = await axis.check_description(source="(defextension foo)")

    assert "ENABLE_UNSAFE_CODE_EXECUTION" in result["error"]


@pytest.mark.asyncio
async def test_an_unlisted_image_is_refused(enabled):
    result = await axis.check_description(
        source="(defextension foo)", image="evil:latest"
    )

    assert "not allowlisted" in result["error"]


@pytest.mark.asyncio
async def test_an_unknown_emit_target_lists_what_is_available(enabled):
    result = await axis.emit_artifact(source="(defextension foo)", target="wat")

    assert "Unknown emit target" in result["error"]
    assert "smt2" in result["error"]


@pytest.mark.asyncio
async def test_a_proof_without_check_sat_asks_the_solver_nothing(enabled):
    """An obligation that never calls check-sat returns no verdict at all."""
    result = await axis.prove_equivalence(
        source="(defextension foo)", obligation="(assert true)"
    )

    assert "(check-sat)" in result["error"]


@pytest.mark.asyncio
async def test_a_missing_obligation_explains_what_unsat_would_mean(enabled):
    result = await axis.prove_equivalence(source="(defextension foo)", obligation="")

    assert "negation" in result["error"]


def test_the_solver_verdict_is_read_from_its_output():
    assert axis.parse_solver_verdict("unsat\n") == "unsat"
    assert axis.parse_solver_verdict("warning: blah\nsat\n") == "sat"
    assert axis.parse_solver_verdict('(error "line 3")\n') == "error"
    assert axis.parse_solver_verdict("") == "error"


def test_unknown_is_not_treated_as_proved():
    """A solver that gave up has neither proved nor disproved the claim."""
    assert axis.parse_solver_verdict("unknown") == "unknown"
    assert axis.parse_solver_verdict("unknown") != "unsat"


def test_a_solver_timeout_is_a_verdict_and_not_a_broken_obligation():
    """z3 prints `timeout` when -T: expires, and it says nothing about the query.

    Reading it as unparseable output produced "the obligation probably does not
    typecheck against the emitted semantics" -- for an obligation structurally
    identical to one the same solver had just discharged as unsat. The caller is
    then sent to rewrite something correct, which is the worst available advice
    because the real remedy is more time or a narrower query.
    """
    assert axis.parse_solver_verdict("timeout\n") == "timeout"
    assert axis.parse_solver_verdict("timeout") != "error"


def test_a_timeout_is_not_treated_as_proved():
    """Not settled is not proved, however the solver failed to settle it."""
    assert axis.parse_solver_verdict("timeout") != "unsat"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,call",
    [
        ("axis_check", lambda: axis.check_description(source="(defextension foo)")),
        (
            "axis_prove",
            lambda: axis.prove_equivalence(
                source="(defextension foo)", obligation="(check-sat)"
            ),
        ),
        (
            "axis_emit",
            lambda: axis.emit_artifact(source="(defextension foo)", target="smt2"),
        ),
    ],
)
async def test_the_evidence_a_tool_emits_is_the_evidence_it_declares(
    enabled, monkeypatch, tool, call
):
    """A spec's `produces` is what contracts are written against; the handler's
    finding type is what a run records. When they disagree the stage runs the
    tool, succeeds, and still ends contract-unmet.

    They did disagree. axis_check declared `axis_description` and emitted
    `axis_description_valid`; axis_prove declared `equivalence_proof` and
    emitted `axis_equivalence_proof`; axis_emit declared nothing and emitted
    `axis_artifact`, so no contract could require what it demonstrably
    produces. Nothing caught it because both halves were independently
    reasonable -- only running the tool and reading the spec together shows it.
    """

    async def fake_run(script, workdir, **kwargs):
        return 0, "unsat" if "z3" in script else "ok: model.axisl", ""

    monkeypatch.setattr(axis.agent_sandbox_runtime, "run_in_sandbox", fake_run)

    result = await call()

    assert result.get("success") is True, result
    emitted = {f["type"] for f in result.get("findings", [])}
    declared = set(next(s for s in tool_specs.all_specs() if s.name == tool).produces)
    assert emitted == declared, f"{tool}: emits {emitted}, declares {declared}"


@pytest.mark.asyncio
async def test_a_proof_carries_the_question_it_answered(enabled, monkeypatch):
    """ "Equivalence proved for all inputs" names no equivalence.

    A live run recorded exactly that, with verdict unsat, and the obligation was
    recoverable from nowhere: the action ledger drops tool input by design, the
    audit log does not cover the autonomous dispatch path, and the LLM snapshots
    hold the plan rather than the executed arguments. A reader of the corpus
    could not tell a real result from a tautology, which makes the finding
    unusable as evidence however true it is.
    """

    async def fake_run(script, workdir, **kwargs):
        return 0, "unsat", ""

    monkeypatch.setattr(axis.agent_sandbox_runtime, "run_in_sandbox", fake_run)
    obligation = "(assert (not (= (axis_instr_a x) (axis_instr_b x))))\n(check-sat)"

    result = await axis.prove_equivalence(
        source="(defextension foo)", obligation=obligation
    )

    finding = result["findings"][0]
    assert finding["obligation"] == obligation
    assert finding["proved"] is True


@pytest.mark.asyncio
async def test_an_enormous_obligation_is_clipped_not_carried_whole(
    enabled, monkeypatch
):
    async def fake_run(script, workdir, **kwargs):
        return 0, "unsat", ""

    monkeypatch.setattr(axis.agent_sandbox_runtime, "run_in_sandbox", fake_run)
    huge = "(assert true)\n" * 5000 + "(check-sat)"

    result = await axis.prove_equivalence(source="(defextension foo)", obligation=huge)

    assert len(result["findings"][0]["obligation"]) <= axis.MAX_OBLIGATION_CHARS


def test_every_emit_target_maps_to_a_real_axis_command():
    for target, command in axis.EMIT_TARGETS.items():
        assert command.startswith("emit-"), target
        assert " " not in command, target
