"""The failure that is precise, stable, reproducible and about something else.

Four of nine per-instruction latencies in this project, and an entire held-out
validation, were measurements of infinity arithmetic. Every control passed.
Every count, bound and uncertainty requirement was satisfied. It reproduced
perfectly, because it was wrong the same way every time.
"""

from __future__ import annotations

from app.services import agent_measurement_sanity as sanity
from app.services import agent_measurement_validity as validity

MIXED = """
int main(void) {
    asm volatile("fmov s0, #1.0" ::: "s0");
    for (long i = 0; i < 100000; i++) {
        asm volatile(
            "fmul s0, s0, s0\\n\\t"
            "fadd s0, s0, s0\\n\\t"
            "fsqrt s0, s0\\n\\t"
            "fmul s0, s0, s0\\n\\t"
            "fadd s0, s0, s0\\n\\t"
            "fmadd s0, s0, s0, s0\\n\\t"
            ::: "s0");
    }
}
"""

STABLE = (
    MIXED.replace("fmul s0, s0, s0", "fmul s0, s0, s1")
    .replace("fadd s0, s0, s0", "fadd s0, s0, s2")
    .replace("fmadd s0, s0, s0, s0", "fmadd s0, s0, s1, s2")
)


def _chain(*ops: str) -> str:
    body = "".join(f'            "{op}\\n\\t"\n' for op in ops)
    return (
        'int main(void) {\n    asm volatile("fmov s0, #1.0" ::: "s0");\n'
        "    for (long i = 0; i < 100000; i++) {\n        asm volatile(\n"
        f'{body}            ::: "s0");\n    }}\n}}\n'
    )


def test_the_real_defect_is_caught():
    """The held-out kernel: infinity at iteration 4 of 100,000."""
    verdict = sanity.chain_leaves_normal_range(MIXED)

    assert verdict is not None
    assert verdict["iterations_before_leaving"] == 4
    assert verdict["reached"] == "inf"


def test_the_documented_overflow_points_are_reproduced():
    """fadd doubles and goes at 128; fmadd at 8. Both were in the table."""
    assert len(sanity.simulate(["fadd"], 1.0)) == 128
    assert len(sanity.simulate(["fmadd"], 1.0)) == 8


def test_the_three_lucky_classes_are_not_flagged():
    """1.0 is a fixed point of x*x, sqrt(x) and x/x. Nothing chose them for
    that, which is why this is checked per sequence."""
    for op in ("fmul", "fsqrt", "fdiv"):
        assert sanity.chain_leaves_normal_range(_chain(f"{op} s0, s0, s0")) is None


def test_the_corrected_form_passes():
    """Keeping the dependence and not the value: neutral operands held in
    another register."""
    assert sanity.chain_leaves_normal_range(STABLE) is None


def test_a_single_precision_width_is_used():
    """fadd overflows at 2**128 in float32 and not until 2**1024 in a double.
    A double-precision check calls this chain stable at any horizon."""
    assert sanity.f32(3.5e38) == float("inf")
    assert sanity.f32(1.0) == 1.0


def test_the_horizon_is_the_length_of_a_real_loop():
    """An 8-iteration walk reports fadd stable when it overflows at 128."""
    assert sanity.HORIZON >= 200
    assert len(sanity.simulate(["fadd"], 1.0, horizon=8)) == 8


def test_the_remedy_names_what_to_do():
    verdict = sanity.chain_leaves_normal_range(MIXED)

    assert "neutral second operand" in verdict["reason"]
    assert "clobber list" in verdict["reason"]


def test_an_unanalysable_program_is_not_called_sound():
    """'Checked and fine' and 'could not check' must never read the same."""
    assert sanity.find_chain("int main(void){return 0;}") is None
    assert sanity.chain_leaves_normal_range("int main(void){return 0;}") is None
    assert sanity.check({"code": "int main(void){return 0;}"}, {})["checked"] is False


def test_operations_reading_elsewhere_are_recorded_as_opaque():
    """A value-stable chain looks exactly like this and is the correct way to
    write one, so it is noted rather than simulated past."""
    chain = sanity.find_chain(STABLE)

    assert chain is not None
    assert chain.opaque, "ops reading s1/s2 cannot be simulated"


# --- reported metrics ----------------------------------------------------


def test_a_non_finite_reported_metric_is_caught():
    """A harness printing ns_per_op=inf has said its measurement failed."""
    result = {"data": {"reported_metrics": {"ns_per_op": [float("inf")], "ok": [4.0]}}}

    assert sanity.nonfinite_metrics(result) == ["ns_per_op"]


def test_a_nan_metric_is_caught():
    result = {"data": {"reported_metrics": {"ratio": float("nan")}}}

    assert sanity.nonfinite_metrics(result) == ["ratio"]


def test_finite_metrics_pass():
    result = {"data": {"reported_metrics": {"ns_per_op": [4.0, 4.1]}}}

    assert sanity.nonfinite_metrics(result) == []


def test_check_combines_both():
    verdict = sanity.check(
        {"code": MIXED},
        {"data": {"reported_metrics": {"ns_per_op": [float("inf")]}}},
    )

    assert verdict["sound"] is False
    assert len(verdict["problems"]) == 2


def test_a_sound_measurement_is_sound():
    verdict = sanity.check(
        {"code": STABLE}, {"data": {"reported_metrics": {"ns_per_op": [4.0]}}}
    )

    assert verdict["checked"] is True
    assert verdict["sound"] is True


# --- the contract predicate ----------------------------------------------


def _state(record: dict) -> dict:
    return {
        "findings": [
            {
                "type": "benchmark_measurement",
                "title": "fsqrt latency",
                "measurement_sanity": record,
            }
        ]
    }


def test_a_contract_can_require_measuring_what_it_names():
    state = _state(
        {"sound": False, "problems": ["the chain reaches infinity at iteration 4"]}
    )

    result = validity.evaluate({"validity": {"measures_what_it_names": True}}, state)

    assert "validity:measures_what_it_names" in result["missing"]


def test_a_sound_measurement_satisfies_it():
    state = _state({"sound": True, "problems": []})

    result = validity.evaluate({"validity": {"measures_what_it_names": True}}, state)

    assert result["missing"] == []


def test_the_remedy_carries_the_problem():
    state = _state(
        {
            "sound": False,
            "problems": ["the chain on s0 reaches infinity at iteration 4"],
        }
    )
    result = validity.evaluate({"validity": {"measures_what_it_names": True}}, state)

    lines = validity.explain(result["missing"], result["details"])

    assert lines and "fsqrt latency" in lines[0]
    assert "iteration 4" in lines[0]


def test_it_is_described_for_the_prompt():
    described = validity.describe({"measures_what_it_names": True})

    assert any("normal range" in line for line in described)


def test_this_is_the_gap_controls_and_replication_leave():
    """Pinned together so the three are never confused: the defective chain is
    perfectly reproducible and would pass any instrument control."""
    from app.services import agent_measurement_replication as replication

    identical = [{"cycles_per_op": 3.339}] * 3
    assert replication.judge(identical)["all_reproduced"] is True
    assert sanity.chain_leaves_normal_range(MIXED) is not None
