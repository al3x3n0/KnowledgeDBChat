"""The verdicts of turning a rewrite into a pass, pinned without Docker.

Each is a way the answer could lie: a plugin that loads and does nothing read
as "no gain", a pass that is fast on the kernel and unsound one step away read
as "faster", a shadow of the idea read as the idea, and a repair that passes
by moving the test rather than fixing the pass.
"""

import asyncio

from app.services import agent_pass_from_rewrite as pr

PLAIN = "define double @f(i8 %x) {\n  %c = uitofp i8 %x to double\n  %r = call double @cos(double %c)\n  ret double %r\n}\n"
PASSED = "define double @f(i8 %x) {\n  %i = zext i8 %x to i64\n  %p = getelementptr [256 x double], ptr @t, i64 0, i64 %i\n  %r = load double, ptr %p\n  ret double %r\n}\n"


def _probe(plain, passed):
    return (
        f"__plain__\n{plain}__passed__\n{passed}__end__\n__stderr__\n__stderr_end__\n"
    )


def test_identical_ir_is_did_not_fire_material():
    assert pr.parse_firing(_probe(PLAIN, PLAIN))["fired"] is False


def test_changed_ir_fires_and_names_the_change():
    out = pr.parse_firing(_probe(PLAIN, PASSED))
    assert out["fired"] is True
    assert out["ir_deltas"]["call @cos"] == -1 and out["ir_deltas"]["load"] == 1


def test_a_filename_difference_alone_is_not_firing():
    a = "; ModuleID = 'a.c'\nsource_filename = \"a.c\"\n" + PLAIN
    b = "; ModuleID = 'b.c'\nsource_filename = \"b.c\"\n" + PLAIN
    assert pr.parse_firing(_probe(a, b))["fired"] is False


def test_build_failure_and_crash_are_told_apart():
    assert pr.parse_firing("__build_failed__\npass.cpp:3: error: x")["stage"] == "build"
    assert pr.parse_firing("__pass_crashed__\nStack dump:")["stage"] == "run"


def test_recovered_share_compares_arms_from_one_run():
    timing = {"arms_fastest_ms": {"orig": 640.0, "cand": 88.0, "rewrite": 82.0}}
    assert 0.9 < pr.recovered_share(timing) < 0.95
    # A rewrite that gained nothing leaves nothing to recover.
    assert (
        pr.recovered_share(
            {"arms_fastest_ms": {"orig": 100, "cand": 90, "rewrite": 100}}
        )
        is None
    )
    assert pr.recovered_share({"arms_fastest_ms": {"orig": 100, "cand": 90}}) is None


def test_overreaching_outranks_the_speed_verdict():
    packaged = {
        "data": {"verdict": "faster", "notes": []},
        "findings": [{"subject": "p", "verdict": "faster", "title": "p: faster"}],
    }
    pr._annotate(packaged, 0.9, {"fired": True, "ir_deltas": {"load": 1}})
    pr._override(packaged, "overreaches")
    assert packaged["data"]["verdict"] == "overreaches"
    assert packaged["data"]["speed_verdict"] == "faster"
    assert packaged["findings"][0]["verdict"] == "overreaches"
    assert packaged["findings"][0]["fired_on_must_decline"] is True


def test_minus_o0_is_refused_because_no_pipeline_runs():
    out = asyncio.run(
        pr.evaluate_pass_on_kernel(
            pass_source="x",
            pass_name="p",
            kernel="int f;",
            driver="int main(){}",
            inputs=["1"],
            flags="-O0",
        )
    )
    assert "-O0" in out["error"]


def test_the_decline_case_is_frozen_across_repairs(monkeypatch):
    """Otherwise the cheapest repair for `overreaches` is a new test."""
    replies = iter(
        [
            {
                "expressible": True,
                "reason": "",
                "pass_name": "p",
                "pass_source": "v1",
                "must_decline": "ORIGINAL",
            },
            {
                "expressible": True,
                "reason": "",
                "pass_name": "p",
                "pass_source": "v2",
                "must_decline": "EASIER",
            },
        ]
    )
    seen = []

    async def fake_call(system, message, schema, *, user_id, db):
        return next(replies)

    async def fake_eval(**kwargs):
        seen.append(kwargs["must_decline"])
        verdict = "overreaches" if len(seen) == 1 else "faster"
        return {"success": True, "data": {"verdict": verdict}, "findings": []}

    async def no_ir(kernel, flags):
        return ""

    import app.services.agent_restructure_proposer as proposer

    monkeypatch.setattr(proposer, "_call", fake_call)
    monkeypatch.setattr(pr, "evaluate_pass_on_kernel", fake_eval)
    monkeypatch.setattr(pr, "_kernel_ir", no_ir)
    monkeypatch.setattr(pr, "sandbox_blocked", lambda image: None)
    out = asyncio.run(
        pr.synthesize_pass_from_rewrite(
            kernel="k", rewrite_kernel="r", driver="int main(){}", inputs=["1"]
        )
    )
    assert seen == ["ORIGINAL", "ORIGINAL"]
    assert [a["verdict"] for a in out["data"]["attempts"]] == ["overreaches", "faster"]


def test_not_expressible_is_a_result_not_an_error(monkeypatch):
    async def fake_call(system, message, schema, *, user_id, db):
        return {"expressible": False, "reason": "relies on the driver never reading vx"}

    async def no_ir(kernel, flags):
        return ""

    import app.services.agent_restructure_proposer as proposer

    monkeypatch.setattr(proposer, "_call", fake_call)
    monkeypatch.setattr(pr, "_kernel_ir", no_ir)
    monkeypatch.setattr(pr, "sandbox_blocked", lambda image: None)
    out = asyncio.run(
        pr.synthesize_pass_from_rewrite(
            kernel="k", rewrite_kernel="r", driver="int main(){}", inputs=["1"]
        )
    )
    assert out["success"] is True and out["data"]["verdict"] == "not_expressible"
    assert out["findings"][0]["type"] == "pass_evaluation"


def test_malformed_ir_is_named_rather_than_left_to_diverge():
    out = pr.parse_firing(
        "__verify_failed__\nBoth operands to a binary operator are not of the same type!\n"
        "  %24 = lshr i128 %23, i64 64\n"
    )
    assert out["stage"] == "invalid_ir" and "lshr i128" in out["errors"]


def test_a_name_opt_does_not_know_is_unregistered():
    out = pr.parse_firing("__verify_failed__\nopt: unknown pass name 'fastmod'\n")
    assert out["stage"] == "unregistered"


def test_the_verifier_runs_right_after_the_pass():
    script = pr._firing_script("-O2", ["kernel.c"], "uchar-trig-table")
    assert "-passes='uchar-trig-table,verify'" in script


def test_an_unresolved_pass_is_measured_once_more_with_more_trials(monkeypatch):
    calls = []

    async def fake_call(system, message, schema, *, user_id, db):
        return {
            "expressible": True,
            "reason": "",
            "pass_name": "p",
            "pass_source": "s",
            "must_decline": "d",
        }

    async def fake_eval(**kwargs):
        calls.append(kwargs.get("trials"))
        verdict = "unresolved" if len(calls) == 1 else "faster"
        return {
            "success": True,
            "data": {"verdict": verdict, "timing": {"n": len(calls)}},
            "findings": [],
        }

    async def no_ir(kernel, flags):
        return ""

    import app.services.agent_restructure_proposer as proposer

    monkeypatch.setattr(proposer, "_call", fake_call)
    monkeypatch.setattr(pr, "evaluate_pass_on_kernel", fake_eval)
    monkeypatch.setattr(pr, "_kernel_ir", no_ir)
    monkeypatch.setattr(pr, "sandbox_blocked", lambda image: None)
    out = asyncio.run(
        pr.synthesize_pass_from_rewrite(
            kernel="k", rewrite_kernel="r", driver="int main(){}", inputs=["1"]
        )
    )
    assert calls == [None, pr.MAX_TRIALS]
    assert out["data"]["verdict"] == "faster"
    assert out["data"]["first_measurement"] == {"n": 1}
