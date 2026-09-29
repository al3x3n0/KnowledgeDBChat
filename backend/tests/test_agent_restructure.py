"""The verdicts of the restructuring tools, pinned without Docker.

Each case here is a way the judgement could lie: a lucky trial read as a win,
a diverging candidate timed anyway, a broken harness blamed on the candidate,
a replacement that never ran measured as "no faster", and a proposal credited
with another proposal's gain.
"""

import asyncio

from app.services import agent_binary_rewrite as binary
from app.services import agent_restructure as r
from app.services import agent_restructure_proposer as proposer


def _stdout(*blocks):
    return "\n".join(blocks) + "\n"


def _run(arm, i, rc, digest, text):
    return _stdout(
        f"__rc__ {arm} {i} {rc}",
        f"__hash__ {arm} {i} {digest}",
        f"__out_begin__ {arm} {i}",
        text,
        "",
        "__out_end__",
    )


class TestEquivalence:
    def _parsed(self, *runs):
        return r.parse_differential(
            _stdout("__prep__ 1", "__built__ orig 1", "__built__ cand 1")
            + "".join(runs)
        )

    def _compare(self, parsed, n, tolerance=r.DEFAULT_TOLERANCE):
        return r.compare_runs(
            parsed, n, baseline="orig", candidate="cand", tolerance=tolerance
        )

    def test_identical_hashes_are_bit_identical(self):
        parsed = self._parsed(
            _run("orig", 0, 0, "aa", "42"), _run("cand", 0, 0, "aa", "42")
        )
        out = self._compare(parsed, 1)
        assert out["status"] == "equivalent" and out["bit_identical"] is True

    def test_a_different_value_diverges_and_says_where(self):
        parsed = self._parsed(
            _run("orig", 0, 0, "aa", "1"),
            _run("cand", 0, 0, "aa", "1"),
            _run("orig", 1, 0, "bb", "479981292"),
            _run("cand", 1, 0, "cc", "98149945882905327"),
        )
        out = self._compare(parsed, 2)
        assert out["status"] == "diverged"
        assert out["first_problem"]["input"] == 1

    def test_float_drift_passes_only_under_the_declared_tolerance(self):
        runs = (
            _run("orig", 0, 0, "aa", "1.0000000"),
            _run("cand", 0, 0, "bb", "1.0000001"),
        )
        assert self._compare(self._parsed(*runs), 1)["status"] == "diverged"
        loose = self._compare(
            self._parsed(*runs), 1, tolerance=r.VALUE_CHANGING_TOLERANCE
        )
        assert loose["status"] == "equivalent" and loose["bit_identical"] is False

    def test_a_candidate_crash_is_not_a_divergence(self):
        parsed = self._parsed(
            _run("orig", 0, 0, "aa", "1"), _run("cand", 0, 139, "zz", "")
        )
        assert self._compare(parsed, 1)["status"] == "crashed"

    def test_an_original_that_fails_is_the_harness_not_the_candidate(self):
        parsed = self._parsed(
            _run("orig", 0, 2, "aa", ""), _run("cand", 0, 0, "bb", "1")
        )
        assert self._compare(parsed, 1)["status"] == "baseline_broken"

    def test_a_missing_run_is_not_silently_equivalent(self):
        parsed = self._parsed(_run("orig", 0, 0, "aa", "1"))
        assert self._compare(parsed, 1)["status"] == "crashed"


class TestSpeed:
    def _judge(self, orig, cand, ceiling=None):
        timings = {"orig": orig, "cand": cand}
        if ceiling:
            timings["ceiling"] = ceiling
        return r.judge_speed(
            timings,
            baseline="orig",
            candidate="cand",
            ceiling="ceiling" if ceiling else None,
        )

    def test_a_clear_win_is_faster(self):
        out = self._judge([200_000] * 7, [100_000] * 7, ceiling=[195_000] * 7)
        assert out["verdict"] == "faster" and out["speedup"] == 2.0

    def test_one_lucky_trial_is_not_a_win(self):
        # The null candidate measured live: fastest 1.089x, median 0.896x.
        out = self._judge(
            [264_000, 270_000, 273_000, 280_000, 290_000],
            [242_000, 300_000, 304_000, 310_000, 320_000],
        )
        assert out["verdict"] == "unresolved"

    def test_matching_the_compilers_ceiling_is_a_flag_not_an_idea(self):
        out = self._judge([200_000] * 7, [100_000] * 7, ceiling=[101_000] * 7)
        assert out["verdict"] == "compiler_already_can"

    def test_an_unresolved_ceiling_is_not_evidence_the_compiler_can(self):
        # Noisy trials: the candidate beats the original clearly, but its 1.3x
        # over the ceiling is inside the spread, and the ceiling shows no gain.
        # Shaped on the live sincos case: 1.40x over -O2, 1.31x over -O3, ~34%
        # trimmed spread, and -O3 itself only ~1.07x over -O2.
        noisy = [142_000, 142_000, 142_000, 190_000, 200_000]
        out = self._judge([200_000] * 5, noisy, ceiling=[187_000] * 5)
        assert out["verdict"] == "faster_than_original"

    def test_a_slower_candidate_is_slower(self):
        assert self._judge([100_000] * 7, [200_000] * 7)["verdict"] == "slower"

    def test_a_workload_under_the_noise_floor_is_warned_about(self):
        out = self._judge([20_000] * 7, [10_000] * 7)
        assert any("under" in w for w in out.get("warnings", []))


def test_timings_parse_from_the_interleaved_stream():
    timings, load, cpus = r.parse_timings(
        "__t__ orig 100\n__t__ cand 50\n__t__ orig 110\n__loadavg__ 1.5\n__cpus__ 8\n"
    )
    assert timings == {"orig": [100, 110], "cand": [50]} and load == 1.5 and cpus == 8


def test_new_mutable_state_is_counted_and_constants_are_not():
    source = "static const double T[4] = {0};\nstatic int ready = 0;\nstatic double cache[256];\nstatic inline int f(void) { return 1; }\n"
    assert r.mutable_statics(source) == 2


def test_no_inputs_is_refused_rather_than_vacuously_equivalent():
    problem, _ = r.check_inputs([], 0)
    assert problem and "vacuous" in problem


class TestBinarySplice:
    def test_the_export_check_fails_the_arm_without_ending_the_script(self):
        # `exit 1` inside the brace group ended the shared script: the log was
        # never printed and later arms never ran.
        build = binary._spliced_build("replacement", "cand", "sum_mod", "-O2")
        assert "exit" not in build and "false" in build
        assert "--weaken-symbol=sum_mod" in build
        assert "T sum_mod$" in build

    def test_symbols_that_are_not_identifiers_are_refused(self):
        out = asyncio.run(
            binary.evaluate_binary_rewrite(
                symbol="f; rm -rf /",
                replacement_asm="x",
                driver="int main(){}",
                inputs=["1"],
            )
        )
        assert "not a C identifier" in out["error"]

    def test_bytes_that_are_not_elf_are_refused(self):
        import base64

        assert "not an ELF" in binary._decode_object(
            base64.b64encode(b"MZ\x90\x00").decode()
        )


class TestProposer:
    def test_raw_newlines_inside_strings_still_parse(self):
        text = '```json\n{"proposals": [{"name": "t", "kernel": "int f(void) {\n\treturn 1;\n}"}]}\n```'
        assert proposer._lenient_json(text)["proposals"][0]["name"] == "t"

    def test_a_winner_that_carries_the_best_ones_idea_is_credited_with_nothing(self):
        judged = [
            {
                "name": "table",
                "verdict": "faster",
                "speedup": 7.2,
                "speedup_over_ceiling": 7.3,
                "notes": [],
            },
            {
                "name": "branchless",
                "verdict": "faster",
                "speedup": 6.3,
                "speedup_over_ceiling": 6.1,
                "notes": [],
            },
            {
                "name": "own-idea",
                "verdict": "faster",
                "speedup": 3.0,
                "speedup_over_ceiling": 3.0,
                "notes": [],
            },
        ]

        async def evaluate(proposal, baseline=None):
            assert baseline["name"] == "table"
            verdict = "faster" if proposal["name"] == "own-idea" else "unresolved"
            return {
                "data": {"verdict": verdict, "timing": {"speedup": 1.0}},
                "findings": [{"type": "restructuring_result"}],
            }

        findings = asyncio.run(proposer._attribute(judged, evaluate))
        assert len(findings) == 2
        by_name = {j["name"]: j for j in judged}
        assert by_name["branchless"]["over_best"]["verdict"] == "unresolved"
        assert any("added nothing" in n for n in by_name["branchless"]["notes"])
        # The one that adds something of its own outranks the one that does not.
        assert [j["name"] for j in judged] == ["table", "own-idea", "branchless"]

    def test_the_prompt_names_the_passes_that_already_exist(self):
        from app.services.agent_toolchains import LLVM_PASSES

        prompt = proposer._source_system_prompt()
        assert all(p.name in prompt for p in LLVM_PASSES)
        assert "AAPCS64" in proposer._binary_system_prompt()


def test_each_proposal_call_is_told_what_is_already_taken(monkeypatch):
    asked = []

    async def fake_call(system, message, schema, *, user_id, db):
        asked.append(message)
        n = len(asked)
        return {
            "proposals": [
                {"name": f"idea-{n}", "idea": f"does thing {n}", "kernel": "int f;"}
            ]
        }

    async def evaluate(proposal, baseline=None):
        return {"data": {"verdict": "unresolved"}, "findings": []}

    monkeypatch.setattr(proposer, "_call", fake_call)
    out = asyncio.run(
        proposer._propose_and_judge(
            system="s",
            message="m",
            schema={},
            body_key="kernel",
            count=3,
            evaluate=evaluate,
            user_id=None,
            db=None,
        )
    )
    assert len(out["proposals"]) == 3
    assert "Already proposed" not in asked[0]
    assert "idea-1" in asked[1] and "idea-1" in asked[2] and "idea-2" in asked[2]


def test_an_unparsable_reply_is_named_not_reported_as_no_ideas(monkeypatch):
    async def fake_call(system, message, schema, *, user_id, db):
        return {
            "_unparsed": "reply of 0 characters did not parse as JSON (stop_reason='length')"
        }

    monkeypatch.setattr(proposer, "_call", fake_call)
    out = asyncio.run(
        proposer._propose_and_judge(
            system="s",
            message="m",
            schema={},
            body_key="kernel",
            count=2,
            evaluate=None,
            user_id=None,
            db=None,
        )
    )
    assert "stop_reason='length'" in out["error"]


def test_a_bare_proposal_object_is_accepted():
    bare = {"name": "neon", "idea": "x", "asm": ".globl step\nstep: ret"}
    assert proposer._first_proposal(bare, "assembly")["assembly"].startswith(".globl")
    wrapped = {"proposals": [{"name": "t", "kernel": "int f;"}]}
    assert proposer._first_proposal(wrapped, "kernel")["name"] == "t"
    assert proposer._first_proposal({"note": "nothing"}, "kernel") is None


def test_an_unresolved_hint_of_a_gain_is_remeasured_once():
    calls = []

    async def evaluate(proposal, baseline=None, trials=7):
        calls.append(trials)
        return {"data": {"verdict": "faster", "timing": {"speedup": 1.5}}}

    first = {"data": {"verdict": "unresolved", "timing": {"speedup": 1.53}}}
    out = asyncio.run(proposer._remeasure_if_hinted(first, {"name": "p"}, evaluate))
    assert calls == [proposer.MAX_TRIALS] and out["data"]["verdict"] == "faster"
    assert out["data"]["first_measurement"] == {"speedup": 1.53}

    # No hint of a gain: nothing is spent re-timing it.
    calls.clear()
    flat = {"data": {"verdict": "unresolved", "timing": {"speedup": 0.99}}}
    assert asyncio.run(proposer._remeasure_if_hinted(flat, {}, evaluate)) is flat
    assert calls == []


class TestCpuBasis:
    def test_cpu_time_is_parsed_beside_wall_time(self):
        out = "__t__ orig 300000 250000\n__t__ cand 150000 120000\n__t__ orig 900000\n"
        assert r.parse_cpu_timings(out) == {"orig": [250000], "cand": [120000]}
        wall, _, _ = r.parse_timings(out)
        assert wall["orig"] == [300000, 900000]

    def test_a_single_threaded_pair_is_judged_on_cpu_time(self):
        # Wall time swamped by queueing; CPU time steady and clearly 1.4x.
        wall = {
            "orig": [300_000, 520_000, 410_000, 700_000],
            "cand": [210_000, 600_000, 380_000, 520_000],
        }
        cpu = {
            "orig": [280_000, 281_000, 282_000, 283_000],
            "cand": [200_000, 201_000, 201_000, 202_000],
        }
        out = r.judge_speed(
            wall, baseline="orig", candidate="cand", ceiling=None, cpu_timings=cpu
        )
        assert out["basis"] == "cpu" and out["verdict"] == "faster"
        assert out["wall"]["verdict"] == "unresolved"

    def test_a_parallel_candidate_falls_back_to_wall_time(self):
        wall = {"orig": [400_000] * 4, "cand": [100_000] * 4}
        cpu = {"orig": [390_000] * 4, "cand": [380_000] * 4}  # four threads
        out = r.judge_speed(
            wall, baseline="orig", candidate="cand", ceiling=None, cpu_timings=cpu
        )
        assert out["basis"] == "wall" and out["verdict"] == "faster"
        assert any("more than one core" in w for w in out["warnings"])


def test_a_failed_timed_run_is_not_a_time():
    out = "__t__ orig 250000 240000\n__tfail__ cand 139\n__t__ cand 3000 2000\n"
    assert r.parse_timing_failures(out) == {"cand": [139]}
    script = r.timing_script(
        [r.Arm("orig", "true")], 0, 3, 10**9, run_args="- 300000"
    )
    assert "__tfail__" in script and "./$a - 300000 <in_0.txt" in script


class TestPaired:
    DRIFT = [250, 380, 260, 520, 300, 270, 410, 255, 600, 280, 265, 330, 450, 262, 290]

    def test_a_consistent_small_gain_under_drift_is_resolved(self):
        # The host's load swings 2.4x across trials; within each trial the
        # candidate is 4% faster. Unpaired, the spread swallows it.
        base = [t * 1000 for t in self.DRIFT]
        cand = [round(t * 1000 / 1.04) for t in self.DRIFT]
        out = r.judge_speed(
            {"orig": base, "cand": cand},
            baseline="orig",
            candidate="cand",
            ceiling=None,
        )
        assert out["verdict"] == "faster"
        assert (
            out["paired"]["ci95"][0] > 1 and abs(out["paired"]["median"] - 1.04) < 0.001
        )

    def test_a_null_under_the_same_drift_stays_unresolved(self):
        import random

        rng = random.Random(7)
        base = [t * 1000 for t in self.DRIFT]
        cand = [round(t * 1000 * rng.uniform(0.97, 1.03)) for t in self.DRIFT]
        out = r.judge_speed(
            {"orig": base, "cand": cand},
            baseline="orig",
            candidate="cand",
            ceiling=None,
        )
        assert out["verdict"] == "unresolved"

    def test_gains_under_code_placement_are_not_claimed(self):
        base = [t * 1000 for t in self.DRIFT]
        cand = [round(t * 1000 / 1.02) for t in self.DRIFT]
        out = r.judge_speed(
            {"orig": base, "cand": cand},
            baseline="orig",
            candidate="cand",
            ceiling=None,
        )
        assert out["paired"]["ci95"][0] > 1 and out["verdict"] == "unresolved"

    def test_too_few_pairs_fall_back_to_the_unpaired_rule(self):
        assert r.paired_ratio([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) is None


def test_arm_order_rotates_across_trials():
    import subprocess

    arms = [r.Arm("orig", "true"), r.Arm("cand", "true"), r.Arm("ceiling", "true")]
    script = r.timing_script(arms, 0, 3, 10**12)
    # Run just the ordering logic in a real POSIX shell.
    probe = script.split("for t in", 1)[1].split("  for a in $order;", 1)[0]
    shell = "for t in" + probe + ' echo "$order"; done'
    out = subprocess.run(
        ["sh", "-c", shell], capture_output=True, text=True
    ).stdout.split("\n")
    assert out[:3] == ["cand ceiling orig", "ceiling orig cand", "orig cand ceiling"]


class TestControl:
    DRIFT = TestPaired.DRIFT

    def _series(self, factor, jitter=None):
        out = []
        for i, t in enumerate(self.DRIFT):
            j = jitter[i] if jitter else 1.0
            out.append(round(t * 1000 / factor * j))
        return out

    def test_a_gain_inside_the_controls_interval_is_not_claimed(self):
        # The identical copy wanders +-8% against the baseline; a 4% candidate
        # cannot be told from that.
        wobble = [
            1.08,
            0.93,
            1.05,
            0.95,
            1.07,
            0.92,
            1.06,
            0.94,
            1.08,
            0.93,
            1.05,
            0.96,
            1.07,
            0.92,
            1.06,
        ]
        timings = {
            "orig": self._series(1.0),
            "cand": self._series(1.04),
            "control": self._series(1.0, wobble),
        }
        out = r.judge_speed(
            timings, baseline="orig", candidate="cand", ceiling=None, control="control"
        )
        assert out["paired"]["ci95"][0] > 1
        assert out["verdict"] == "unresolved"

    def test_a_gain_clear_of_a_quiet_control_is_claimed(self):
        steady = [1.0 + (0.004 if i % 2 else -0.004) for i in range(15)]
        timings = {
            "orig": self._series(1.0),
            "cand": self._series(1.10),
            "control": self._series(1.0, steady),
        }
        out = r.judge_speed(
            timings, baseline="orig", candidate="cand", ceiling=None, control="control"
        )
        assert out["verdict"] == "faster" and "control" in out

    def test_a_control_that_reads_as_different_is_called_out(self):
        timings = {
            "orig": self._series(1.0),
            "cand": self._series(1.2),
            "control": self._series(1.05),
        }
        out = r.judge_speed(
            timings, baseline="orig", candidate="cand", ceiling=None, control="control"
        )
        assert any("manufacturing differences" in w for w in out.get("warnings", []))


class TestAnyBoundContract:
    """A contract that asks "did some proposal win" must not be satisfied by
    one that diverged and so carries no numbers at all."""

    SPEC = {"bounds": {"restructuring_result": {"field": "win", "min": 1, "any": True}}}

    def _check(self, findings):
        from app.services import agent_measurement_validity as v

        return v.evaluate({"validity": self.SPEC}, {"findings": findings})

    def test_one_winner_among_losers_satisfies_it(self):
        out = self._check(
            [
                {"type": "restructuring_result", "win": 0},
                {"type": "restructuring_result", "win": 1},
                {"type": "restructuring_result", "win": 0},
            ]
        )
        assert out["missing"] == []

    def test_findings_without_the_field_do_not_satisfy_it(self):
        out = self._check([{"type": "restructuring_result", "verdict": "diverged"}])
        assert out["missing"] == ["validity:bounds:restructuring_result"]

    def test_the_finding_carries_win(self):
        packaged = r.package(
            {"verdict": "faster", "timing": {"speedup": 2.0}},
            kind="restructuring_result",
            label="x",
            invariant="",
            value_preserving=True,
            n_inputs=3,
        )
        assert packaged["findings"][0]["win"] == 1
