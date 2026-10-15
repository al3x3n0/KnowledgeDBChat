"""The evolutionary search over rewrites: what it keeps, breeds and believes.

The loop is run for real with the model and the measurements replaced: each
proposed kernel names the verdict and median the stand-in measurement will
report for it, so every test says exactly what was measured.
"""

import pytest

from app.services import agent_restructure_search as search_module

pytestmark = pytest.mark.unit

ORIGINAL = "int k(int x) { return x % 7; }"


def kernel(name, verdict, median=None, holdout=None):
    """A candidate whose measurement is written into it."""
    return f"/* {name} {verdict} {median} {holdout or verdict} */"


def measured(source):
    _, name, verdict, median, holdout, _ = source.split(" ", 5)
    return name, verdict, (None if median == "None" else float(median)), holdout


class Model:
    """Proposes the queued kernels in order, recording what it was asked."""

    def __init__(self, *kernels):
        self.queue = list(kernels)
        self.asked = []

    async def __call__(self, message):
        self.asked.append(message)
        if not self.queue:
            return None
        source = self.queue.pop(0)
        name = measured(source)[0]
        return {"name": name, "idea": f"idea {name}", "kernel": source}


async def evaluate(proposal):
    _, verdict, median, _ = measured(proposal["kernel"])
    return {
        "data": {"verdict": verdict, "timing": {"median_speedup": median}},
        "findings": [{"type": "restructuring_result", "phase": "search"}],
    }


async def holdout(candidate):
    name, _, median, verdict = measured(candidate["kernel"])
    return {
        "data": {"verdict": verdict, "timing": {"median_speedup": median}},
        "findings": [{"type": "restructuring_result", "subject": name}],
    }


async def run(model, **kw):
    kw.setdefault("generations", 3)
    kw.setdefault("children", 1)
    return await search_module.search(
        kernel=ORIGINAL,
        base_message="BASE",
        propose=model,
        evaluate=evaluate,
        holdout=holdout,
        **kw,
    )


class TestBreeding:
    async def test_a_winner_becomes_the_parent_of_what_follows(self):
        model = Model(
            kernel("table", "faster", 2.0),
            kernel("table2", "faster", 3.0),
            kernel("table3", "faster", 3.5),
        )

        out = await run(model)

        # Seeded, so the sampling is fixed: the third child was grown from
        # the second, and the model was shown that parent's source.
        lineage = [(e["name"], e["parent"]) for e in out["archive"]]
        assert lineage == [
            ("original", None),
            ("table", "original"),
            ("table2", "original"),
            ("table3", "table2"),
        ]
        assert kernel("table2", "faster", 3.0) in model.asked[2]
        assert out["winner"]["name"] == "table3"

    async def test_a_failed_child_scores_zero_and_the_search_goes_on(self):
        model = Model(
            kernel("broken", "did_not_compile"),
            kernel("fine", "faster", 1.8),
        )

        out = await run(model, generations=2)

        broken = next(e for e in out["archive"] if e["name"] == "broken")
        assert broken["fitness"] == 0.0
        assert out["winner"]["name"] == "fine"

    async def test_an_unresolved_gain_is_not_fitness(self):
        model = Model(kernel("lucky", "unresolved", 4.0))

        out = await run(model, generations=1)

        (lucky,) = [e for e in out["archive"] if e["name"] == "lucky"]
        assert lucky["fitness"] == 1.0
        assert out["winner"] is None

    async def test_the_ask_shows_what_was_tried_and_what_it_measured(self):
        model = Model(kernel("a", "diverged"), kernel("b", "faster", 1.5))

        await run(model, generations=2)

        assert "a (from original): diverged" in model.asked[1]


class TestTheHoldout:
    async def test_a_winner_that_diverges_on_held_out_inputs_is_not_a_result(self):
        model = Model(
            kernel("overfit", "faster", 5.0, holdout="diverged"),
            kernel("honest", "faster", 2.0),
        )

        out = await run(model, generations=2)

        assert [h["name"] for h in out["holdout"]] == ["overfit", "honest"]
        assert out["holdout"][0]["verdict"] == "diverged"
        assert out["winner"]["name"] == "honest"

    async def test_only_held_out_measurements_are_evidence(self):
        model = Model(kernel("w", "faster", 2.0))

        out = await run(model, generations=1)

        assert out["findings"] == [{"type": "restructuring_result", "subject": "w"}]

    async def test_nothing_holds_and_it_says_so(self):
        model = Model(kernel("x", "faster", 3.0, holdout="crashed"))

        result = search_module.as_result(await run(model, generations=1), "k")

        assert result["data"]["winner"] is None
        assert "fit to those inputs" in result["data"]["summary"]


class TestWhenItStops:
    async def test_two_rounds_without_a_better_candidate_end_it(self):
        model = Model(
            kernel("good", "faster", 2.0),
            kernel("worse", "slower", 0.9),
            kernel("worse2", "slower", 0.8),
            kernel("never", "faster", 9.0),
        )

        out = await run(model, generations=5)

        assert "never" not in [e["name"] for e in out["archive"]]
        assert out["candidates_judged"] == 3

    async def test_a_broken_original_ends_it_with_the_reason(self):
        model = Model(kernel("x", "baseline_broken"))

        out = await run(model)

        assert "ORIGINAL kernel does not build" in out["error"]

    async def test_a_second_call_continues_from_the_archive(self):
        first = await run(Model(kernel("g1", "faster", 2.0)), generations=1)
        kept = search_module.kept(first["archive"])

        model = Model(kernel("g2", "faster", 2.5))
        second = await run(model, generations=1, archive=kept)

        names = [e["name"] for e in second["archive"]]
        assert names[:2] == ["original", "g1"] and names[-1] == "g2"
        assert second["archive"][-1]["generation"] == 2

    def test_what_is_kept_is_the_original_and_the_fittest(self):
        archive = [{"name": "original", "verdict": "original", "fitness": 1.0}] + [
            {"name": f"c{i}", "verdict": "faster", "fitness": float(i)}
            for i in range(30)
        ]

        kept = search_module.kept(archive)

        assert len(kept) == 16 and kept[0]["name"] == "original"
        assert {e["name"] for e in kept[1:]} == {f"c{i}" for i in range(15, 30)}


class TestTheTool:
    async def test_no_holdout_is_refused(self):
        result = await search_module.search_restructurings(
            kernel=ORIGINAL, driver="int main(){}", inputs=["1"], holdout_inputs=[]
        )

        assert "holdout_inputs are required" in result["error"]

    async def test_a_holdout_of_search_inputs_is_refused(self):
        result = await search_module.search_restructurings(
            kernel=ORIGINAL,
            driver="int main(){}",
            inputs=["1", "2"],
            holdout_inputs=["2"],
        )

        assert "also a search input" in result["error"]

    async def test_the_archive_lives_in_the_run_s_state(self, monkeypatch):
        from types import SimpleNamespace

        from app.services.agent_tool_providers import workspace_mutation

        seen = []

        async def fake(**kwargs):
            seen.append(kwargs["archive"])
            return {"success": True, "data": {}, "_archive": [{"name": "original"}]}

        monkeypatch.setattr(search_module, "search_restructurings", fake)
        handler = workspace_mutation.build_autonomous_workspace_mutation_provider(
            object()
        )._handlers["search_restructurings"]
        ctx = SimpleNamespace(state={}, user_id=None, job=None, db=None)
        params = {
            "kernel": ORIGINAL,
            "driver": "d",
            "inputs": ["1"],
            "holdout_inputs": ["2"],
        }

        first = await handler(params, ctx)
        await handler(params, ctx)

        assert "_archive" not in first
        assert seen == [None, [{"name": "original"}]]


async def test_the_holdout_times_the_search_s_bench_input(monkeypatch):
    """Found live: timed on the first held-out input, it timed a 5-element array."""
    calls = []

    async def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {"data": {"verdict": "faster", "timing": {"median_speedup": 2.0}}}

    async def fake_call(*args, **kwargs):
        return {"proposals": [{"name": "c", "idea": "i", "kernel": "k2"}]}

    async def no_listing(*args):
        return ""

    monkeypatch.setattr(search_module, "evaluate_restructuring", fake_evaluate)
    monkeypatch.setattr(search_module, "_call", fake_call)
    monkeypatch.setattr(search_module, "compiler_listing", no_listing)
    monkeypatch.setattr(search_module, "sandbox_blocked", lambda image: None)
    monkeypatch.setattr(
        search_module, "_first_proposal", lambda payload, key: payload["proposals"][0]
    )

    await search_module.search_restructurings(
        kernel=ORIGINAL,
        driver="d",
        inputs=["1000000 7 1", "10 3 2"],
        holdout_inputs=["5 1 1", "999 4294967291 5"],
        generations=1,
        children=1,
    )

    held = calls[-1]
    assert held["inputs"] == ["5 1 1", "999 4294967291 5", "1000000 7 1"]
    assert held["inputs"][held["bench_input"]] == "1000000 7 1"


class TestStateBetweenCalls:
    MEMO = """
#include <stdint.h>
uint64_t sum_mod(const uint32_t *a, int n, uint32_t d) {
    static const uint32_t *last_a; static uint64_t last_total;
    if (a == last_a) return last_total;
    uint64_t t = 0; for (int i = 0; i < n; i++) t += a[i] % d;
    last_a = a; last_total = t; return t;
}
"""

    def test_the_live_search_s_memoising_candidate_is_flagged(self):
        """Its shape, from the live run: a result cached in a static."""
        assert search_module.keeps_state(self.MEMO, ORIGINAL)

    def test_a_const_table_and_a_static_helper_are_not(self):
        table = (
            "static const uint8_t lut[4] = {0, 1, 2, 3};\n"
            "static inline uint32_t f(uint32_t x) { return lut[x & 3]; }\n"
            "static uint32_t g(uint32_t x) { return f(x); }\n"
        )

        assert not search_module.keeps_state(table, ORIGINAL)

    async def test_the_flag_reaches_the_summary_of_a_winner(self):
        async def evaluate_memo(proposal):
            return {"data": {"verdict": "faster", "timing": {"median_speedup": 4.0}}}

        async def propose(message):
            return {"name": "memo", "idea": "cache", "kernel": self.MEMO}

        out = await search_module.search(
            kernel=ORIGINAL,
            base_message="BASE",
            propose=propose,
            evaluate=evaluate_memo,
            holdout=evaluate_memo,
            generations=1,
            children=1,
        )
        result = search_module.as_result(out, "sum_mod")

        assert "keeps state between calls" in result["data"]["summary"]


class TestAnUnreachableModel:
    """Found live: the provider was down, and the search reported 0 judged."""

    async def test_it_stops_at_the_first_failure_and_says_why(self):
        calls = []

        async def down(message):
            calls.append(message)
            raise ConnectionError("Connection error.")

        out = await search_module.search(
            kernel=ORIGINAL,
            base_message="BASE",
            propose=down,
            evaluate=evaluate,
            holdout=holdout,
            generations=3,
            children=2,
        )
        result = search_module.as_result(out, "k")

        assert len(calls) == 1
        assert result["success"] is False
        assert "could not be reached (Connection error.)" in result["error"]

    async def test_a_reply_with_no_kernel_says_what_came_back(self):
        async def empty(message):
            return {"_failure": "reply of 0 characters did not parse as JSON"}

        out = await search_module.search(
            kernel=ORIGINAL,
            base_message="BASE",
            propose=empty,
            evaluate=evaluate,
            holdout=holdout,
            generations=1,
            children=1,
        )

        assert "did not parse as JSON" in search_module.as_result(out, "k")["error"]
