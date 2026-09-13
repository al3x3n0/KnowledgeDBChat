"""Refuse a core model this image cannot run, instead of hanging on it.

CPU_TYPES names five ARM cores. In the image this project uses, two are not
compiled in and one deadlocks on every glibc binary. The deadlock is the
expensive kind: the run consumes its whole timeout and then reports a timeout,
which reads as "the workload was too big" and sends the caller to shrink a
workload that was never the problem. One such probe cost 300 seconds before
anyone suspected the model.
"""

import pytest

from app.services import agent_gem5_sandbox as gem5


def _probe(monkeypatch, op_classes=(), log=""):
    """Stand in for the container, returning a config dump and a gem5 log."""
    pool = "\n".join(f"opClass={c}" for c in op_classes)

    async def _fake(script, workdir, **kwargs):
        return 0, f"{pool}\n---\n{log}", ""

    monkeypatch.setattr(gem5.agent_sandbox_runtime, "run_in_sandbox", _fake)
    gem5.forget_model_support()


WORKING = ("IntAlu", "MemRead", "MemWrite", "FloatMisc", "FloatAdd")


class TestAModelThatCannotRun:
    @pytest.mark.asyncio
    async def test_a_model_missing_floatmisc_is_refused(self, monkeypatch):
        """The class ex5_big lacks. Established by elimination: NeoverseV2 has
        no FloatMemRead either and runs fine, so that was not it."""
        _probe(monkeypatch, op_classes=[c for c in WORKING if c != "FloatMisc"])

        result = await gem5.model_support("img", "ex5_big")

        assert result["usable"] is False
        assert "FloatMisc" in result["reason"]

    @pytest.mark.asyncio
    async def test_the_refusal_says_it_is_not_the_workload(self, monkeypatch):
        """The failure it replaces was a timeout, and a timeout sends you to
        shrink a workload that was never the problem."""
        _probe(monkeypatch, op_classes=[c for c in WORKING if c != "FloatMisc"])

        reason = (await gem5.model_support("img", "ex5_big"))["reason"]

        assert "not of the workload" in reason
        assert "smaller workload hangs too" in reason

    @pytest.mark.asyncio
    async def test_a_model_not_compiled_in_is_refused(self, monkeypatch):
        _probe(monkeypatch, op_classes=[], log="HPI is unavailable.")

        result = await gem5.model_support("img", "HPI")

        assert result["usable"] is False
        assert "not compiled into the gem5 build" in result["reason"]


class TestAModelThatCanRun:
    @pytest.mark.asyncio
    async def test_a_complete_pool_is_usable(self, monkeypatch):
        _probe(monkeypatch, op_classes=WORKING)
        assert (await gem5.model_support("img", "ex5_LITTLE"))["usable"] is True

    @pytest.mark.asyncio
    async def test_a_sparse_pool_with_floatmisc_is_still_usable(self, monkeypatch):
        """NeoverseV2 declares 35 classes against ex5_LITTLE's 82, including
        no FP memory ops at all, and runs everything asked of it."""
        _probe(monkeypatch, op_classes=("IntAlu", "MemRead", "FloatMisc"))
        assert (await gem5.model_support("img", "NeoverseV2"))["usable"] is True


class TestTheProbeItself:
    @pytest.mark.asyncio
    async def test_an_answer_is_cached_per_image_and_model(self, monkeypatch):
        calls = []

        async def _counting(script, workdir, **kwargs):
            calls.append(1)
            return 0, "opClass=FloatMisc\n---\n", ""

        monkeypatch.setattr(gem5.agent_sandbox_runtime, "run_in_sandbox", _counting)
        gem5.forget_model_support()

        await gem5.model_support("img", "m")
        await gem5.model_support("img", "m")
        assert len(calls) == 1, "a settled answer must not re-probe"

        await gem5.model_support("img", "other")
        assert len(calls) == 2, "a different model is a different question"

    @pytest.mark.asyncio
    async def test_a_probe_that_could_not_run_is_not_cached(self, monkeypatch):
        """A docker hiccup must not disable a model for the life of the
        process, and must not block a call either."""

        async def _broken(script, workdir, **kwargs):
            raise RuntimeError("docker daemon is not running")

        monkeypatch.setattr(gem5.agent_sandbox_runtime, "run_in_sandbox", _broken)
        gem5.forget_model_support()

        result = await gem5.model_support("img", "m")

        assert result["usable"] is True, "an unknown model is allowed through"
        assert result["probed"] is False
        assert not gem5._MODEL_SUPPORT, "nothing may be remembered from a failed probe"

    @pytest.mark.asyncio
    async def test_forgetting_one_image_leaves_the_others(self, monkeypatch):
        _probe(monkeypatch, op_classes=WORKING)
        await gem5.model_support("img-a", "m")
        await gem5.model_support("img-b", "m")

        gem5.forget_model_support("img-a")

        assert not any(k.startswith("img-a::") for k in gem5._MODEL_SUPPORT)
        assert any(k.startswith("img-b::") for k in gem5._MODEL_SUPPORT)
