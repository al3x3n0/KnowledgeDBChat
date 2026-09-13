"""Guards on the simulation referee.

gem5 is not run here; what is checked is that the tool refuses inputs that
would produce a misleading measurement, and that it does not quietly accept a
configuration whose result would mean something other than asked.
"""

import pytest

from app.services import agent_gem5_sandbox as gem5


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(gem5.agent_sandbox_runtime, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        gem5.agent_sandbox_runtime, "allowed_images", lambda: [gem5.DEFAULT_IMAGE]
    )


@pytest.mark.asyncio
async def test_execution_must_be_enabled(monkeypatch):
    monkeypatch.setattr(gem5.agent_sandbox_runtime, "execution_enabled", lambda: False)

    result = await gem5.simulate_c_workload(code="int main(void){return 0;}")

    assert "ENABLE_UNSAFE_CODE_EXECUTION" in result["error"]


@pytest.mark.asyncio
async def test_an_unlisted_image_is_refused(enabled):
    result = await gem5.simulate_c_workload(
        code="int main(void){return 0;}", image="evil:latest"
    )

    assert "not allowlisted" in result["error"]


@pytest.mark.asyncio
async def test_an_unknown_cpu_model_lists_the_real_choice(enabled):
    """Which core was modelled decides what the number means."""
    result = await gem5.simulate_c_workload(
        code="int main(void){return 0;}", cpu_type="neoverse-n1"
    )

    assert "Unknown cpu_type" in result["error"]
    assert "O3CPU" in result["error"]
    assert "out-of-order" in result["error"]


@pytest.mark.asyncio
async def test_run_arguments_may_not_smuggle_shell_syntax(enabled):
    result = await gem5.simulate_c_workload(
        code="int main(void){return 0;}", run_args="4; rm -rf /"
    )

    assert "run_args contain unsupported characters" in result["error"]


@pytest.mark.asyncio
async def test_flags_may_not_smuggle_shell_syntax(enabled):
    result = await gem5.simulate_c_workload(
        code="int main(void){return 0;}", flags="-O3 `curl evil`"
    )

    assert "flags contain unsupported characters" in result["error"]


def test_every_offered_cpu_model_explains_what_it_models():
    """A caller picking a model must be able to tell what it will measure."""
    for name, why in gem5.CPU_TYPES.items():
        assert why.strip(), name
    assert "timing-capable" in gem5.CPU_TYPES["O3CPU"]
    assert "no timing model" in gem5.CPU_TYPES["AtomicSimpleCPU"]
    # The generic models must not read as stand-ins for silicon. Measured
    # against this host, O3CPU is 40% off per instruction and NeoverseV2 77%,
    # so a caller choosing between them is choosing which core to model, not
    # merely how much detail to pay for.
    assert "no real core" in gem5.CPU_TYPES["O3CPU"]
    assert "real core" in gem5.CPU_TYPES["NeoverseV2"]


def test_models_missing_scalar_fma_are_named():
    """The deadlock is silent, so the set that hits it must be explicit.

    NeoverseV2, ex5_big and ex5_LITTLE declare SimdFloatMultAcc but not
    FloatMultAcc, so a scalar fmadd can never issue and the simulation hangs
    instead of failing. Every model named here is one the guard must cover.
    """
    assert gem5.MODELS_WITHOUT_SCALAR_FMA <= set(gem5.CPU_TYPES)
    assert "NeoverseV2" in gem5.MODELS_WITHOUT_SCALAR_FMA
    assert "O3CPU" not in gem5.MODELS_WITHOUT_SCALAR_FMA


@pytest.mark.asyncio
async def test_parameter_overrides_must_be_full_paths(enabled):
    """A path without `system` is rejected rather than silently prefixed.

    Prepending `system.cpu[0].` for the caller doubled the prefix whenever a
    full path was passed, and gem5 reported that as `KeyError: system` from
    deep inside its config machinery.
    """
    result = await gem5.simulate_c_workload(
        code="int main(void){return 0;}",
        param_overrides=["instQueues[0].fuPool.FUList[3].opList[4].opLat=10"],
    )

    assert "system.<path>=<value>" in result["error"]


@pytest.mark.asyncio
async def test_parameter_overrides_reject_shell_syntax(enabled):
    result = await gem5.simulate_c_workload(
        code="int main(void){return 0;}",
        param_overrides=["system.cpu[0].numROBEntries=64; curl evil"],
    )

    assert "not of the form" in result["error"]


class TestGem5sOwnCpuNames:
    """A live run asked for DerivO3CPU -- the class O3CPU resolves to, offered
    by gem5's own ObjectList -- and was refused as unknown. Being stricter
    than the simulator costs an iteration and teaches nothing."""

    def test_the_class_o3cpu_resolves_to_is_accepted(self):
        from app.services.agent_gem5_sandbox import resolve_cpu_type

        assert resolve_cpu_type("DerivO3CPU") == "O3CPU"
        assert resolve_cpu_type("ArmMinorCPU") == "MinorCPU"

    def test_a_name_that_means_nothing_is_left_alone_to_be_refused(self):
        from app.services.agent_gem5_sandbox import CPU_TYPES, resolve_cpu_type

        assert resolve_cpu_type("MagicCPU") == "MagicCPU"
        assert "MagicCPU" not in CPU_TYPES

    def test_the_documented_names_still_mean_themselves(self):
        from app.services.agent_gem5_sandbox import CPU_TYPES, resolve_cpu_type

        for name in CPU_TYPES:
            assert resolve_cpu_type(name) == name


class TestTheAliasReachesEveryEntryPoint:
    """The alias was added and applied to the first call site the pattern
    matched -- describe_model_parameters -- while the tool a live run actually
    called, simulate_c_workload, kept refusing DerivO3CPU. The unit test passed
    because it tested the helper rather than the path."""

    def test_no_entry_point_bypasses_it(self):
        import ast
        import inspect

        from app.services import agent_gem5_sandbox as gem5

        source = inspect.getsource(gem5)
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.get_source_segment(source, node) or ""
            if "not in CPU_TYPES" in body and "resolve_cpu_type(" not in body:
                offenders.append(node.name)

        assert offenders == [], f"these check CPU_TYPES without the alias: {offenders}"
