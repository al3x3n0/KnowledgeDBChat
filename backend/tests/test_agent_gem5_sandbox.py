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
    assert "timing claim" in gem5.CPU_TYPES["O3CPU"]
    assert "no timing model" in gem5.CPU_TYPES["AtomicSimpleCPU"]
