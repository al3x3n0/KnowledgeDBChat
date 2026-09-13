"""Guards and summaries for the dynamic profiling tool.

The container is not run here; the docker-backed path is exercised through its
preflight and through the summarizers that turn a parsed profile into what an
agent reads.
"""

import pytest

from app.services import agent_profile_sandbox as prof
from app.services import callgrind_profile as cg


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(prof.agent_sandbox_runtime, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        prof.agent_sandbox_runtime, "allowed_images", lambda: [prof.DEFAULT_IMAGE]
    )


@pytest.mark.asyncio
async def test_execution_must_be_enabled(monkeypatch):
    monkeypatch.setattr(prof.agent_sandbox_runtime, "execution_enabled", lambda: False)

    result = await prof.profile_c_workload(code="int main(void){return 0;}")

    assert "ENABLE_UNSAFE_CODE_EXECUTION" in result["error"]


@pytest.mark.asyncio
async def test_an_unlisted_image_is_refused(enabled):
    result = await prof.profile_c_workload(
        code="int main(void){return 0;}", image="evil:latest"
    )

    assert "not allowlisted" in result["error"]


@pytest.mark.asyncio
async def test_run_arguments_may_not_smuggle_shell_syntax(enabled):
    result = await prof.profile_c_workload(
        code="int main(void){return 0;}", run_args="8; rm -rf /"
    )

    assert "run_args contain unsupported characters" in result["error"]


@pytest.mark.asyncio
async def test_flags_may_not_smuggle_shell_syntax(enabled):
    result = await prof.profile_c_workload(
        code="int main(void){return 0;}", flags="-O3 && curl evil"
    )

    assert "flags contain unsupported characters" in result["error"]


def test_functions_are_ranked_with_their_share_of_the_run():
    profile = cg.Profile(total=1000, by_function={"hot": 880, "cold": 120})

    ranked = prof.summarize_functions(profile, limit=5)

    assert ranked[0] == {"function": "hot", "instructions": 880, "share": 0.88}
    assert ranked[1]["function"] == "cold"


def test_a_hot_block_carries_the_instructions_a_candidate_is_built_from():
    profile = cg.Profile(
        total=1000,
        by_address={0x1000: 400, 0x1004: 400, 0x2000: 200},
        by_function={},
    )
    listing = {0x1000: "fsqrt s0, s1", 0x1004: "fdiv s0, s2, s0", 0x2000: "ret"}

    blocks = prof.summarize_blocks(profile, listing, limit=2)

    assert blocks[0]["start"] == "0x1000"
    assert blocks[0]["executions"] == 400
    assert blocks[0]["disassembly"] == ["fsqrt s0, s1", "fdiv s0, s2, s0"]
    assert blocks[0]["share"] == 0.8
    assert blocks[0]["disassembly_truncated"] is False


def test_a_long_block_says_its_disassembly_was_cut():
    """A clipped listing that does not say so reads as the whole block."""
    addresses = {0x1000 + 4 * i: 10 for i in range(prof.MAX_BLOCK_INSTRUCTIONS + 5)}
    profile = cg.Profile(total=1000, by_address=addresses, by_function={})

    blocks = prof.summarize_blocks(profile, {}, limit=1)

    assert len(blocks[0]["disassembly"]) == prof.MAX_BLOCK_INSTRUCTIONS
    assert blocks[0]["disassembly_truncated"] is True


def test_an_empty_profile_summarizes_to_nothing_rather_than_raising():
    empty = cg.Profile()

    assert prof.summarize_functions(empty, limit=5) == []
    assert prof.summarize_blocks(empty, {}, limit=5) == []
