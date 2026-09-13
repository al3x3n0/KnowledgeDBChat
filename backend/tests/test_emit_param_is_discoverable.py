"""The accepted values of `emit` lived only in the validator.

Measured live: an implement stage called compile_c_snippet three times with
emit='counts' and lost three of its six iterations to it. The guess was a
reasonable one -- the codegen counts are exactly what that tool returns, they
come back in every reply, and `emit` only chooses which listing accompanies
them. Nothing the caller could read said so: the schema described two values in
prose without enumerating them, and the rejection listed the legal values
without naming the one it had rejected.
"""

import pytest

pytestmark = pytest.mark.unit


class TestTheSchemaStatesTheAcceptedSet:
    def test_emit_is_enumerated(self):
        from app.agent_core.tool_specs.measurement import SPECS

        spec = next(s for s in SPECS if s.name == "compile_c_snippet")
        emit = spec.parameters["properties"]["emit"]

        assert emit.get("enum") == ["asm", "ir"], (
            "a set that exists only in the validator cannot be read by the "
            "caller that has to satisfy it"
        )

    def test_every_enumerated_value_is_actually_accepted(self):
        """The two halves must not drift: a schema offering a value the
        validator rejects is worse than no schema."""
        from app.agent_core.tool_specs.measurement import SPECS
        from app.services.agent_compiler_sandbox import EMIT_ALIASES

        spec = next(s for s in SPECS if s.name == "compile_c_snippet")
        for value in spec.parameters["properties"]["emit"]["enum"]:
            assert value in EMIT_ALIASES, value

    def test_the_description_says_counts_come_back_anyway(self):
        """The misunderstanding that caused it: `emit` looked like it selected
        what the tool computes, when it selects only the listing."""
        from app.agent_core.tool_specs.measurement import SPECS

        spec = next(s for s in SPECS if s.name == "compile_c_snippet")
        description = spec.parameters["properties"]["emit"]["description"]

        assert "COUNTS" in description or "counts" in description
        assert "regardless" in description or "every call" in description


@pytest.mark.asyncio
class TestTheRejectionNamesTheValue:
    async def test_it_quotes_what_was_passed(self, monkeypatch):
        from app.services import agent_compiler_sandbox as sandbox

        # Past the preflight guard, which the suite otherwise leaves shut --
        # every other sandbox test asserts the disabled message. Nothing is
        # executed: an illegal `emit` is rejected before a container is spent,
        # which is the whole point of the assertion below. Without this the
        # test passed only where ENABLE_UNSAFE_CODE_EXECUTION happened to be
        # on, which is a developer's stack and not CI.
        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)

        result = await sandbox.compile_c_snippet(
            code="int main(void){return 0;}", emit="counts"
        )

        assert "counts" in result["error"], (
            "listing the legal values without naming the illegal one leaves "
            "the caller to guess which argument was wrong"
        )
        assert "asm" in result["error"]

    async def test_a_legal_value_is_not_rejected(self):
        """The control: the guard must only fire on values the tool cannot
        serve."""
        from app.services import agent_compiler_sandbox as sandbox

        result = await sandbox.compile_c_snippet(code="", emit="asm")

        # Empty code is refused for its own reason -- what matters is that the
        # emit check did not fire.
        assert "emit=" not in (result.get("error") or "")
