"""Guards on the pass builder.

The container is not run here; what is tested is the reasoning that decides
what a run MEANT, which is where this tool's value is.
"""

import pytest

from app.services import agent_pass_builder as pb


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(pb.agent_sandbox_runtime, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        pb.agent_sandbox_runtime, "allowed_images", lambda: [pb.DEFAULT_IMAGE]
    )


class TestTellingTheFourOutcomesApart:
    """A pass can fail four ways that look alike and need different fixes."""

    IR = """; ModuleID = 'before.ll'
source_filename = "t.c"

define double @f(double %a) {
  %1 = fmul double %a, %a
  %2 = call double @sqrt(double %1)
  ret double %2
}
"""

    def test_a_filename_echo_is_not_a_transformation(self):
        """baseline.ll and after.ll are produced from differently named inputs,
        so `; ModuleID = ...` always differs. Comparing bytes reported every
        pass as having fired, including one that touches nothing."""
        other = self.IR.replace("before.ll", "baseline.ll")
        assert self.IR != other
        assert pb.normalised(self.IR) == pb.normalised(other)

    def test_a_rewrite_that_keeps_the_opcode_is_still_visible(self):
        """Replacing libm sqrt with llvm.sqrt is call -> call. An opcode census
        alone called that no change, for a pass whose own output said it had
        rewritten three sites."""
        after = self.IR.replace("@sqrt", "@llvm.sqrt.f64")
        before_c = pb.opcode_counts(self.IR)
        after_c = pb.opcode_counts(after)
        assert before_c != after_c
        assert "call @sqrt" in before_c
        assert "call @llvm.sqrt.f64" in after_c

    def test_identical_ir_yields_no_deltas(self):
        assert pb._deltas(pb.opcode_counts(self.IR), pb.opcode_counts(self.IR)) == {}


class TestWhatItRefusesBeforeStartingAContainer:
    @pytest.mark.asyncio
    async def test_no_source(self, enabled):
        out = await pb.build_llvm_pass(
            source=" ", pass_name="x", test_code="int main(){}"
        )
        assert "source is required" in out["error"]

    @pytest.mark.asyncio
    async def test_a_pass_name_is_interpolated_so_it_is_constrained(self, enabled):
        out = await pb.build_llvm_pass(
            source="x", pass_name="not a name; rm -rf /", test_code="int main(){}"
        )
        assert "not usable" in out["error"]

    @pytest.mark.asyncio
    async def test_test_code_is_required_and_the_reason_is_given(self, enabled):
        """A pass that builds is not a pass that works."""
        out = await pb.build_llvm_pass(source="x", pass_name="p", test_code="")
        assert "test_code is required" in out["error"]
        assert "whether it fired" in out["error"]

    @pytest.mark.asyncio
    async def test_execution_must_be_enabled(self, monkeypatch):
        monkeypatch.setattr(
            pb.agent_sandbox_runtime, "execution_enabled", lambda: False
        )
        out = await pb.build_llvm_pass(
            source="x", pass_name="p", test_code="int main(){}"
        )
        assert "ENABLE_UNSAFE_CODE_EXECUTION" in out["error"]

    @pytest.mark.asyncio
    async def test_an_unlisted_image_is_refused(self, enabled):
        out = await pb.build_llvm_pass(
            source="x", pass_name="p", test_code="int main(){}", image="evil:latest"
        )
        assert "not allowlisted" in out["error"]
