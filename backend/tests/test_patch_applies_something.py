"""A patch that changes nothing is not an applied patch.

`apply_patch` declares that it produces `patch_applied`, and a coding loop's
contract asks for exactly that. It returned this for a diff it could not parse:

    {"success": true,
     "data": {"applied_files": [], "errors": [], "files_count": 0},
     "findings": [{"type": "patch_applied", "applied_files": []}]}

Success, no errors, and the evidence a contract is waiting for -- for a patch
that touched no file at all. A loop could satisfy its contract, believe it had
fixed the code, and watch the tests go on failing for the original reason.

The same shape as a correctness check with no cases: zero of zero is vacuously
fine under any naive counting, and it is the worst possible reading because it
certifies the thing it never examined. Both are now reported as errors, which
also lets the repeat-failure diagnosis see an agent making the same mistake
twice.

Verified end to end against a repository with one genuinely failing test: the
malformed diff is refused with an explanation, the well-formed one applies,
and the gate flips from `5 passed, 1 failed` to `6 passed, 0 failed`.
"""

import pytest

from app.services.code_patch_apply_service import CodePatchApplyService

pytestmark = pytest.mark.unit

MALFORMED = """--- a/src/intervals.py
+++ b/src/intervals.py
@@
-        if start < last[1]:
+        if start <= last[1]:
"""

WELL_FORMED = """--- a/src/intervals.py
+++ b/src/intervals.py
@@ -14,7 +14,7 @@
     merged = [list(ordered[0])]
     for start, end in ordered[1:]:
         last = merged[-1]
-        if start < last[1]:
+        if start <= last[1]:
             last[1] = max(last[1], end)
         else:
             merged.append([start, end])
"""


class TestTheTriggerIsReal:
    def test_a_hunk_header_without_line_numbers_parses_to_nothing(self):
        """The exact input that produced a successful no-op patch.

        A bare `@@` is what a model writes when it is describing a change
        rather than emitting a diff, so this is not an exotic case.
        """
        assert CodePatchApplyService().parse(MALFORMED) == []

    def test_a_proper_unified_diff_parses(self):
        parsed = CodePatchApplyService().parse(WELL_FORMED)
        assert len(parsed) == 1
        assert parsed[0].path == "src/intervals.py"


class TestTheHandlerRefusesANoOp:
    """Guards the two returns, since the handler is a closure over a live
    workspace that a unit test cannot stand up."""

    @staticmethod
    def _body() -> str:
        from pathlib import Path

        source = Path("app/services/agent_tool_dispatch.py")
        if not source.exists():  # pragma: no cover
            source = (
                Path(__file__).resolve().parents[1]
                / "app"
                / "services"
                / "agent_tool_dispatch.py"
            )
        text = source.read_text()
        start = text.index("    async def _apply_patch(")
        return text[start : text.index("    async def _run_repo_tests(")]

    def test_an_unparseable_diff_is_an_error(self):
        body = self._body()
        assert "if not file_diffs:" in body
        # And says what a usable diff looks like, because "it did not parse"
        # leaves the model guessing at the format.
        assert "@@ -12,7 +12,7 @@" in body

    def test_applying_no_files_is_an_error(self):
        body = self._body()
        assert "if not applied_files:" in body

    def test_the_finding_is_only_reached_after_those_guards(self):
        body = self._body()
        assert body.index("if not applied_files:") < body.index(
            '"type": "patch_applied"'
        )
