"""Cases keyed `expected` were dropped as unusable, three calls running.

Measured live in the implement stage:

    cases: [{"input": "", "expected": "PASS\\n"}, {"input": "", "expected": "PASS\\n"}]
    -> "No usable reference cases were supplied"   (reason: no_cases)

Two cases were supplied. The tool wanted `expected_output` and said so, but
never said what it had received, so the caller could not see which half of the
disagreement to change -- and made the identical call twice more. This codebase
already answers exactly this for `emit`, where "assembly" is accepted as "asm"
because rejecting a synonym costs an iteration to learn a naming difference and
teaches nothing about the code.
"""

import pytest

from app.services import agent_implementation_check as impl

pytestmark = pytest.mark.unit


class TestTheVocabularyCallersActuallyUse:
    @pytest.mark.parametrize(
        "key", ["expected_output", "expected", "output", "expects"]
    )
    def test_each_accepted_spelling_yields_a_usable_case(self, key):
        cases = impl.normalize_cases([{"input": "10", key: "55"}])

        assert len(cases) == 1
        assert cases[0]["expected_output"] == "55"
        assert cases[0]["input"] == "10"

    def test_the_live_payload_is_now_usable(self):
        """The exact cases the run supplied."""
        cases = impl.normalize_cases(
            [{"input": "", "expected": "PASS\n"}, {"input": "", "expected": "PASS\n"}]
        )

        assert len(cases) == 2

    def test_canonical_wins_when_both_are_present(self):
        cases = impl.normalize_cases(
            [{"expected_output": "right", "expected": "wrong"}]
        )

        assert cases[0]["expected_output"] == "right"


class TestACaseWithNothingToCheckIsStillDropped:
    """The control. The synonyms must not become a way in for cases that
    cannot pass or fail -- that is the vacuous-truth hole `verified` closes."""

    @pytest.mark.parametrize(
        "case",
        [
            {"input": "10"},
            {"input": "10", "expected": ""},
            {"input": "10", "expected": "   "},
            "not an object",
        ],
    )
    def test_it_is_not_a_case(self, case):
        assert impl.normalize_cases([case]) == []


class TestTheRejectionSaysWhatItGot:
    def test_it_names_the_keys_that_were_supplied(self):
        message = impl.describe_unusable_cases(
            [{"input": "", "whatever": "PASS"}, {"input": "", "whatever": "PASS"}]
        )

        assert "2 case(s)" in message
        assert "whatever" in message, "name what arrived, not only what was wanted"
        assert "expected_output" in message, "and what to call it instead"

    def test_a_non_object_case_is_described_as_such(self):
        assert "none was an object" in impl.describe_unusable_cases(["nope"])

    def test_no_cases_at_all_needs_no_explanation(self):
        assert impl.describe_unusable_cases([]) == ""
        assert impl.describe_unusable_cases(None) == ""
