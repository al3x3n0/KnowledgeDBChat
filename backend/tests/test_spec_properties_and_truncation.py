"""An empty `reference_cases` meant two different things and said neither.

Measured across two live runs of the reproduction pipeline:

* `extract_algorithm_spec` read `doc.content[:12000]`. That cap was generous
  when an ingested "paper" was its 1520-character abstract; against the real
  36,886-character paper it silently kept the first third, so the claims came
  from the abstract and introduction while the algorithm listings and the
  benchmark tables were never read. "The paper gives no worked examples" and
  "none in the third of it I was shown" are different facts.

* The prompt said to leave `reference_cases` empty rather than invent
  examples -- right, and it collapsed two unlike things. A case invented to
  match the implementation is circular. A property the paper's own description
  states ("the output lies in [0,s)") is not invented at all. With no way to
  tell them apart the implement stage spent six iterations hunting for worked
  examples that do not exist in that paper, wrote no code, and ended with its
  contract unmet.
"""

import pytest

pytestmark = pytest.mark.unit


class TestTheExtractorReadsMoreThanAThirdOfAPaper:
    def test_the_window_is_a_setting_and_fits_a_real_paper(self):
        from app.core.config import settings

        window = settings.SPEC_EXTRACTION_MAX_CHARS
        assert window >= 36886, (
            "the paper that exposed this is 36,886 characters; a window under "
            "that truncates it again"
        )

    def test_the_old_hardcoded_cap_is_gone(self):
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        assert "content[:12000]" not in source


class TestTruncationIsSaidNotDoneQuietly:
    def test_the_prompt_tells_the_model_it_is_reading_a_fragment(self):
        """Otherwise the model reports "the paper omits X" about a part it was
        never given."""
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        assert "not all of it" in source
        assert "may be in the part you were not given" in source

    def test_the_finding_records_how_much_was_read(self):
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        for field in ("paper_truncated", "paper_chars_read", "paper_chars_total"):
            assert f'"{field}"' in source, field


class TestPropertiesAreDistinctFromInventedCases:
    def test_the_prompt_asks_for_properties(self):
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        assert '"properties"' in source
        assert "must hold for ANY valid run" in source

    def test_it_still_forbids_inventing_worked_examples(self):
        """The control. The new field must not read as permission to make up
        input/output pairs -- that rule is the reason the gate means anything."""
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        assert "a case you made up checks nothing" in source
        assert "is not an exception to that rule" in source

    def test_the_finding_counts_properties_beside_cases(self):
        import inspect

        from app.services import agent_tool_dispatch

        source = inspect.getsource(agent_tool_dispatch)
        assert '"property_count"' in source
        assert '"reference_case_count"' in source


class TestTheNextStageIsToldWhatToDo:
    def test_check_implementation_says_properties_are_a_valid_check(self):
        from app.agent_core.tool_specs.measurement import SPECS

        spec = next(s for s in SPECS if s.name == "check_implementation")
        cases = spec.parameters["properties"]["cases"]["description"]

        assert "PROPERTIES" in cases
        assert "invented" in cases
        # The consequence, stated: this is what stops a run treating "no cases"
        # as a reason to skip checking.
        assert "no way to be checked is one nobody may time" in cases

    def test_the_spec_tool_mentions_them_too(self):
        from app.agent_core.tool_specs.research import SPECS

        spec = next(s for s in SPECS if s.name == "extract_algorithm_spec")

        assert "properties" in spec.description
