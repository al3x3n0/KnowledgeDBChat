"""Flags sent as a list cost an iteration to re-send as a string.

Measured live, in an implement stage:

    compile_c_snippet(flags=["-O2", "-std=gnu11"])
    -> "field flags should be string, got list"

The coercion layer refuses a multi-item list where a string is wanted, and its
reasoning is right: joining them would be a guess about the separator. But it
names precisely the condition under which it is not a guess. For compiler flags
the separator IS a space -- in the schema's own example ('-O3 -ffast-math'),
and on the command line the value is interpolated into. The run had expressed
exactly what it meant and was refused for the spelling.
"""

import pytest

from app.services.agent_tool_validation import (
    SPACE_SEPARATED_STRING_FIELDS,
    coerce_tool_params,
)

pytestmark = pytest.mark.unit


class TestFlagListsBecomeFlagStrings:
    def test_the_live_payload_is_repaired(self):
        params = {"code": "int main(void){}", "flags": ["-O2", "-std=gnu11"]}

        repaired = coerce_tool_params("compile_c_snippet", params)

        assert params["flags"] == "-O2 -std=gnu11"
        assert "flags" in repaired

    def test_a_single_flag_in_a_list_still_works(self):
        params = {"code": "int main(void){}", "flags": ["-O2"]}

        coerce_tool_params("compile_c_snippet", params)

        assert params["flags"] == "-O2"

    def test_a_plain_string_is_untouched(self):
        params = {"code": "int main(void){}", "flags": "-O2 -std=gnu11"}

        repaired = coerce_tool_params("compile_c_snippet", params)

        assert params["flags"] == "-O2 -std=gnu11"
        assert "flags" not in repaired


class TestItStaysNarrow:
    """The general refusal is right and must survive: for most string fields a
    list IS a real mistake about what the tool does."""

    def test_only_named_fields_are_joined(self):
        assert "flags" in SPACE_SEPARATED_STRING_FIELDS
        # argv-shaped fields are not: a space join is wrong the moment an
        # argument contains a space.
        assert "run_args" not in SPACE_SEPARATED_STRING_FIELDS
        assert "code" not in SPACE_SEPARATED_STRING_FIELDS

    def test_a_token_containing_a_space_is_not_joined(self):
        """Then the list was not a flag list, and joining would change what
        the caller asked for."""
        params = {"code": "x", "flags": ["-O2", "-D NAME=with space"]}

        coerce_tool_params("compile_c_snippet", params)

        assert isinstance(params["flags"], list), "left alone to be refused honestly"

    def test_a_multi_item_list_elsewhere_is_still_refused(self):
        params = {"code": "x", "label": ["one", "two"]}

        coerce_tool_params("compile_c_snippet", params)

        assert params["label"] == ["one", "two"]
