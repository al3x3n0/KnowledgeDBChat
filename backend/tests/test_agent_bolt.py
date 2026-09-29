"""The BOLT tools' decisions, pinned without Docker."""

from app.services import agent_bolt as b


class TestOptions:
    def test_the_standard_recipe_passes_its_own_allowlist(self):
        problem, kept = b.check_options(b.STANDARD_RECIPE)
        assert problem is None and kept == b.STANDARD_RECIPE

    def test_options_that_write_files_are_refused(self):
        for bad in (
            "-o /etc/x",
            "-instrumentation-file=/x",
            "-data=/other",
            "-print-all",
        ):
            problem, _ = b.check_options(bad)
            assert problem, bad

    def test_shell_metacharacters_never_pass(self):
        for bad in ("-icf=1;rm", "-reorder-blocks=$(x)", "-lite=1 && true", "`id`"):
            problem, _ = b.check_options(bad)
            assert problem, bad

    def test_values_must_be_ones_the_option_takes(self):
        assert b.check_options("-reorder-blocks=teleport")[0]
        assert b.check_options("-split-functions=yes")[0]
        assert b.check_options("-reorder-blocks")[0]
        assert b.check_options("-split-functions -align-functions=64")[0] is None

    def test_double_dash_spelling_is_normalised(self):
        assert b.check_options("--split-functions")[1] == "-split-functions"


def test_dyno_stats_pair_before_with_after():
    log = (
        "            45352525 : executed forward branches\n"
        "            34569777 : taken branches\n"
        "            49383244 : executed forward branches (+8.9%)\n"
        "             2351531 : taken branches (-93.2%)\n"
    )
    stats = b.parse_dyno(log)
    assert stats["taken branches"] == {
        "before": 34569777,
        "after": 2351531,
        "change_pct": -93.2,
    }


def test_hot_functions_come_from_branch_records():
    fdata = (
        "1 luaV_execute 10 1 luaV_execute 40 0 900\n"
        "1 luaH_get 4 1 luaH_get 8 2 100\n"
        "0 junk\n"
    )
    hot = b.hot_functions(fdata)
    assert hot[0] == {
        "function": "luaV_execute",
        "branch_executions": 900,
        "share": 0.9,
    }


def test_the_timed_input_is_held_out_of_the_profile():
    assert b.default_profile_inputs(5, 0) == [1, 2, 3, 4]
    assert b.default_profile_inputs(1, 0) == [0]


def test_a_refused_configuration_is_repairable_not_a_sandbox_failure():
    import asyncio

    out = asyncio.run(
        b.judge_configuration(profiled={}, options="-o /tmp/x", inputs=["1"])
    )
    assert out["data"]["verdict"] == "did_not_compile"
    assert "not an option this tool allows" in out["data"]["compile_errors"]


def test_run_args_cannot_carry_shell():
    for bad in ("- 1; id", "$(id)", "a|b", "x > y"):
        assert not b.SAFE_RUN_ARGS.match(bad), bad
    assert b.SAFE_RUN_ARGS.match("- 300000")
