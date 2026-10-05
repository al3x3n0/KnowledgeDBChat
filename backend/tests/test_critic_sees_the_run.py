"""The critic judges the run it is shown, so it must be shown the run.

Measured: with the last six raw records cut at 5,000 characters, a critic
said at 0.9 confidence that a scan run at iteration 2 had never happened,
and forced the same parameterless browse_repo_files twice.
"""

from app.services.autonomous_agent_executor import (
    _critic_action_ledger,
    _repeats_recent_call,
)


def _act(it, tool, params=None, ok=True, error=None, purpose="", node="act"):
    result = {"success": ok}
    if error:
        result["error"] = error
    return {
        "action": {"tool": tool, "params": params or {}, "purpose": purpose},
        "result": result,
        "iteration": it,
        "node": node,
    }


def _progress(it):
    return _act(it, "write_progress_report", {"summary": "x" * 400}, node="summarize")


RUN = [
    _act(
        1,
        "clone_and_index_repo",
        {"repo_url": "https://github.com/raysan5/raylib.git", "branch": "5.0"},
    ),
    _progress(1),
    _act(
        2,
        "scan_for_optimizations",
        {"paths": ["src/rtextures.c"], "include_dirs": ["src"]},
    ),
    _progress(2),
    _act(
        3,
        "get_symbol_context",
        {"symbol_name": "ImageColorTint", "file_path": "src/rtextures.c"},
    ),
    _progress(3),
    _act(4, "search_documents", {"query": "q" * 300}),
    _progress(4),
    _act(
        5,
        "propose_restructurings",
        {"kernel": "k" * 5000, "inputs": ["1 1"]},
        ok=False,
        error="bench_input should be integer",
    ),
    _progress(5),
    _act(6, "benchmark_c_snippet", {"code": "c" * 9000}),
    _progress(6),
    _act(7, "browse_repo_files", {}, purpose="Critic-directed pivot."),
    _progress(7),
]


def test_early_steps_stay_visible_past_support_records_and_large_results():
    ledger = _critic_action_ledger(RUN)
    assert "it2 scan_for_optimizations(paths=['src/rtextures.c']" in ledger
    assert "it3 get_symbol_context(" in ledger
    assert "write_progress_report" not in ledger


def test_large_arguments_are_summarised_and_failures_carry_their_error():
    ledger = _critic_action_ledger(RUN)
    assert "kernel=<5000 chars>" in ledger
    assert (
        "it5 propose_restructurings" in ledger
        and "FAILED: bench_input should be integer" in ledger
    )
    assert "[Critic-directed pivot.]" in ledger
    assert len(ledger) < 2000


def test_a_pivot_does_not_repeat_the_call_just_made():
    state = {"actions_taken": RUN}
    assert _repeats_recent_call(state, {"tool": "browse_repo_files", "params": {}})
    assert not _repeats_recent_call(
        state, {"tool": "browse_repo_files", "params": {"path": "src"}}
    )
