"""An extracted kernel is checked against the repository's own function.

Live, on raylib: run 8's kernel came back faithful, and the same kernel with
its alpha tinting removed -- the specialisation two earlier runs made without
saying so -- came back extraction_unfaithful on the zero-alpha input.
"""

import asyncio

import pytest

from app.services import agent_restructure as r


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "rtextures.c").write_text("void ImageColorTint(void) {}\n")
    return tmp_path


def _ref(**over):
    ref = {
        "adapter": "void f(void) {}",
        "paths": ["src/rtextures.c"],
        "include_dirs": ["src"],
    }
    ref.update(over)
    return ref


def test_the_reference_spec_is_checked_before_anything_runs(repo):
    assert (
        "adapter is required" in r._check_reference_spec(_ref(adapter=""), str(repo))[0]
    )
    assert "paths is required" in r._check_reference_spec(_ref(paths=[]), str(repo))[0]
    assert (
        "not in the workspace"
        in r._check_reference_spec(_ref(paths=["src/x.c"]), str(repo))[0]
    )
    assert (
        "plain repository-relative"
        in r._check_reference_spec(_ref(paths=["../etc/x.c"]), str(repo))[0]
    )
    assert "clone_and_index_repo" in r._check_reference_spec(_ref(), "")[0]
    problem, spec = r._check_reference_spec(_ref(paths="src/rtextures.c"), str(repo))
    assert problem is None and spec["paths"] == ["src/rtextures.c"]


@pytest.mark.parametrize(
    "comparison,verdict",
    [
        (
            {"verdict": "equivalent", "equivalence": {"status": "equivalent"}},
            "faithful",
        ),
        (
            {
                "verdict": "diverged",
                "equivalence": {
                    "first_problem": {
                        "input": 2,
                        "expected_excerpt": "a",
                        "actual_excerpt": "b",
                    }
                },
            },
            "extraction_unfaithful",
        ),
        (
            {
                "verdict": "crashed",
                "equivalence": {"first_problem": {"detail": "exited 139"}},
            },
            "reference_crashed",
        ),
        (
            {"verdict": "did_not_compile", "compile_errors": "undefined reference"},
            "reference_did_not_build",
        ),
    ],
)
def test_each_comparison_outcome_maps_to_its_own_verdict(
    repo, monkeypatch, comparison, verdict
):
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return comparison

    monkeypatch.setattr(r, "run_comparison", fake)
    out = asyncio.run(
        r.check_extraction(
            kernel="k",
            driver="int main(){}",
            inputs=["1"],
            reference=_ref(),
            root=str(repo),
        )
    )
    assert out["verdict"] == verdict
    # The repository is linked leniently, and nothing is timed.
    assert seen["time_it"] is False and seen["tree"] == str(repo)
    assert "--unresolved-symbols=ignore-all" in seen["arms"][1].build


def test_an_unfaithful_extraction_names_both_outputs(repo, monkeypatch):
    async def fake(**kwargs):
        return {
            "verdict": "diverged",
            "equivalence": {
                "first_problem": {
                    "input": 2,
                    "expected_excerpt": "ea98cecf",
                    "actual_excerpt": "106406d2",
                }
            },
        }

    monkeypatch.setattr(r, "run_comparison", fake)
    out = asyncio.run(
        r.check_extraction(
            kernel="k",
            driver="int main(){}",
            inputs=["1"],
            reference=_ref(),
            root=str(repo),
        )
    )
    assert "kernel printed 'ea98cecf'" in out["detail"]
    assert "repository's function printed '106406d2'" in out["detail"]


def test_a_refusal_is_never_a_win():
    out = r.extraction_refusal(
        {"verdict": "extraction_unfaithful", "detail": "differs"}, "tint"
    )
    assert out["success"] is False and out["findings"][0]["verified_win"] == 0
    assert out["findings"][0]["extraction"] == "unfaithful"


def test_only_a_verified_extraction_earns_verified_win():
    common = dict(
        kind="restructuring_result",
        label="x",
        invariant="",
        value_preserving=True,
        n_inputs=3,
    )
    won = {"verdict": "faster", "timing": {"speedup": 2.0}}
    assert (
        r.package(won, extraction_verified=True, **common)["findings"][0][
            "verified_win"
        ]
        == 1
    )
    unchecked = r.package(won, extraction_verified=False, **common)["findings"][0]
    assert (
        unchecked["win"] == 1
        and unchecked["verified_win"] == 0
        and unchecked["extraction"] == "unchecked"
    )
