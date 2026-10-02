"""Helpers that existed as several identical copies are defined once.

Each of these was copied between modules character for character, which stays
true only until one copy is fixed. The tests name the definition that is
allowed to exist and fail when another appears.
"""

import ast
from datetime import datetime
from pathlib import Path

import pytest

from app.services import agent_failure_diagnosis, agent_repeated_success
from app.services import coding_backlog_decomposition as decomposition
from app.services import config_values
from app.services.bibtex import (
    _bibtex_month_macro,
    _escape_bibtex,
    _extract_arxiv_id,
    _sanitize_bib_filename,
)
from app.services.citation_lines import is_line_citable
from app.services.connectors.repo_tree import RepoTreeMixin
from app.services.repo_analysis_service import parse_repo_url

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"

#: function name -> the one module allowed to define it.
ONE_HOME = {
    "_sanitize_bib_filename": "services/bibtex.py",
    "_escape_bibtex": "services/bibtex.py",
    "_extract_arxiv_id": "services/bibtex.py",
    "_bibtex_month_macro": "services/bibtex.py",
    "_is_line_citable": None,
    "is_line_citable": "services/citation_lines.py",
    "parse_repo_url": "services/repo_analysis_service.py",
    "build_collaboration_user_lookup": "services/collaboration_service.py",
    "_build_collaboration_user_lookup": None,
    "_build_backlog_user_lookup": None,
    "_canonical_params": "services/agent_failure_diagnosis.py",
    # The backlog endpoint and the backlog orchestrator edit one document.
    "timeline_entry": "services/coding_backlog_decomposition.py",
    "_timeline_entry": None,
    "append_backlog_timeline": "services/coding_backlog_decomposition.py",
    "_append_backlog_timeline": None,
    "append_slice_timeline": "services/coding_backlog_decomposition.py",
    "_append_slice_timeline": None,
    "append_lineage_id": "services/coding_backlog_decomposition.py",
    "_append_lineage_id": None,
    "append_artifact_history": "services/coding_backlog_decomposition.py",
    "_append_artifact_history": None,
    "find_slice": "services/coding_backlog_decomposition.py",
    "_find_slice": None,
    "append_unique": "services/coding_backlog_decomposition.py",
    "_append_unique": None,
    "upsert_promotion_decision": "services/coding_backlog_decomposition.py",
    "_upsert_promotion_decision": None,
    "_tree_to_text": "services/connectors/repo_tree.py",
    "_should_ignore": "services/connectors/repo_tree.py",
    "_run_async": None,
    "_arxiv_id_from_source_identifier": "services/paper_enrichment_service.py",
    "_normalize_track_type": "schemas/domain_research_profile.py",
    "normalize_research_mode": "schemas/domain_research_profile.py",
    "_as_list": None,
    "_safe_float": "api/endpoints/agent_jobs.py",  # Optional result; not the same
    "safe_float": "services/config_values.py",
    "_safe_int": None,
    "_positive_int": "services/autonomous_rnd_trajectory_service.py",  # Optional result
    "_as_number": None,
    "_succeeded": None,
    "succeeded": "services/agent_repeated_success.py",
    "_parse_ts": None,
    "_get_owned_job": "services/agent_playbook_service.py",  # raises its own error
    "_require_owned_job": None,
    "get_owned_job": "modules/autonomy/api/owned_job.py",
    "_bib_key_from_uuid": "services/bibtex.py",
    "_launch_mode": "modules/autonomy/application/relaunch_lineage.py",
}

#: A pydantic validator may carry a helper's name; it must hand the work on.
DELEGATING_METHODS = {"_normalize_track_type"}


def _delegates(node) -> bool:
    """`def name(cls, value): return name(value)` is a call, not a copy."""
    return (
        node.name in DELEGATING_METHODS
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Return)
        and isinstance(node.body[0].value, ast.Call)
        and getattr(node.body[0].value.func, "id", None) == node.name
    )


def test_each_helper_is_defined_in_one_place():
    strays = []
    for path in sorted(APP.rglob("*.py")):
        relative = str(path.relative_to(APP))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in ONE_HOME
                and ONE_HOME[node.name] != relative
                and not _delegates(node)
            ):
                strays.append(f"{relative}:{node.lineno} defines {node.name}")
    assert not strays, strays


def test_bibtex_helpers():
    assert _sanitize_bib_filename("") == "refs.bib"
    assert _sanitize_bib_filename("../etc/passwd") == "refs.bib"
    assert _sanitize_bib_filename("papers") == "papers.bib"
    assert _escape_bibtex("  50% of\nA_B & {C}  ") == r"50\% of A\_B \& \{C\}"
    assert _extract_arxiv_id("https://arxiv.org/pdf/2401.01234v2.pdf") == "2401.01234v2"
    assert _extract_arxiv_id("https://example.com/2401.01234") is None
    assert _bibtex_month_macro(datetime(2026, 9, 1)) == "sep"
    assert _bibtex_month_macro(None) is None


def test_citable_lines():
    assert is_line_citable("A claim about caches.")
    for line in ("", "   ", "# Heading", "> quoted", "```python", "---", None):
        assert not is_line_citable(line)


def test_parse_repo_url():
    assert parse_repo_url("https://github.com/raysan5/raylib.git") == (
        "github",
        "raysan5",
        "raylib",
    )
    assert parse_repo_url("git@gitlab.example.org:group/project") == (
        "gitlab",
        "group",
        "project",
    )
    with pytest.raises(ValueError):
        parse_repo_url("https://example.com/nothing")


def test_shared_canonical_params_keeps_each_callers_ignore_set():
    """The function is shared; what counts as noise is not. A failure ignores
    `title`, a success does not, and a success ignores `purpose`."""
    assert (
        agent_failure_diagnosis.IGNORED_PARAMS != agent_repeated_success.IGNORED_PARAMS
    )
    same = agent_repeated_success.signature
    assert same("t", {"q": 1, "purpose": "a"}) == same("t", {"q": 1, "purpose": "b"})
    assert same("t", {"q": 1, "title": "a"}) != same("t", {"q": 1, "title": "b"})

    failed = agent_failure_diagnosis.signature
    assert failed("t", {"q": 1, "title": "a"}, "boom") == failed(
        "t", {"q": 1, "title": "b"}, "boom"
    )
    assert failed("t", {"q": 1, "purpose": "a"}, "boom") != failed(
        "t", {"q": 1, "purpose": "b"}, "boom"
    )


def test_config_readers_are_not_redefined_as_closures():
    """The clamped readers were closures over `cfg` named `_as_int` and
    `_as_float`; other functions share those names with other signatures, so
    this looks at the shape rather than the name."""
    closures = []
    for path in sorted(APP.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and [
                a.arg for a in node.args.args
            ] == ["key", "default", "lo", "hi"]:
                closures.append(f"{path.relative_to(APP)}:{node.lineno} {node.name}")
    assert not closures, closures


def test_config_values():
    cfg = {"n": "7", "bad": "many", "big": 10**6, "f": "0.25"}
    assert config_values.clamped_int(cfg, "n", 3, 0, 50) == 7
    assert config_values.clamped_int(cfg, "bad", 3, 0, 50) == 3
    assert config_values.clamped_int(cfg, "big", 3, 0, 50) == 50
    assert config_values.clamped_int(cfg, "absent", 3, 5, 50) == 5
    assert config_values.clamped_float(cfg, "f", 0.5, 0.0, 1.0) == 0.25
    assert config_values.clamped_float(cfg, "bad", 0.5, 0.0, 1.0) == 0.5
    assert config_values.string_list("a, b,,c ") == ["a", "b", "c"]
    assert config_values.string_list([" a ", "", 3]) == ["a", "3"]
    assert config_values.string_list(None) == []
    assert config_values.coerce_bool("Off", default=True) is False
    assert config_values.coerce_bool("yes", default=False) is True
    assert config_values.coerce_bool("perhaps", default=True) is True
    assert config_values.coerce_bool(None, default=False) is False


def test_decomposition_histories_are_capped_and_tolerant():
    dec = {
        "backlog_timeline": "not a list",
        "planned_slices": [None, {"slice_id": "s1"}],
    }
    for i in range(decomposition.BACKLOG_TIMELINE_LIMIT + 5):
        decomposition.append_backlog_timeline(
            dec, decomposition.timeline_entry(actor="a", action=str(i), note="n")
        )
    assert len(dec["backlog_timeline"]) == decomposition.BACKLOG_TIMELINE_LIMIT
    assert dec["backlog_timeline"][-1]["action"] == "104"
    assert "job_id" not in dec["backlog_timeline"][-1]

    slice_state = decomposition.find_slice(dec, " s1 ")
    assert slice_state == {"slice_id": "s1"}
    assert decomposition.find_slice(dec, "") is None

    decomposition.append_lineage_id(slice_state, "repair_job_ids", "j1")
    decomposition.append_lineage_id(slice_state, "repair_job_ids", "j1")
    decomposition.append_lineage_id(slice_state, "repair_job_ids", None)
    assert slice_state["job_lineage"] == {"repair_job_ids": ["j1"]}

    decomposition.append_artifact_history(slice_state, "proposal", None)
    assert "artifact_history" not in slice_state
    decomposition.append_artifact_history(slice_state, "proposal", "p1")
    assert slice_state["artifact_history"][0]["label"] == "proposal"

    decomposition.upsert_promotion_decision(dec, {"slice_id": "s1", "v": 1})
    decomposition.upsert_promotion_decision(dec, {"slice_id": "s1", "v": 2})
    assert dec["promotion_decisions"] == [{"slice_id": "s1", "v": 2}]


def test_repo_tree_mixin():
    class Repo(RepoTreeMixin):
        ignore_globs = ["*.lock", "vendor/*"]

    repo = Repo()
    assert repo._should_ignore("vendor/x.c") and repo._should_ignore("a.lock")
    assert not repo._should_ignore("src/x.c")
    lines: list = []
    repo._tree_to_text(
        {
            "children": [
                {"type": "file", "name": "b.c"},
                {
                    "type": "directory",
                    "name": "src",
                    "children": [{"type": "file", "name": "a.c"}],
                },
            ]
        },
        "",
        lines,
    )
    assert lines == ["├── src/", "│   └── a.c", "└── b.c"]


def test_numbers_read_from_untrusted_values():
    assert config_values.safe_float("1.5") == 1.5
    assert config_values.safe_float(None, 2.0) == 2.0
    assert config_values.safe_int("x", 4) == 4
    assert config_values.positive_int("3", 9) == 3
    assert config_values.positive_int(0, 9) == 9
    assert config_values.positive_int(None, 9) == 9
    assert config_values.as_number("2") == 2.0
    assert config_values.as_number(True) is None
    assert config_values.as_number("fast") is None


def test_uuid_list_keeps_order_drops_junk_and_caps():
    a, b = (
        "0b0e7c0e-9b1c-4f0e-8f43-5d5f0c2a9a11",
        "1b0e7c0e-9b1c-4f0e-8f43-5d5f0c2a9a11",
    )
    assert config_values.uuid_list([b, "nope", a.upper(), b, None], 10) == [b, a]
    assert config_values.uuid_list([a, b], 1) == [a]
    assert config_values.uuid_list("not a list", 10) == []


def test_parse_iso_naive():
    from app.utils.datetimes import parse_iso_naive

    assert parse_iso_naive("2026-09-01T10:00:00Z") == datetime(2026, 9, 1, 10)
    assert parse_iso_naive("2026-09-01T10:00:00").tzinfo is None
    assert parse_iso_naive("2026-09-01T10:00:00+05:30") == datetime(2026, 9, 1, 4, 30)
    assert parse_iso_naive("yesterday") is None
    assert parse_iso_naive(None) is None


def test_flat_theme_overrides_known_keys_only():
    from app.services.document_styles import FlatThemeMixin

    class Builder(FlatThemeMixin):
        STYLES = {"professional": {"title_color": "#000", "body_size": 11}}

    built = Builder()._parse_custom_theme({"body_size": 14, "invented": 1})
    assert built == {"title_color": "#000", "body_size": 14}
    assert Builder.STYLES["professional"]["body_size"] == 11
    assert set(Builder.get_available_styles()) == {
        "professional",
        "casual",
        "technical",
    }


async def test_a_job_that_is_not_yours_is_not_found(db_session, test_user):
    from uuid import uuid4

    from fastapi import HTTPException

    from app.modules.autonomy.api.owned_job import get_owned_job

    with pytest.raises(HTTPException) as refused:
        await get_owned_job(job_id=uuid4(), user_id=test_user.id, db=db_session)
    assert refused.value.status_code == 404


def test_unique_strings():
    assert config_values.unique_strings([" a ", "a", None, "", 3, "b"], 10) == [
        "a",
        "3",
        "b",
    ]
    assert config_values.unique_strings(["a", "b", "c"], 2) == ["a", "b"]
    assert config_values.unique_strings("a,b", 10) == []
