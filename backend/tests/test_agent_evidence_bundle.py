"""A reproducibility bundle assembled while the run happens.

Built at the end, a bundle depends on the run surviving and the agent
remembering; both have failed in practice. These tests pin the properties that
make an as-it-goes bundle worth having: failures are kept, artifacts are
hashed, and the image is pinned by id rather than tag.
"""

from app.services import agent_evidence_bundle as bundle


def _record(tmp_path, tool, params, result, image_id=""):
    return bundle.record_entry(
        job_id="job-1",
        tool=tool,
        params=params,
        result=result,
        image_id=image_id,
        root=tmp_path,
    )


def test_a_call_is_recorded_with_hashes_of_what_went_in_and_came_out(tmp_path):
    entry = _record(
        tmp_path,
        "compile_c_snippet",
        {"code": "int main(void){return 0;}", "flags": "-O3"},
        {"success": True, "data": {"output": "x" * 3000}},
        image_id="sha256:abc123",
    )

    assert entry["sequence"] == 1
    assert entry["succeeded"] is True
    assert len(entry["params_sha256"]) == 64
    assert len(entry["result_sha256"]) == 64
    assert entry["image_id"] == "sha256:abc123"
    directory = bundle.bundle_dir("job-1", tmp_path) / entry["artifact_dir"]
    assert (directory / "params.json").exists()
    assert (directory / "result.json").exists()


def test_a_large_field_is_spilled_to_its_own_file_for_review(tmp_path):
    """So a reviewer can diff an assembly listing rather than dig it out of JSON."""
    entry = _record(
        tmp_path,
        "compile_c_snippet",
        {"code": "void f(void){}"},
        {"success": True, "data": {"output": "fmla v0.4s\n" * 500}},
    )

    assert "output.txt" in entry["spilled_files"]
    spilled = (
        bundle.bundle_dir("job-1", tmp_path) / entry["artifact_dir"] / "output.txt"
    )
    assert "fmla" in spilled.read_text()


def test_failures_are_recorded_not_filtered(tmp_path):
    """A run cited a measurement from a call that had failed; that is only
    visible if the failure is in the record."""
    _record(
        tmp_path, "analyze_snippet_cycles", {"cpu": "neoverse-n1"}, {"success": True}
    )
    _record(
        tmp_path,
        "analyze_snippet_cycles",
        {"cpu": "bogus"},
        {"success": False, "error": "llvm-mca failed with exit code 1"},
    )

    summary = bundle.summarize("job-1", tmp_path)

    assert summary["entries"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    entries = bundle.read_manifest("job-1", tmp_path)
    assert "llvm-mca failed" in entries[1]["error"]


def test_entries_accumulate_in_order_as_the_run_goes(tmp_path):
    for i in range(3):
        _record(tmp_path, "simulate_c_workload", {"n": i}, {"success": True})

    entries = bundle.read_manifest("job-1", tmp_path)

    assert [e["sequence"] for e in entries] == [1, 2, 3]
    assert all(e["tool"] == "simulate_c_workload" for e in entries)


def test_an_unpinned_entry_is_counted_rather_than_hidden(tmp_path):
    """A bundle that cannot say which image produced a result should say so."""
    _record(
        tmp_path, "axis_prove", {"source": "x"}, {"success": True}, image_id="sha256:d"
    )
    _record(tmp_path, "axis_prove", {"source": "y"}, {"success": True}, image_id="")

    summary = bundle.summarize("job-1", tmp_path)

    assert summary["unpinned_entries"] == 1
    assert summary["image_ids"] == ["sha256:d"]


def test_recording_never_raises_on_unserializable_input(tmp_path):
    """A bundle is a description of the run, not a participant in it."""

    class Opaque:
        pass

    entry = _record(tmp_path, "run_command", {"obj": Opaque()}, {"success": True})

    assert entry is not None


def test_only_evidence_producing_tools_are_bundled():
    assert "simulate_c_workload" in bundle.EVIDENCE_TOOLS
    assert "profile_c_workload" in bundle.EVIDENCE_TOOLS
    assert "axis_prove" in bundle.EVIDENCE_TOOLS
    # A bundle of every search would bury the measurements a reviewer checks.
    assert "search_arxiv" not in bundle.EVIDENCE_TOOLS
    assert "write_progress_report" not in bundle.EVIDENCE_TOOLS
