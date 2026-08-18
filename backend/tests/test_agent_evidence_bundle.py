"""A reproducibility bundle assembled while the run happens.

Built at the end, a bundle depends on the run surviving and the agent
remembering; both have failed in practice. These tests pin the properties that
make an as-it-goes bundle worth having: failures are kept, artifacts are
hashed, and the image is pinned by id rather than tag.
"""

import pytest

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


def test_the_bundle_carries_its_own_verifier_from_the_first_entry(tmp_path):
    """A run that dies partway still leaves a bundle that can check itself."""
    _record(tmp_path, "simulate_c_workload", {"code": "x"}, {"success": True})

    directory = bundle.bundle_dir("job-1", tmp_path)
    verifier = directory / bundle.VERIFIER_NAME
    readme = directory / bundle.README_NAME

    assert verifier.exists() and verifier.stat().st_mode & 0o111
    assert "manifest.jsonl" in verifier.read_text()
    assert "simulate_c_workload x1" in readme.read_text()


def test_integrity_passes_on_an_untouched_bundle(tmp_path):
    _record(tmp_path, "compile_c_snippet", {"code": "a"}, {"success": True})
    _record(tmp_path, "axis_prove", {"source": "b"}, {"success": False, "error": "no"})

    report = bundle.verify_integrity("job-1", tmp_path)

    assert report["intact"] is True
    assert report["entries"] == 2
    assert report["artifacts_checked"] == 4


def test_integrity_notices_an_edited_artifact(tmp_path):
    """The point of hashing: a result that was changed after the fact."""
    entry = _record(tmp_path, "compile_c_snippet", {"code": "a"}, {"success": True})
    tampered = (
        bundle.bundle_dir("job-1", tmp_path) / entry["artifact_dir"] / "result.json"
    )
    tampered.write_text('{"success": true, "data": {"cycles": 1}}')

    report = bundle.verify_integrity("job-1", tmp_path)

    assert report["intact"] is False
    assert report["changed"] == ["1/result.json"]


def test_integrity_notices_a_deleted_artifact(tmp_path):
    entry = _record(tmp_path, "profile_c_workload", {"code": "a"}, {"success": True})
    (
        bundle.bundle_dir("job-1", tmp_path) / entry["artifact_dir"] / "params.json"
    ).unlink()

    report = bundle.verify_integrity("job-1", tmp_path)

    assert report["intact"] is False
    assert report["missing"] == ["1/params.json"]


def test_the_readme_states_what_verification_does_not_prove(tmp_path):
    """Integrity is not reproduction, and the bundle should not imply it is."""
    _record(tmp_path, "simulate_c_workload", {"code": "x"}, {"success": True})

    readme = (bundle.bundle_dir("job-1", tmp_path) / bundle.README_NAME).read_text()

    assert "does not re-execute" in readme
    assert "does not prove they can be produced again" in readme


class TestReplay:
    """Re-running the recorded calls and judging what came back."""

    @staticmethod
    def _executor(results):
        async def execute(tool, params):
            return results.pop(0)

        return execute

    @pytest.mark.asyncio
    async def test_an_identical_result_counts_as_reproduced(self, tmp_path):
        original = {"success": True, "data": {"cycles_per_iteration": 59.05}}
        _record(tmp_path, "analyze_snippet_cycles", {"cpu": "neoverse-n1"}, original)

        report = await bundle.replay_bundle(
            "job-1", self._executor([dict(original)]), root=tmp_path
        )

        assert report["verdict"] == "reproduced"
        assert report["reproduced"] == 1
        assert report["differed"] == []

    @pytest.mark.asyncio
    async def test_a_differing_measurement_is_reported_with_both_hashes(self, tmp_path):
        _record(
            tmp_path,
            "simulate_c_workload",
            {"code": "x"},
            {"success": True, "data": {"cycles": 1259204}},
        )

        report = await bundle.replay_bundle(
            "job-1",
            self._executor([{"success": True, "data": {"cycles": 999999}}]),
            root=tmp_path,
        )

        assert report["verdict"] == "differed"
        assert report["differed"][0]["tool"] == "simulate_c_workload"
        assert report["differed"][0]["recorded"] != report["differed"][0]["actual"]

    @pytest.mark.asyncio
    async def test_a_changed_timestamp_is_not_a_failure_to_reproduce(self, tmp_path):
        """Otherwise every replay fails and the report teaches nothing."""
        _record(
            tmp_path,
            "profile_c_workload",
            {"code": "x"},
            {
                "success": True,
                "data": {"instructions_executed": 40891677, "timestamp": "t1"},
            },
        )

        report = await bundle.replay_bundle(
            "job-1",
            self._executor(
                [
                    {
                        "success": True,
                        "data": {"instructions_executed": 40891677, "timestamp": "t2"},
                    }
                ]
            ),
            root=tmp_path,
        )

        assert report["verdict"] == "reproduced"

    @pytest.mark.asyncio
    async def test_a_benchmark_is_replayed_but_never_judged(self, tmp_path):
        """Two honest runs of a wall-clock benchmark disagree."""
        _record(
            tmp_path,
            "benchmark_c_snippet",
            {"code": "x"},
            {"success": True, "data": {"fastest_ms": 126}},
        )

        report = await bundle.replay_bundle("job-1", self._executor([]), root=tmp_path)

        assert report["judged"] == 0
        assert report["skipped"][0]["tool"] == "benchmark_c_snippet"
        assert "wall clock" in report["skipped"][0]["why"]
        assert report["verdict"] == "inconclusive"

    @pytest.mark.asyncio
    async def test_a_failed_call_is_not_replayed(self, tmp_path):
        _record(
            tmp_path,
            "analyze_snippet_cycles",
            {"cpu": "bogus"},
            {"success": False, "error": "unknown cpu"},
        )

        report = await bundle.replay_bundle("job-1", self._executor([]), root=tmp_path)

        assert report["judged"] == 0
        assert report["skipped"][0]["why"] == "did not succeed"

    @pytest.mark.asyncio
    async def test_nothing_judged_is_inconclusive_not_reproduced(self, tmp_path):
        """A replay that judged nothing must not read as a bundle that held up."""
        _record(tmp_path, "benchmark_c_snippet", {"code": "x"}, {"success": True})

        report = await bundle.replay_bundle("job-1", self._executor([]), root=tmp_path)

        assert report["verdict"] == "inconclusive"

    def test_canonicalisation_drops_only_volatile_fields(self):
        payload = {
            "success": True,
            "data": {"cycles": 42, "timestamp": "now", "url": "https://signed"},
        }

        canonical = bundle.canonicalize(payload)

        assert canonical == {"success": True, "data": {"cycles": 42}}


class TestImagePortability:
    """A pinned id is only useful if a reader can act on it."""

    def test_a_project_image_says_how_to_rebuild_it(self):
        origin = bundle.image_origin("ghcr.io/al3x3n0/kdbc-profiling-research:latest")

        assert "profiling-research/Dockerfile" in origin["dockerfile"]

    def test_the_axis_image_warns_that_its_context_is_another_repository(self):
        """Building it against this repo silently produces the wrong image."""
        origin = bundle.image_origin("ghcr.io/al3x3n0/kdbc-axis-research:latest")

        assert "AXIS repository" in origin["context"]

    def test_an_unknown_image_claims_no_origin(self):
        assert bundle.image_origin("someone-elses/image:latest") == {}

    def test_the_images_manifest_lists_what_a_replay_would_need(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(
            bundle._image_details,
            "ghcr.io/al3x3n0/kdbc-gem5-research:latest",
            {
                "reference": "ghcr.io/al3x3n0/kdbc-gem5-research:latest",
                "id": "sha256:aaa",
                "size_bytes": 574_000_000,
                "created": "2026-08-18",
                "origin": bundle.image_origin("kdbc-gem5-research"),
            },
        )
        _record(
            tmp_path,
            "simulate_c_workload",
            {"code": "x"},
            {"success": True},
            image_id="sha256:aaa",
        )

        import json as _json

        payload = _json.loads(
            (bundle.bundle_dir("job-1", tmp_path) / bundle.IMAGES_NAME).read_text()
        )

        assert payload["images"][0]["id"] == "sha256:aaa"
        assert "docker load" in payload["images"][0]["obtain"]
        assert "packages move" in payload["images"][0]["obtain"]
        assert payload["unknown_image_ids"] == []

    def test_an_image_the_run_cannot_describe_is_listed_as_unknown(self, tmp_path):
        """Silently omitting it would understate what a replay needs."""
        _record(
            tmp_path,
            "simulate_c_workload",
            {"code": "x"},
            {"success": True},
            image_id="sha256:zzz",
        )

        import json as _json

        payload = _json.loads(
            (bundle.bundle_dir("job-1", tmp_path) / bundle.IMAGES_NAME).read_text()
        )

        assert payload["unknown_image_ids"] == ["sha256:zzz"]

    def test_the_readme_does_not_claim_the_images_are_included(self, tmp_path):
        _record(tmp_path, "simulate_c_workload", {"code": "x"}, {"success": True})

        readme = (bundle.bundle_dir("job-1", tmp_path) / bundle.README_NAME).read_text()

        # Asserted on fragments that do not span the template's line wraps.
        assert "themselves are not here" in readme
        assert "cannot be" in readme and "replayed" in readme
        assert "Reproducing it elsewhere" in readme
