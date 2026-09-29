"""Guards on the optimisation scanner tool.

The docker-backed path is exercised through its preflight and its parsing; the
container itself is not run here.
"""

import pytest

from app.services import agent_optscan as scan


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(scan.agent_sandbox_runtime, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        scan.agent_sandbox_runtime, "allowed_images", lambda: [scan.DEFAULT_IMAGE]
    )


class TestWhatItRefusesBeforeStartingAContainer:
    @pytest.mark.asyncio
    async def test_no_sources(self, enabled):
        assert (
            "sources is required"
            in (await scan.scan_for_optimizations(sources={}))["error"]
        )

    @pytest.mark.asyncio
    async def test_a_path_is_not_a_filename(self, enabled):
        """The name is written into the sandbox and interpolated into a shell
        command, so it is a bare filename or nothing."""
        out = await scan.scan_for_optimizations(sources={"../../etc/passwd.c": "x"})
        assert "not usable" in out["error"]

    @pytest.mark.asyncio
    async def test_shell_metacharacters_in_flags(self, enabled):
        out = await scan.scan_for_optimizations(
            sources={"a.c": "int main(void){return 0;}"}, flags="-O2; rm -rf /"
        )
        assert "unsupported characters" in out["error"]

    @pytest.mark.asyncio
    async def test_execution_must_be_enabled(self, monkeypatch):
        monkeypatch.setattr(
            scan.agent_sandbox_runtime, "execution_enabled", lambda: False
        )
        out = await scan.scan_for_optimizations(sources={"a.c": "int main(void){}"})
        assert "ENABLE_UNSAFE_CODE_EXECUTION" in out["error"]

    @pytest.mark.asyncio
    async def test_an_unlisted_image_is_refused(self, enabled):
        out = await scan.scan_for_optimizations(
            sources={"a.c": "int main(void){}"}, image="evil:latest"
        )
        assert "not allowlisted" in out["error"]


class TestReadingTheScannerOutput:
    OUTPUT = "\n".join(
        [
            "OPTSCAN\tmeta\tcounts_are_static\twritten not run",
            "OPTSCAN\tknown\tsqrt-errno-elision\tnormalise\t2",
            "OPTSCAN\tdeclined\tsqrt-errno-elision\topaque\t1",
            "OPTSCAN\tknown\tcommon-divisor-reciprocal\tnormalise\t1\tgroup_size=3",
            "OPTSCAN\tcandidate\tloop-invariant-divisor\tinv\t4\tof_loop_divisions=9",
            "OPTSCAN\tshape\tfdiv.numerator\tuitofp\t7",
            "noise that is not a record",
        ]
    )

    def test_it_counts_what_a_pass_can_take(self):
        p = scan.parse_records(self.OUTPUT)
        assert p["known"]["sqrt-errno-elision"] == 2
        assert p["declined"]["sqrt-errno-elision"] == 1

    def test_a_hoistable_division_becomes_a_suggestion(self):
        """The loop-invariant count is a candidate record, not a `known` one,
        and has to be routed to the pass that handles it or the tool reports an
        opportunity with no way to take it."""
        p = scan.parse_records(self.OUTPUT)
        assert p["known"]["loop-invariant-reciprocal"] == 4
        assert p["loop_invariant_divisions"] == 4
        assert p["loop_divisions"] == 9

    def test_shapes_are_kept_apart_from_opportunities(self):
        p = scan.parse_records(self.OUTPUT)
        assert p["shapes"]["fdiv.numerator"]["uitofp"] == 7
        assert "fdiv.numerator" not in p["known"]

    def test_lines_that_are_not_records_are_ignored(self):
        assert scan.parse_records("hello\nworld")["known"] == {}

    def test_every_suggestion_names_a_pass_that_exists(self):
        """A suggestion whose flag is None would tell a caller to apply a pass
        this image does not carry."""
        p = scan.parse_records(self.OUTPUT)
        for s in scan._suggestions(p):
            assert s["flag"], f"{s['pass']} has no plugin path"
            assert s["value_preserving"] is not None


def test_workspace_paths_cannot_escape_or_inject():
    import asyncio

    from app.services import agent_optscan as o

    for bad in (
        "../etc/passwd.c",
        "src/../../x.c",
        "/abs/x.c",
        "a.c; rm -rf /",
        "a'b.c",
    ):
        out = asyncio.run(o.scan_workspace(root="/tmp", paths=[bad]))
        assert "not a plain repository-relative path" in out["error"], bad


def test_each_failed_file_is_named_with_its_own_error():
    from app.services import agent_optscan as o

    script = o._scan_script(["src/a.c", "src/b.c"], "-O1 -Isrc")
    assert "failed_file" in script and "'src/a.c' 'src/b.c'" in script


def test_a_directory_means_the_c_files_directly_in_it(tmp_path, monkeypatch):
    import asyncio

    from app.services import agent_optscan as o

    (tmp_path / "src" / "external").mkdir(parents=True)
    (tmp_path / "src" / "a.c").write_text("int a;")
    (tmp_path / "src" / "b.c").write_text("int b;")
    (tmp_path / "src" / "external" / "vendored.c").write_text("int v;")
    seen = {}

    async def fake_scan(workdir, paths, flags, **kwargs):
        seen["paths"] = paths
        return {"success": True}

    monkeypatch.setattr(o, "_scan_in", fake_scan)
    monkeypatch.setattr(o.agent_sandbox_runtime, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        o.agent_sandbox_runtime, "allowed_images", lambda: [o.DEFAULT_IMAGE]
    )
    asyncio.run(o.scan_workspace(root=str(tmp_path), paths=["src"]))
    assert seen["paths"] == ["src/a.c", "src/b.c"]

    out = asyncio.run(o.scan_workspace(root=str(tmp_path), paths=["nope/x.c"]))
    assert "no such file or directory" in out["error"]
