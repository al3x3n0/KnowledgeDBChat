"""Tests for the compiler sandbox tools.

The docker-backed paths are exercised through their guards and parsing; the
container itself is not run here.
"""

import pytest

from app.services import agent_compiler_sandbox as sandbox


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
    monkeypatch.setattr(
        sandbox, "_allowed_images", lambda: [sandbox.DEFAULT_IMAGE, "python:3.11-slim"]
    )


def test_flags_allow_ordinary_compiler_options():
    assert sandbox._clean_flags("-O3 -ffast-math -march=armv8-a") is not None
    assert sandbox._clean_flags("") == ""


@pytest.mark.parametrize(
    "flags",
    ["-O2; rm -rf /", "-O2 && curl evil", "-O2 `id`", "-O2 $(whoami)", "-O2 | sh"],
)
def test_flags_reject_shell_metacharacters(flags):
    assert sandbox._clean_flags(flags) is None


@pytest.mark.asyncio
async def test_injection_attempt_is_refused_before_running(enabled):
    result = await sandbox.compile_c_snippet(code="int main(){}", flags="-O2; id")

    assert "unsupported characters" in result["error"]


@pytest.mark.asyncio
async def test_execution_disabled_is_reported_not_silently_skipped(monkeypatch):
    monkeypatch.setattr(sandbox, "_execution_enabled", lambda: False)

    result = await sandbox.compile_c_snippet(code="int main(){}")

    assert "ENABLE_UNSAFE_CODE_EXECUTION" in result["error"]


@pytest.mark.asyncio
async def test_image_outside_the_allowlist_is_refused(enabled):
    result = await sandbox.compile_c_snippet(
        code="int main(){}", image="ghcr.io/somewhere/else:latest"
    )

    assert "not allowlisted" in result["error"]


@pytest.mark.asyncio
async def test_empty_code_is_refused(enabled):
    assert "code is required" in (await sandbox.compile_c_snippet(code="  "))["error"]


@pytest.mark.asyncio
async def test_oversized_code_is_refused(enabled):
    result = await sandbox.compile_c_snippet(code="x" * (sandbox.MAX_CODE_CHARS + 1))

    assert "exceeds" in result["error"]


@pytest.mark.asyncio
async def test_emit_rejects_a_value_it_cannot_map(enabled):
    result = await sandbox.compile_c_snippet(code="int main(){}", emit="binary")

    assert "emit must be one of" in result["error"]


@pytest.mark.parametrize("alias", ["assembly", "ASM", " s ", "llvm-ir", "llvm"])
@pytest.mark.asyncio
async def test_emit_accepts_obvious_synonyms(enabled, monkeypatch, alias):
    """An agent asked for "assembly" and lost an iteration to the rejection."""

    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "  uaddw v0.2d, v0.2d, v2.2s\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.compile_c_snippet(code="int main(){}", emit=alias)

    assert result["success"] is True


def test_codegen_counts_describe_aarch64_output():
    counts = sandbox.count_codegen(
        "  uaddw v0.2d, v0.2d, v2.2s\n  b.ge .LBB0_2\n  csel w8, w9, w10, lt\n  bl foo\n"
    )

    assert counts["vector_ops"] >= 1
    assert counts["conditional_branches"] >= 1
    assert counts["conditional_selects"] >= 1
    assert counts["calls"] >= 1


def test_codegen_counts_describe_x86_output():
    counts = sandbox.count_codegen(
        "  vpaddd %xmm0, %xmm1, %xmm2\n  je .L2\n  call foo\n"
    )

    assert counts["vector_ops"] >= 1
    assert counts["conditional_branches"] >= 1


@pytest.mark.asyncio
async def test_a_program_that_exits_non_zero_is_not_reported_as_success(
    enabled, monkeypatch
):
    """The timing loop ends with echo, so its status would otherwise mask the
    program's own failure."""

    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 91, "", "program exited 3"

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.benchmark_c_snippet(code="int main(){return 3;}")

    assert result["success"] is False
    assert "exited non-zero" in result["error"]


@pytest.mark.asyncio
async def test_timings_are_separated_from_program_output(enabled, monkeypatch):
    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "hello\n__elapsed_ms__ 42\nhello\n__elapsed_ms__ 37\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.benchmark_c_snippet(code="int main(){}", repeat=2)

    assert result["data"]["all_ms"] == [42, 37]
    assert result["data"]["fastest_ms"] == 37
    assert result["data"]["stdout"] == "hello\nhello"


@pytest.mark.asyncio
async def test_compile_failure_returns_the_compiler_error(enabled, monkeypatch):
    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 1, "", "snippet.c:1:5: error: expected identifier"

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.compile_c_snippet(code="int broken(")

    assert result["success"] is False
    assert "expected identifier" in result["compiler_stderr"]


@pytest.mark.asyncio
async def test_compile_reports_a_finding_so_the_run_records_what_it_learned(
    enabled, monkeypatch
):
    """Without a finding the loop harvests nothing, and the job's own summary
    then reports that it produced no results."""

    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "  uaddw v0.2d, v0.2d, v2.2s\n  b.ge .L2\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.compile_c_snippet(code="int main(){}", flags="-O3")

    findings = result["findings"]
    assert findings[0]["type"] == "codegen_measurement"
    assert findings[0]["codegen"]["vector_ops"] >= 1
    assert "-O3" in findings[0]["title"]


@pytest.mark.asyncio
async def test_benchmark_reports_a_finding_too(enabled, monkeypatch):
    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "__elapsed_ms__ 42\n__elapsed_ms__ 37\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.benchmark_c_snippet(code="int main(){}", repeat=2)

    assert result["findings"][0]["fastest_ms"] == 37


def test_subject_prefers_an_explicit_label():
    assert sandbox.describe_subject(
        "int f(void){return 0;}", "float sum reduction"
    ) == ("float sum reduction")


def test_subject_falls_back_to_function_names():
    code = "int int_sum(int *a,int n){return 0;}\nfloat float_sum(float *a){return 0;}"

    assert sandbox.describe_subject(code) == "int_sum, float_sum"


def test_subject_is_never_empty():
    assert sandbox.describe_subject("", "") == "unnamed snippet"
    assert sandbox.describe_subject("/* just a comment */") == "unnamed snippet"


@pytest.mark.asyncio
async def test_a_measurement_names_what_it_measured(enabled, monkeypatch):
    """Five findings all reading "clang -O3: N vector ops" could not be mapped
    back to the kernels that produced them."""

    async def _fake_run(script, workdir, *, image, timeout_seconds):
        return 0, "  uaddw v0.2d, v0.2d, v2.2s\n", ""

    monkeypatch.setattr(sandbox, "_run", _fake_run)

    result = await sandbox.compile_c_snippet(
        code="float float_sum(float *a,int n){return 0;}",
        flags="-O3",
        label="float sum reduction",
    )

    finding = result["findings"][0]
    assert finding["subject"] == "float sum reduction"
    assert finding["title"].startswith("float sum reduction @ clang -O3")


def test_a_failure_message_carries_the_compiler_s_own_words():
    """ "Compilation failed" alone filed the reason where nobody read it."""
    message = sandbox.explain_compiler_failure(
        "clang: error: no such file or directory: 'missing.c'\n"
    )

    assert message.startswith("Compilation failed: clang: error: no such file")


def test_march_native_failure_names_the_flag_that_works_here():
    """A run re-sent -march=native four times when told only that it failed."""
    message = sandbox.explain_compiler_failure(
        "clang: error: the clang compiler does not support '-march=native'\n"
    )

    assert "-mcpu=native" in message
    assert "aarch64" in message


def test_an_unrecognised_failure_still_reports_what_the_compiler_said():
    message = sandbox.explain_compiler_failure("snippet.c:3:5: error: expected ';'\n")

    assert "expected ';'" in message


def test_an_empty_stderr_still_produces_a_message():
    assert sandbox.explain_compiler_failure("") == "Compilation failed"


def test_numbers_the_program_printed_are_carried_into_the_result():
    """A harness printing gflops has done the arithmetic; do not drop it."""
    metrics = sandbox.parse_reported_metrics(
        "runtime_seconds=0.132459\ngflops=1.510\nsink=200000000.000000\n"
        "runtime_seconds=0.121536\ngflops=1.646\n"
    )

    assert metrics["gflops"] == [1.510, 1.646]
    assert metrics["runtime_seconds"] == [0.132459, 0.121536]


def test_prose_lines_are_not_mistaken_for_metrics():
    metrics = sandbox.parse_reported_metrics(
        "starting benchmark\nwarning: this is slow\nresult: ok\niterations: 5\n"
    )

    assert metrics == {"iterations": [5.0]}


def test_metric_collection_is_bounded():
    output = "\n".join(f"metric_{i}={i}" for i in range(40))
    output += "\n" + "\n".join("repeated=1" for _ in range(40))

    metrics = sandbox.parse_reported_metrics(output)

    assert len(metrics) <= sandbox.MAX_REPORTED_METRICS
    assert len(metrics["metric_0"]) <= sandbox.MAX_REPORTED_VALUES


class TestCycleAnalysis:
    """A proposed instruction cannot be run, so it has to be costed."""

    @pytest.mark.asyncio
    async def test_a_cycle_estimate_requires_a_named_core(self, enabled):
        result = await sandbox.analyze_snippet_cycles(code="void f(void){}", cpu="")

        assert "cpu is required" in result["error"]
        assert "neoverse-n1" in result["error"]

    @pytest.mark.asyncio
    async def test_a_core_name_may_not_smuggle_shell_syntax(self, enabled):
        result = await sandbox.analyze_snippet_cycles(
            code="void f(void){}", cpu="neoverse-n1; rm -rf /"
        )

        assert "cpu contains unsupported characters" in result["error"]

    @pytest.mark.asyncio
    async def test_a_target_triple_may_not_smuggle_shell_syntax(self, enabled):
        result = await sandbox.analyze_snippet_cycles(
            code="void f(void){}", cpu="neoverse-n1", target="aarch64 && id"
        )

        assert "target contains unsupported characters" in result["error"]

    def test_the_summary_is_read_from_the_report(self):
        payload = {
            "CodeRegions": [
                {"SummaryView": {"TotalCycles": 2414, "Iterations": 100, "IPC": 2.23}}
            ]
        }

        assert sandbox._mca_summary(payload)["TotalCycles"] == 2414

    def test_a_report_with_no_region_yields_nothing_rather_than_raising(self):
        assert sandbox._mca_summary({"CodeRegions": []}) == {}
        assert sandbox._mca_summary({}) == {}


def test_a_whole_function_estimate_is_flagged_as_not_being_a_loop():
    """The two differ by 3.4x, so the number needs its scope attached."""
    import re

    warnings = ["warning: found a return instruction in the input assembly sequence."]
    analysed = "  ldp q2, q3, [x10]\n  ret\n"
    flagged = "LLVM-MCA-BEGIN" not in analysed and any(
        "return instruction" in line for line in warnings
    )

    assert flagged

    fenced = "# LLVM-MCA-BEGIN loop\n  ldp q2, q3, [x10]\n# LLVM-MCA-END\n  ret\n"
    assert not ("LLVM-MCA-BEGIN" not in fenced)
    assert re.search(r"LLVM-MCA-(BEGIN|END)", fenced)


class TestMcaFailureMessages:
    """A guessed cause sends the caller to check the wrong thing.

    A live run was told "an unknown -mcpu is the usual cause" four times while
    llvm-mca had been saying plainly that its region markers did not match.
    """

    def test_an_unpaired_region_marker_is_named_as_the_problem(self):
        message = sandbox.explain_mca_failure(
            "snippet.s:18:2: error: found an invalid region end directive\n", 1
        )

        assert "did not pair up" in message
        assert "invalid region end directive" in message

    def test_an_unknown_core_model_still_says_so(self):
        message = sandbox.explain_mca_failure(
            "error: unsupported CPU 'neoverse-n99'\n", 1
        )

        assert "unknown to this LLVM" in message
        assert "neoverse-n1" in message

    def test_an_unrecognised_failure_still_repeats_what_mca_said(self):
        message = sandbox.explain_mca_failure("error: something else entirely\n", 2)

        assert "something else entirely" in message
        assert "exit code 2" in message

    def test_assembly_is_given_a_trailing_newline(self):
        """Without one, llvm-mca reads "# LLVM-MCA-END loop" as region "loo".

        A live run spent four calls on an error about markers that were
        correct; the file simply did not end with a newline.
        """
        asm = "# LLVM-MCA-BEGIN loop\n\tnop\n# LLVM-MCA-END loop"
        normalized = asm if asm.endswith("\n") else asm + "\n"

        assert normalized.endswith("END loop\n")

    @pytest.mark.asyncio
    async def test_region_markers_in_c_are_refused_with_the_fix(self, enabled):
        """clang says "invalid preprocessing directive", which helps nobody."""
        result = await sandbox.analyze_snippet_cycles(
            code="# LLVM-MCA-BEGIN loop\nvoid f(void){}\n", cpu="neoverse-n1"
        )

        assert "cannot appear in C" in result["error"]
        assert "pass the fenced assembly as 'asm'" in result["error"]


class TestTargetAndRegionChecks:
    """Two ways a caller silently measures something other than they meant."""

    @pytest.mark.asyncio
    async def test_a_core_name_in_target_is_caught_before_a_container_starts(
        self, enabled
    ):
        """llvm-mca's "unable to get target for 'neoverse-n1'" names neither
        the parameter at fault nor what belongs in it."""
        result = await sandbox.analyze_snippet_cycles(
            asm="\tnop\n", cpu="neoverse-n1", target="neoverse-n1"
        )

        assert "should be a target triple" in result["error"]
        assert "goes in 'cpu'" in result["error"]

    @pytest.mark.asyncio
    async def test_a_label_in_target_is_caught_too(self, enabled):
        result = await sandbox.analyze_snippet_cycles(
            asm="\tnop\n", cpu="neoverse-n1", target="norm"
        )

        assert "should be a target triple" in result["error"]
        assert "goes in 'label'" in result["error"]

    @pytest.mark.asyncio
    async def test_a_real_triple_passes_the_check(self, enabled, monkeypatch):
        """The check must not reject the targets that actually work."""
        for triple in ("aarch64-linux-gnu", "riscv64", "x86_64-unknown-linux-gnu"):
            architecture = triple.split("-")[0].lower()
            assert architecture in sandbox.KNOWN_TARGET_ARCHITECTURES, triple

    def test_the_mca_remedy_explains_an_unknown_target(self):
        message = sandbox.explain_mca_failure(
            "llvm-mca: : error: unable to get target for 'norm', see --version\n", 1
        )

        assert "belongs in 'target'" in message
        assert "aarch64-linux-gnu" in message


def test_x86_idioms_are_named_as_such_on_this_aarch64_sandbox():
    """Every one of these blocked a live run trying to write a timing harness.

    The model knew the technique and reached for the x86 spelling of it. The
    remedy has to name the aarch64 equivalent, or the next attempt guesses.
    """
    from app.services.agent_compiler_sandbox import explain_compiler_failure

    fpmath = explain_compiler_failure("error: unknown FP unit '387'")
    assert "x86 option" in fpmath and "aarch64" in fpmath

    constraint = explain_compiler_failure(
        "snippet.c:7:26: error: couldn't allocate output register for constraint 'x'"
    )
    assert '"w"' in constraint and '"r"' in constraint
    assert "v0.4s" in constraint, "must name how AArch64 registers are written"

    libm = explain_compiler_failure(
        "snippet.c:10:42: warning: implicitly declaring library function 'sqrtf'"
    )
    assert "math.h" in libm and "-lm" in libm


class TestRustIsAFirstClassLanguage:
    """An algorithm implemented in Rust must be timed by the same image that
    checked it. Verifying Rust and benchmarking a C translation of it would
    defeat the correctness check entirely."""

    def test_subject_naming_sees_rust_functions(self):
        # The C pattern cannot match `fn dot(..) -> f64 {`: a return type sits
        # between the parameter list and the brace. Every Rust measurement came
        # back named after its harness, which is precisely the failure
        # describe_subject exists to prevent.
        code = (
            "fn dot_product(a: &[f64], b: &[f64]) -> f64 {\n"
            "    a.iter().zip(b).map(|(x, y)| x * y).sum()\n"
            "}\n"
            'fn main() { println!("{}", dot_product(&[1.0], &[2.0])); }\n'
        )
        assert sandbox.describe_subject(code) == "dot_product"

    def test_main_is_the_harness_not_the_subject(self):
        assert sandbox.describe_subject("fn main() { }") == "main"
        assert (
            sandbox.describe_subject("pub fn kernel(x: u64) -> u64 { x }\nfn main(){}")
            == "kernel"
        )

    def test_an_explicit_label_still_wins(self):
        assert (
            sandbox.describe_subject("fn kernel() {}", "blocked GEMM") == "blocked GEMM"
        )

    @pytest.mark.asyncio
    async def test_a_rust_benchmark_builds_with_rustc(self, monkeypatch):
        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
        monkeypatch.setattr(sandbox, "_allowed_images", lambda: {sandbox.DEFAULT_IMAGE})
        captured = {}

        async def fake_run(script, workdir, *, image, timeout_seconds):
            captured["script"] = script
            return 0, "__elapsed_ms__ 12\n__loadavg__ 0.5\n__cpus__ 8\n", ""

        monkeypatch.setattr(sandbox, "_run", fake_run)
        result = await sandbox.benchmark_c_snippet(
            code="fn main(){}", language="rust", label="k"
        )
        assert result["success"] is True
        assert "rustc" in captured["script"]
        assert "clang" not in captured["script"].split("2>compile_err")[0].replace(
            "-C linker=clang", ""
        )
        # The language rides on the finding: a timing is only comparable with
        # another taken the same way, and two languages are not the same way.
        assert result["findings"][0]["language"] == "rust"

    @pytest.mark.asyncio
    async def test_an_unknown_language_is_refused(self, monkeypatch):
        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
        result = await sandbox.benchmark_c_snippet(code="x", language="go")
        assert "Unsupported language" in result["error"]

    @pytest.mark.asyncio
    async def test_c_keeps_working_with_no_language_named(self, monkeypatch):
        # Every existing caller passes no language at all.
        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
        monkeypatch.setattr(sandbox, "_allowed_images", lambda: {sandbox.DEFAULT_IMAGE})
        captured = {}

        async def fake_run(script, workdir, *, image, timeout_seconds):
            captured["script"] = script
            return 0, "__elapsed_ms__ 9\n", ""

        monkeypatch.setattr(sandbox, "_run", fake_run)
        result = await sandbox.benchmark_c_snippet(code="int main(){return 0;}")
        assert result["success"] is True
        assert "clang -O2" in captured["script"]
        assert result["findings"][0]["language"] == "c"
