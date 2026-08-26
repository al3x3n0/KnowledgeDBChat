"""Four defects a real PDF export exposed.

The platform's LLM-backed exporter had never been called by anything -- it
turned up in a dead-code scan -- and the first time it was asked to write a
report it produced one that looked complete and was not.
"""

import sys

from app.services.job_results_exporter import JobResultsExporter, _used_of


def _job():
    """A real AgentJob rather than a stand-in.

    The renderer reads a couple of dozen attributes; a hand-rolled double grows
    one AttributeError at a time and still does not prove the renderer works on
    the thing it is given in production.
    """
    from app.models.agent_job import AgentJob, AgentJobStatus

    job = AgentJob(
        name="study",
        goal="measure something",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=11,
        max_iterations=16,
        tool_calls_used=22,
        max_tool_calls=22,
        llm_calls_used=35,
        max_llm_calls=40,
        results={"findings": []},
        config={},
    )
    job.execution_log = []
    job.output_artifacts = []
    return job


class TestBudgetsRenderWithoutNone:
    """A budget is optional; "11/None" is the repr of a missing value, printed
    into a document handed to somebody."""

    def test_a_count_against_its_limit(self):
        assert _used_of(11, 16) == "11/16"

    def test_a_count_with_no_limit_stands_alone(self):
        assert _used_of(11, None) == "11"

    def test_an_unknown_count_says_so(self):
        assert _used_of(None, None) == "N/A"
        assert _used_of(None, 40) == "N/A"

    def test_zero_is_a_count_not_an_absence(self):
        assert _used_of(0, None) == "0"
        assert _used_of(0, 40) == "0/40"

    def test_no_rendering_path_can_print_none(self):
        """Every site that used to interpolate the pair directly."""
        import inspect

        from app.services import job_results_exporter as mod

        source = inspect.getsource(mod)
        for limit in ("max_iterations", "max_tool_calls", "max_llm_calls"):
            assert (
                "{job.%s}" % limit not in source
            ), f"{limit} is interpolated directly; use _used_of"


class TestPdfDoesNotNeedPowerPoint:
    def test_the_module_imports_without_python_pptx(self, monkeypatch):
        """PDF and DOCX export were blocked by a missing PowerPoint library,
        because the PPTX builder was imported at module scope."""
        import importlib

        monkeypatch.setitem(sys.modules, "pptx", None)
        for name in [m for m in list(sys.modules) if "job_results_exporter" in m]:
            monkeypatch.delitem(sys.modules, name, raising=False)

        module = importlib.import_module("app.services.job_results_exporter")
        assert module.JobResultsExporter is not None


class TestAnIncompleteReportSaysSo:
    def test_a_failed_section_is_named_in_the_document(self):
        exporter = JobResultsExporter()
        enhanced = {
            "executive_summary": None,
            "key_insights": None,
            "recommendations": None,
            "enhanced_findings": [],
            "failed_sections": [("Key Insights", "returned no content")],
        }

        content = exporter._build_document_content_enhanced(
            _job(), enhanced, False, False
        )
        text = " ".join(str(block.get("text", "")) for block in content)

        assert "Incomplete Report" in text
        assert "Key Insights" in text
        assert "returned no content" in text

    def test_a_complete_report_carries_no_such_notice(self):
        exporter = JobResultsExporter()
        enhanced = {
            "executive_summary": "All good.",
            "key_insights": "Insightful.",
            "recommendations": "Do the thing.",
            "enhanced_findings": [],
            "failed_sections": [],
        }

        content = exporter._build_document_content_enhanced(
            _job(), enhanced, False, False
        )
        text = " ".join(str(block.get("text", "")) for block in content)

        assert "Incomplete Report" not in text
