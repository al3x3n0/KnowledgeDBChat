"""Reports must lead with the conclusion.

Exports listed the measurements and left the reader to work out what they
meant, because the exporter predated the conclusion the run now writes.
"""

import pytest

from app.services.job_results_exporter import JobResultsExporter, _conclusion_content


class _Job:
    id = "job-1"
    name = "Vectorization survey"
    job_type = "research"
    goal = "Determine which kernels clang vectorizes at -O3"
    status = "completed"
    progress = 100
    iteration = 4
    max_iterations = 6
    tool_calls_used = 6
    max_tool_calls = 15
    llm_calls_used = 12
    max_llm_calls = 25
    created_at = None
    started_at = None
    completed_at = None
    error = None
    description = None
    execution_log: list = []
    config: dict = {}

    def __init__(self, conclusion=None):
        self.results = {
            "findings": [{"title": "float reduction @ -O3: 0 vector ops"}],
            "summary": "Research run completed: 3 codegen measurement.",
        }
        if conclusion is not None:
            self.results["conclusion"] = conclusion


ANSWERED = {
    "answer": "The float reduction does not vectorize at -O3; -ffast-math unlocks it.",
    "confidence": "high",
    "evidence": ["float reduction @ clang -O3: 0 vector ops"],
    "gaps": ["Only three kernels were tested."],
}


def _texts(blocks):
    return " ".join(
        str(b.get("text", "")) + " " + " ".join(str(i) for i in b.get("items", []))
        for b in blocks
    )


def test_conclusion_leads_with_the_answer():
    blocks = _conclusion_content(_Job(ANSWERED))

    assert blocks[0] == {"type": "heading", "level": 2, "text": "Conclusion"}
    text = _texts(blocks)
    assert "does not vectorize at -O3" in text
    assert "Confidence: high" in text


def test_conclusion_cites_evidence_and_states_gaps():
    text = _texts(_conclusion_content(_Job(ANSWERED)))

    assert "0 vector ops" in text
    assert "Only three kernels were tested." in text


def test_a_run_without_an_answer_says_so_rather_than_omitting_the_section():
    """An absent section reads as an author with nothing to add, not a run
    that could not answer its goal."""
    blocks = _conclusion_content(
        _Job({"answer": None, "gaps": ["The run recorded no findings."]})
    )

    text = _texts(blocks)
    assert "did not reach a conclusion" in text
    assert "The run recorded no findings." in text


def test_a_job_predating_conclusions_exports_unchanged():
    assert _conclusion_content(_Job()) == []


@pytest.mark.parametrize("fmt", ["docx", "pdf"])
def test_document_formats_build_with_a_conclusion(fmt):
    data = JobResultsExporter().export(_Job(ANSWERED), fmt)

    assert isinstance(data, bytes) and len(data) > 1000


def test_the_deck_puts_the_conclusion_next_to_the_goal():
    """python-pptx is stubbed in tests, so assert the outline rather than the
    rendered file."""
    outline = JobResultsExporter()._build_presentation_outline(
        _Job(ANSWERED), include_log=False, include_metadata=True
    )
    titles = [s.title for s in outline.slides]

    assert "Conclusion" in titles
    assert titles.index("Conclusion") == titles.index("Goal") + 1
    conclusion = outline.slides[titles.index("Conclusion")]
    assert any("does not vectorize" in str(c) for c in conclusion.content)


def test_a_deck_for_an_unanswered_run_says_so():
    outline = JobResultsExporter()._build_presentation_outline(
        _Job({"answer": None, "gaps": ["The run recorded no findings."]}),
        include_log=False,
        include_metadata=False,
    )
    slide = next(s for s in outline.slides if s.title == "Conclusion")

    assert any("did not reach a conclusion" in str(c) for c in slide.content)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("**Executive Summary**", "Executive Summary"),
        ("*1. Float reduction fails*", "1. Float reduction fails"),
        ("### Heading", "Heading"),
        ("- bullet point", "bullet point"),
        ("use `-ffast-math` here", "use -ffast-math here"),
        ("**bold** and *italic* mixed", "bold and italic mixed"),
        ("A * B multiplied", "A  B multiplied"),
        ("", ""),
        (None, ""),
    ],
)
def test_model_markdown_is_rendered_as_plain_prose(raw, expected):
    """The builders render text literally, so markdown reached the page with
    its asterisks intact."""
    from app.services.job_results_exporter import markdown_to_plain

    assert markdown_to_plain(raw) == expected


def test_multiline_markdown_keeps_its_paragraphs():
    from app.services.job_results_exporter import markdown_to_plain

    out = markdown_to_plain("**First para**\n\nSecond *para*")

    assert out == "First para\n\nSecond para"


@pytest.mark.parametrize(
    "raw",
    [
        "integer_sum_reduction_O3 @ clang -O3",
        "float_sum_reduction_O3_fastmath",
        "a_b_c_d_e",
        "__dunder__",
    ],
)
def test_snake_case_identifiers_survive_intact(raw):
    """Underscores are markdown italics, and stripping them silently renamed
    every labelled finding in the report."""
    from app.services.job_results_exporter import markdown_to_plain

    assert markdown_to_plain(raw) == raw
