"""Strict about whether the evidence exists, generous about how it is spelled.

Every accepted case here is one a live model actually wrote and was refused
for, while holding the evidence it was citing. Every rejected case is one where
there is genuinely nothing to check the claim against.
"""

from __future__ import annotations

from app.services import agent_evidence_citation as citation

AVAILABLE = [
    "benchmark_measurement",
    "codegen_measurement",
    "document",
    "simulated_measurement",
]


def test_an_exact_type_resolves():
    assert (
        citation.resolve("simulated_measurement", AVAILABLE) == "simulated_measurement"
    )


def test_a_type_with_a_description_appended_resolves():
    """Written verbatim by a live run, and refused three times."""
    cited = "simulated_measurement: fsqrt dependent chain O3CPU simulation"

    assert citation.resolve(cited, AVAILABLE) == "simulated_measurement"


def test_a_sentence_mentioning_one_type_resolves():
    assert (
        citation.resolve("derived from a benchmark_measurement I took", AVAILABLE)
        == "benchmark_measurement"
    )


def test_a_citation_naming_two_types_claims_both():
    """Keeping only the first would narrow the evidence a record rests on."""
    resolved, unresolved = citation.resolve_all(
        ["benchmark_measurement and simulated_measurement"], AVAILABLE
    )

    assert resolved == ["benchmark_measurement", "simulated_measurement"]
    assert unresolved == []


def test_prose_naming_no_type_is_still_refused():
    """This is the check that caught a run predicting from a measurement it
    never obtained; it must not be softened into always passing."""
    cited = "The fsqrt benchmark I recorded this run with fastest_ms 159"

    assert citation.resolve(cited, AVAILABLE) is None
    assert citation.resolve_all([cited], AVAILABLE) == ([], [cited])


def test_a_bare_number_is_refused():
    assert citation.resolve("4", AVAILABLE) is None


def test_a_type_that_does_not_exist_yet_is_refused():
    assert citation.resolve("simulated_measurement", ["document"]) is None


def test_a_type_embedded_in_a_longer_word_is_not_a_mention():
    assert citation.resolve("predocumented notes", ["document"]) is None


def test_duplicate_citations_collapse():
    resolved, _ = citation.resolve_all(
        ["benchmark_measurement", "benchmark_measurement: retry"], AVAILABLE
    )

    assert resolved == ["benchmark_measurement"]


def test_the_refusal_lists_the_exact_spellings_accepted():
    message = citation.explain_unresolved(["4"], AVAILABLE)

    assert "benchmark_measurement" in message and "simulated_measurement" in message
    assert "exact finding types" in message


def test_nothing_recorded_yet_is_said_plainly():
    assert "none recorded yet" in citation.explain_unresolved(["x"], [])
