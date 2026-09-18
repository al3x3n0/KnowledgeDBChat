"""Why an inbox item was surfaced.

The first class is the load-bearing one: this module explains a score computed
elsewhere, so if its tokenizer drifts from the learner's it will confidently
name a term that was never scored. That is checked against the real profile
service rather than a fixture, because a copy of the stopword list would agree
with itself while disagreeing with the thing that matters.
"""

import pytest

from app.services import research_discovery_signals as signals
from app.services.research_monitor_profile_service import ResearchMonitorProfileService


class TestItAgreesWithTheThingThatDidTheLearning:
    def test_the_stopwords_are_the_learners_stopwords(self):
        # Drift here means explaining a word the learner never scored.
        assert set(signals.STOPWORDS) == set(ResearchMonitorProfileService.STOPWORDS)

    def test_it_tokenizes_what_the_learner_tokenizes(self):
        text = "Sparse Attention for Long-Context Transformer models, a new dataset"
        assert signals.tokenize(text) == ResearchMonitorProfileService().tokenize(text)

    def test_it_builds_the_phrases_the_learner_stores(self):
        text = "Sparse attention for long context retrieval"
        assert signals.bigrams(signals.tokenize(text)) == (
            ResearchMonitorProfileService().extract_phrases(text)
        )


class TestItKeepsTheTermsTheOldScorerThrewAway:
    BIAS = {
        "token_scores": {"attention": 4, "blockchain": -6},
        "phrase_scores": {"sparse attention": 3},
        "source_type_scores": {"arxiv": 2},
    }

    def explain(self, title, **kw):
        return signals.explain(bias=self.BIAS, title=title, **kw)

    def test_it_names_the_phrase_that_matched(self):
        out = self.explain("Sparse attention at long context")
        assert "sparse attention" in [s.term for s in out.signals]

    def test_the_most_specific_reason_is_read_first(self):
        # A phrase says more about why than a bare word does.
        out = self.explain("Sparse attention methods", item_type="arxiv")
        assert [s.kind for s in out.signals][:2] == ["phrase", "token"]

    def test_a_term_you_dismiss_is_reported_as_such(self):
        (reason,) = self.explain("Blockchain for science").reasons
        assert "dismissed" in reason and "blockchain" in reason

    def test_a_repeated_word_is_one_reason_not_two(self):
        out = self.explain("Attention, attention, and more attention")
        assert [s.term for s in out.signals] == ["attention"]

    def test_nothing_learned_means_nothing_claimed(self):
        out = signals.explain(bias={}, title="Sparse attention")
        assert out.signals == () and out.as_metadata()["score_explained"] is False

    def test_a_missing_profile_is_not_an_error(self):
        assert signals.explain(bias=None, title="x").score == 0


class TestItDoesNotOverclaimWhatTheNumberMeans:
    """The weight is a net over *occurrences*, not a count of items."""

    def test_it_never_says_you_accepted_it_n_times(self):
        out = signals.explain(
            bias={"token_scores": {"attention": 4}}, title="Attention is all"
        )
        (reason,) = out.reasons
        assert "4" not in reason, "the weight is not an item count"
        assert "kept" in reason

    def test_the_exact_weight_survives_for_a_tooltip(self):
        out = signals.explain(
            bias={"token_scores": {"attention": 4}}, title="Attention is all"
        )
        assert out.as_metadata()["discovery_signals"][0]["weight"] == 4

    def test_source_types_are_worded_as_outcomes_not_as_keeps(self):
        # These weights come from follow-up launch/completion, so "items you
        # kept" would be the wrong sentence for them.
        out = signals.explain(
            bias={"source_type_scores": {"arxiv": 3}}, title="x", item_type="arxiv"
        )
        (reason,) = out.reasons
        assert "kept" not in reason and "led somewhere" in reason


class TestTheArithmeticIsUnchanged:
    """The scorer this replaced set these numbers; items must not re-rank."""

    @pytest.mark.parametrize(
        "bias,item_type,expected",
        [
            ({"source_type_scores": {"arxiv": 3}}, "arxiv", 18),  # delta * 6
            ({"token_scores": {"attention": 4}}, None, 4),  # raw
            ({"phrase_scores": {"sparse attention": 3}}, None, 6),  # * 2
        ],
    )
    def test_weights_apply_as_before(self, bias, item_type, expected):
        out = signals.explain(bias=bias, title="Sparse attention", item_type=item_type)
        assert out.score == expected

    def test_only_the_first_ten_tokens_are_scored(self):
        # The window is the old scorer's. A term outside it contributed nothing,
        # so naming it would explain the score with something that is not in it.
        filler = " ".join(f"word{i}" for i in range(12))
        out = signals.explain(
            bias={"token_scores": {"attention": 5}}, title=f"{filler} attention"
        )
        assert out.score == 0 and out.signals == ()

    def test_at_most_four_reasons_are_given(self):
        bias = {"token_scores": {f"term{i}": 3 for i in range(8)}}
        title = " ".join(f"term{i}" for i in range(8))
        assert len(signals.explain(bias=bias, title=title).signals) == 4


class TestTheShortFormAListRowShows:
    """A row has space for the term that matched, not for why it is known."""

    def test_a_phrase_is_shown_as_the_phrase(self):
        (signal,) = signals.explain(
            bias={"phrase_scores": {"sparse attention": 3}}, title="Sparse attention"
        ).signals
        assert signal.label() == "\u201csparse attention\u201d"
        assert "kept" in signal.describe(), "the justification is still available"

    def test_a_source_type_reads_as_a_kind_of_item(self):
        (signal,) = signals.explain(
            bias={"source_type_scores": {"arxiv": 2}}, title="x", item_type="arxiv"
        ).signals
        assert signal.label() == "arxiv items"

    def test_both_forms_reach_the_client(self):
        (payload,) = signals.explain(
            bias={"token_scores": {"attention": 4}}, title="Attention"
        ).as_metadata()["discovery_signals"]
        assert payload["label"] and payload["text"] and payload["favourable"] is True
