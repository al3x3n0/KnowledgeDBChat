"""The scoring rules the domain-research orchestrators rank candidates with.

These had no tests because they could not be called: they were nested inside a
2176-line function, and the only way to reach them was to run the whole
orchestrator against a database, an LLM and a job. One end-to-end test covered
them incidentally, which meant a change to a threshold here showed up, if at
all, as a different memo at the other end.
"""

from app.services import agent_domain_research_scoring as scoring


class TestSafeFloat:
    def test_a_number_survives(self):
        assert scoring.safe_float("1.5") == 1.5

    def test_junk_becomes_the_default_rather_than_an_exception(self):
        """These read model output, where a score can arrive as prose."""
        assert scoring.safe_float("about seven") == 0.0
        assert scoring.safe_float(None, 0.35) == 0.35
        assert scoring.safe_float({"score": 1}, -1.0) == -1.0


class TestNormalizeKey:
    def test_text_becomes_a_stable_identifier(self):
        assert scoring.normalize_key("Cache  Behaviour!") == "cache_behaviour"

    def test_two_spellings_of_one_theme_agree(self):
        """This is what deduplicates clusters, so it has to collapse both."""
        assert scoring.normalize_key("Branch-Prediction") == scoring.normalize_key(
            "branch prediction"
        )

    def test_nothing_normalizes_to_empty_rather_than_crashing(self):
        assert scoring.normalize_key(None) == ""
        assert scoring.normalize_key("!!!") == ""


class TestTrackKeywordSets:
    def test_each_track_gets_its_own_vocabulary(self):
        compiler, compiler_prompt = scoring.track_keyword_sets("compiler")
        uarch, uarch_prompt = scoring.track_keyword_sets("microarchitecture")

        assert "vectorization" in compiler and "ipc" not in compiler
        assert "ipc" in uarch and "mlir" not in uarch
        assert compiler_prompt != uarch_prompt

    def test_an_unknown_track_falls_back_to_generic(self):
        generic, prompt = scoring.track_keyword_sets("astrology")
        assert generic == scoring.track_keyword_sets("generic")[0]
        assert "novel" in prompt

    def test_a_caller_mutating_the_set_does_not_edit_the_rule(self):
        """The sets are module constants; handing one out by reference would
        let one call change how every later call scores."""
        first, _ = scoring.track_keyword_sets("compiler")
        first.add("astrology")

        second, _ = scoring.track_keyword_sets("compiler")
        assert "astrology" not in second


class TestTrackFitScore:
    def test_on_topic_text_outscores_off_topic_text(self):
        on = scoring.track_fit_score("compiler", ["LLVM vectorization pass"])
        off = scoring.track_fit_score("compiler", ["a study of medieval poetry"])
        assert on > off

    def test_more_hits_score_higher(self):
        one = scoring.track_fit_score("microarchitecture", ["cache"])
        many = scoring.track_fit_score("microarchitecture", ["cache branch stall ipc"])
        assert many > one

    def test_the_score_is_bounded_at_one(self):
        """Every keyword at once must not produce a score above 1.0, or a
        ranking built on it puts a keyword-stuffed title above everything."""
        keywords, _ = scoring.track_keyword_sets("compiler")
        assert scoring.track_fit_score("compiler", [" ".join(keywords)]) == 1.0

    def test_empty_text_scores_at_the_base_not_at_zero(self):
        """Nothing to judge is a different claim from judged and irrelevant."""
        assert scoring.track_fit_score("generic", []) == 0.5
        assert scoring.track_fit_score("compiler", ["", "   "]) == 0.35


class TestSignalClustersFromIdeas:
    def test_signals_and_idea_titles_both_become_clusters(self):
        clusters = scoring.signal_clusters_from_ideas(
            ["cache pressure"], [{"title": "branch mispredicts"}]
        )
        assert [c["label"] for c in clusters] == [
            "cache pressure",
            "branch mispredicts",
        ]

    def test_one_theme_arriving_twice_is_one_cluster(self):
        clusters = scoring.signal_clusters_from_ideas(
            ["Cache Pressure"], [{"title": "cache  pressure"}]
        )
        assert len(clusters) == 1

    def test_blank_entries_are_dropped(self):
        clusters = scoring.signal_clusters_from_ideas(["", "   ", None], [])
        assert clusters == []

    def test_the_cluster_count_is_capped(self):
        clusters = scoring.signal_clusters_from_ideas(
            [f"signal {i}" for i in range(40)], []
        )
        assert len(clusters) == scoring.MAX_SIGNAL_CLUSTERS

    def test_only_the_best_ideas_contribute(self):
        """Six, so a long idea list cannot crowd out the run's own signals."""
        clusters = scoring.signal_clusters_from_ideas(
            [], [{"title": f"idea {i}"} for i in range(20)]
        )
        assert len(clusters) == 6

    def test_a_long_label_is_truncated_but_the_cluster_survives(self):
        clusters = scoring.signal_clusters_from_ideas(["x" * 500], [])
        assert len(clusters[0]["label"]) == scoring.MAX_CLUSTER_LABEL_CHARS


def _policy(minimum_sources=1, minimum_subscore=0.4):
    return {
        "weights": {"novelty": 0.4, "evidence": 0.3, "testability": 0.3},
        "minimum_subscore": minimum_subscore,
        "minimum_supporting_sources": minimum_sources,
    }


def _source(title, **extra):
    row = {"source_type": "paper", "id": title, "title": title}
    row.update(extra)
    return row


class TestMatchEvidenceSources:
    def test_a_source_named_in_the_title_is_matched(self):
        refs = scoring.match_evidence_sources(
            [],
            "Revisiting cache prefetching",
            "",
            source_rows=[_source("cache prefetching")],
            minimum_supporting_sources=0,
        )
        assert [r["title"] for r in refs] == ["cache prefetching"]

    def test_enough_shared_tokens_count_as_a_match(self):
        """Three tokens, so a passing word in common is not a citation."""
        refs = scoring.match_evidence_sources(
            [],
            "branch predictor accuracy on wide cores",
            "",
            source_rows=[_source("branch predictor accuracy")],
            minimum_supporting_sources=0,
        )
        assert len(refs) == 1

    def test_an_unrelated_source_is_not_matched(self):
        refs = scoring.match_evidence_sources(
            [],
            "cache prefetching",
            "",
            source_rows=[_source("medieval poetry")],
            minimum_supporting_sources=0,
        )
        assert refs == []

    def test_the_supporting_list_is_capped(self):
        rows = [_source(f"cache prefetching {i}") for i in range(20)]
        refs = scoring.match_evidence_sources(
            [], "cache prefetching", "", source_rows=rows, minimum_supporting_sources=0
        )
        assert len(refs) == scoring.MAX_SUPPORTING_SOURCES

    def test_an_unmatched_source_is_used_to_reach_the_minimum(self):
        """Pinned, not endorsed.

        When matching finds too few sources, the list is topped up with
        sources that did not match, so a candidate reaches its quota on
        evidence unrelated to it. Downstream cannot tell the difference
        between a padded citation list and an earned one -- this test exists
        so the behaviour is visible rather than discovered in a memo.
        """
        refs = scoring.match_evidence_sources(
            [],
            "cache prefetching",
            "",
            source_rows=[_source("medieval poetry")],
            minimum_supporting_sources=1,
        )
        assert [r["title"] for r in refs] == ["medieval poetry"]


class TestBuildCandidate:
    def _build(self, item, **over):
        kwargs = {
            "domain": "compilers",
            "track_type": "compiler",
            "previous_idea_titles": set(),
            "scoring_policy": _policy(),
            "confidence_threshold": 0.5,
            "source_rows": [],
        }
        kwargs.update(over)
        return scoring.build_candidate(item, 0, **kwargs)

    def test_an_empty_idea_is_rejected(self):
        assert self._build({"title": "", "hypothesis": "", "opportunity": ""}) is None
        assert self._build("not a dict") is None

    def test_a_repeated_idea_scores_lower_than_a_new_one(self):
        item = {"title": "Vectorize the inner loop", "hypothesis": "it helps"}
        fresh = self._build(item)
        repeat = self._build(
            item,
            previous_idea_titles={scoring.normalize_key("Vectorize the inner loop")},
        )
        assert fresh["is_new"] is True and repeat["is_new"] is False
        assert fresh["overall_score"] > repeat["overall_score"]

    def test_a_candidate_with_no_sources_cannot_pass_the_gate(self):
        """The gate is an AND: a strong idea with no evidence still fails."""
        candidate = self._build(
            {
                "title": "LLVM vectorization pass for kernels",
                "hypothesis": "codegen improves",
                "next_steps": ["a", "b", "c"],
                "confidence": 1.0,
            },
            scoring_policy=_policy(minimum_sources=1),
        )
        assert candidate["supporting_sources"] == []
        assert candidate["passes_threshold"] is False

    def test_a_missing_title_falls_back_to_the_hypothesis(self):
        candidate = self._build({"title": "", "hypothesis": "loop fusion pays off"})
        assert candidate["title"] == "loop fusion pays off"

    def test_a_candidate_with_neither_gets_a_generated_title(self):
        candidate = self._build({"opportunity": "worth a look"})
        assert candidate["title"] == "compilers hypothesis 1"

    def test_confidence_from_the_model_is_clamped(self):
        assert self._build({"title": "x", "confidence": 42})["confidence"] == 1.0
        assert self._build({"title": "x", "confidence": -5})["confidence"] == 0.0
        assert self._build({"title": "x", "confidence": "high"})["confidence"] == 0.55

    def test_every_score_stays_within_range(self):
        candidate = self._build(
            {
                "title": "LLVM MLIR vectorization codegen kernel pipeline tiling",
                "hypothesis": "everything at once",
                "next_steps": ["a", "b", "c", "d", "e"],
                "confidence": 1.0,
            }
        )
        for field in (
            "novelty_score",
            "evidence_score",
            "testability_score",
            "track_fit_score",
            "overall_score",
        ):
            assert 0.0 <= candidate[field] <= 1.0, field

    def test_a_candidate_without_next_steps_is_still_actionable(self):
        assert self._build({"title": "x"})["next_steps"] == [
            "Validate on a bounded benchmark slice"
        ]
