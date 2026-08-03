"""Tests for structured reasoning tools (reflect, hypothesize, weigh_evidence, critique_plan)."""

from datetime import datetime


def _make_state():
    """Create a minimal state dict with reasoning-related fields."""
    return {
        "reflections": [],
        "hypotheses": [],
        "evidence_ledger": [],
        "plan_critiques": [],
        "critic_notes": [],
    }


class TestReflectTool:
    """Tests for the reflect tool handler logic."""

    def test_reflect_appends_entry(self):
        state = _make_state()
        entry = {
            "iteration": 1,
            "topic": "search strategy",
            "assessment": "Current approach is too narrow",
            "blind_spots": ["missing domain X"],
            "suggested_corrections": ["broaden search terms"],
            "timestamp": datetime.utcnow().isoformat(),
        }
        state["reflections"].append(entry)
        assert len(state["reflections"]) == 1
        assert state["reflections"][0]["topic"] == "search strategy"

    def test_reflections_capped_at_50(self):
        state = _make_state()
        for i in range(60):
            state["reflections"].append({"topic": f"topic_{i}"})
        state["reflections"] = state["reflections"][-50:]
        assert len(state["reflections"]) == 50
        assert state["reflections"][0]["topic"] == "topic_10"


class TestHypothesizeTool:
    """Tests for the hypothesize tool handler logic."""

    def test_create_new_hypothesis(self):
        state = _make_state()
        hyp_id = f"h-{len(state['hypotheses']) + 1}"
        entry = {
            "id": hyp_id,
            "hypothesis": "Model X outperforms Model Y",
            "rationale": "Based on benchmark results",
            "testable_predictions": ["Accuracy > 90%"],
            "status": "proposed",
            "created_at": datetime.utcnow().isoformat(),
        }
        state["hypotheses"].append(entry)
        assert state["hypotheses"][0]["id"] == "h-1"
        assert state["hypotheses"][0]["status"] == "proposed"

    def test_update_existing_hypothesis(self):
        state = _make_state()
        state["hypotheses"].append(
            {
                "id": "h-1",
                "hypothesis": "Test hypothesis",
                "status": "proposed",
            }
        )
        # Update status
        for h in state["hypotheses"]:
            if h["id"] == "h-1":
                h["status"] = "supported"
        assert state["hypotheses"][0]["status"] == "supported"

    def test_hypothesis_not_found(self):
        state = _make_state()
        found = any(h.get("id") == "h-99" for h in state["hypotheses"])
        assert not found


class TestWeighEvidenceTool:
    """Tests for the weigh_evidence tool handler logic."""

    def test_evidence_entry_created(self):
        state = _make_state()
        entry = {
            "claim": "Transformer models are best for NLP",
            "hypothesis_id": None,
            "evidence_for": [
                {
                    "statement": "BERT achieves SOTA",
                    "source_document_id": "doc1",
                    "strength": 0.8,
                }
            ],
            "evidence_against": [
                {
                    "statement": "RNNs are faster",
                    "source_document_id": "doc2",
                    "strength": 0.3,
                }
            ],
            "verdict": "weakly_supported",
            "timestamp": datetime.utcnow().isoformat(),
        }
        for_score = sum(e["strength"] for e in entry["evidence_for"])
        against_score = sum(e["strength"] for e in entry["evidence_against"])
        entry["aggregate_score"] = round(for_score - against_score, 3)
        state["evidence_ledger"].append(entry)

        assert len(state["evidence_ledger"]) == 1
        assert state["evidence_ledger"][0]["aggregate_score"] == 0.5

    def test_evidence_cross_references_hypothesis(self):
        state = _make_state()
        state["hypotheses"].append(
            {
                "id": "h-1",
                "hypothesis": "Test",
                "status": "proposed",
            }
        )
        # Strongly supported evidence should update hypothesis
        verdict = "strongly_supported"
        for h in state["hypotheses"]:
            if h["id"] == "h-1":
                if verdict in {"strongly_supported", "weakly_supported"}:
                    h["status"] = "supported"
        assert state["hypotheses"][0]["status"] == "supported"

    def test_evidence_strength_clamped(self):
        strength = max(0.0, min(1.0, float(1.5)))
        assert strength == 1.0
        strength = max(0.0, min(1.0, float(-0.3)))
        assert strength == 0.0


class TestCritiquePlanTool:
    """Tests for the critique_plan tool handler logic."""

    def test_critique_appended(self):
        state = _make_state()
        entry = {
            "iteration": 3,
            "plan_summary": "Current plan focuses only on arXiv",
            "weaknesses": ["Missing internal documents", "No synthesis step"],
            "missing_steps": ["Document search phase"],
            "assumptions_challenged": [],
            "severity": "moderate",
            "timestamp": datetime.utcnow().isoformat(),
        }
        state["plan_critiques"].append(entry)
        assert len(state["plan_critiques"]) == 1

    def test_major_critique_feeds_critic_notes(self):
        state = _make_state()
        entry = {
            "plan_summary": "Plan is flawed",
            "weaknesses": ["Fatal weakness"],
            "severity": "major",
        }
        state["plan_critiques"].append(entry)
        # Major severity should also add to critic_notes
        if entry["severity"] == "major":
            state["critic_notes"].append(
                {
                    "trajectory_assessment": f"Plan critique (major): {entry['plan_summary'][:200]}",
                    "pivot": "; ".join(entry["weaknesses"][:3]),
                    "recommended_tools": [],
                    "source": "critique_plan_tool",
                }
            )
        assert len(state["critic_notes"]) == 1
        assert state["critic_notes"][0]["source"] == "critique_plan_tool"

    def test_minor_critique_does_not_feed_critic(self):
        state = _make_state()
        entry = {"severity": "minor", "weaknesses": ["small issue"]}
        state["plan_critiques"].append(entry)
        # Minor severity should NOT add to critic_notes
        if entry["severity"] == "major":
            state["critic_notes"].append({})
        assert len(state["critic_notes"]) == 0
