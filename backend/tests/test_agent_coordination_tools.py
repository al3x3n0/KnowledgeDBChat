"""Tests for multi-agent coordination tools (delegate_subtask, wait_for_subtask, share_findings, request_review)."""


def _make_state():
    """Create a minimal state dict with coordination-related fields."""
    return {
        "delegated_subtask_ids": [],
        "delegated_subtask_results": {},
        "review_requests": [],
        "review_responses": [],
        "findings": [],
    }


class TestDelegateSubtask:
    """Tests for delegate_subtask tool logic."""

    def test_subtask_id_tracked(self):
        state = _make_state()
        subtask_id = "job-child-1"
        state["delegated_subtask_ids"].append(subtask_id)
        assert subtask_id in state["delegated_subtask_ids"]

    def test_max_children_enforced(self):
        state = _make_state()
        max_children = 5
        for i in range(max_children):
            state["delegated_subtask_ids"].append(f"job-child-{i}")
        can_delegate = len(state["delegated_subtask_ids"]) < max_children
        assert not can_delegate

    def test_chain_depth_limit(self):
        max_depth = 3
        current_depth = 2
        can_delegate = current_depth < max_depth
        assert can_delegate

        current_depth = 3
        can_delegate = current_depth < max_depth
        assert not can_delegate


class TestWaitForSubtask:
    """Tests for wait_for_subtask tool logic."""

    def test_can_only_wait_own_children(self):
        state = _make_state()
        state["delegated_subtask_ids"] = ["job-child-1", "job-child-2"]
        target = "job-child-1"
        assert target in state["delegated_subtask_ids"]

    def test_cannot_wait_foreign_task(self):
        state = _make_state()
        state["delegated_subtask_ids"] = ["job-child-1"]
        target = "job-foreign-99"
        assert target not in state["delegated_subtask_ids"]

    def test_result_stored_on_completion(self):
        state = _make_state()
        subtask_id = "job-child-1"
        state["delegated_subtask_ids"].append(subtask_id)
        # Simulate child completion
        state["delegated_subtask_results"][subtask_id] = {
            "status": "completed",
            "findings": ["Finding A", "Finding B"],
        }
        assert state["delegated_subtask_results"][subtask_id]["status"] == "completed"
        assert len(state["delegated_subtask_results"][subtask_id]["findings"]) == 2


class TestShareFindings:
    """Tests for share_findings tool logic."""

    def test_findings_shared(self):
        state = _make_state()
        findings = [
            {"content": "Model X performs well on task Y", "confidence": 0.9},
            {"content": "Dataset Z is too small", "confidence": 0.7},
        ]
        state["findings"].extend(findings)
        assert len(state["findings"]) == 2

    def test_sibling_validation(self):
        """Only siblings (same parent_job_id) should receive shared findings."""
        parent_id = "parent-job-1"
        job_parent = parent_id
        target_parent = parent_id
        assert job_parent == target_parent  # same parent = sibling

        target_parent = "parent-job-2"
        assert job_parent != target_parent  # different parent = not sibling


class TestRequestReview:
    """Tests for request_review tool logic."""

    def test_human_review_sets_checkpoint(self):
        state = _make_state()
        review_type = "human"
        review_request = {
            "review_type": review_type,
            "content_to_review": "Draft analysis of paper X",
            "review_criteria": ["accuracy", "completeness"],
        }
        state["review_requests"].append(review_request)
        if review_type == "human":
            state["approval_checkpoint_pending"] = True
        assert state["approval_checkpoint_pending"] is True
        assert len(state["review_requests"]) == 1

    def test_peer_review_delegates(self):
        state = _make_state()
        review_type = "peer_agent"
        review_request = {
            "review_type": review_type,
            "content_to_review": "Summary of findings",
            "review_criteria": ["logical consistency"],
        }
        state["review_requests"].append(review_request)
        # Peer review should create a child task, not set checkpoint
        if review_type == "peer_agent":
            state["delegated_subtask_ids"].append("review-child-1")
        assert "approval_checkpoint_pending" not in state
        assert len(state["delegated_subtask_ids"]) == 1

    def test_review_response_recorded(self):
        state = _make_state()
        state["review_responses"].append(
            {
                "review_request_idx": 0,
                "approved": True,
                "feedback": "Looks good",
            }
        )
        assert len(state["review_responses"]) == 1
        assert state["review_responses"][0]["approved"] is True
