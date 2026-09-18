"""What a rejection teaches the monitor profile.

The profile learns from triage by counting the words of accepted and rejected
items. Those are *topic* words, so before rejections carried a reason, turning
down a weak paper on your own subject taught the profile to hide that subject.
The first class is that bug, expressed as a test.
"""

import asyncio

from app.models.research_inbox import ResearchInboxItem
from app.services import research_rejection_reasons as reasons
from app.services.research_monitor_profile_service import ResearchMonitorProfileService

BASE = "/api/v1/research/inbox"


class TestTheVocabulary:
    def test_an_unspecified_reason_still_teaches_the_topic(self):
        # What every rejection meant before reasons existed. Rows recorded
        # under those rules must keep saying it.
        assert reasons.teaches_topic(None) is True
        assert reasons.teaches_topic("") is True

    def test_only_a_subject_rejection_teaches_the_words(self):
        assert reasons.teaches_topic("off_topic") is True
        assert reasons.teaches_topic("low_quality") is False
        assert reasons.teaches_topic("already_known") is False
        assert reasons.teaches_topic("not_now") is False

    def test_an_unknown_reason_is_not_silently_accepted(self):
        assert reasons.normalize("whatever") is None
        assert reasons.is_valid("whatever") is False
        # But an absent one is fine: it means "unspecified", not "invalid".
        assert reasons.is_valid(None) is True

    def test_every_choice_says_what_it_will_do(self):
        # A choice whose effect is unstated is a choice made blind.
        assert all(entry["effect"] for entry in reasons.describe())


class TestWhatTheProfileLearns:
    """Against the real learner, which is the thing that had the bug."""

    def seed(self, db, user_id, rows):
        async def _seed():
            for title, status, reason in rows:
                db.add(
                    ResearchInboxItem(
                        user_id=user_id,
                        item_type="arxiv",
                        item_key=f"k-{title}-{status}-{reason}",
                        title=title,
                        summary="",
                        status=status,
                        rejection_reason=reason,
                    )
                )
            await db.commit()

        asyncio.get_event_loop().run_until_complete(_seed())

    def recompute(self, db, user_id):
        return asyncio.get_event_loop().run_until_complete(
            ResearchMonitorProfileService().recompute_profile(
                db=db, user_id=user_id, customer=None
            )
        )

    def test_rejecting_a_weak_paper_no_longer_hides_its_subject(
        self, db_session, test_user
    ):
        # The bug: "sparse attention done badly" is not "not sparse attention".
        # Two accepts and one quality complaint. The learner keeps a token only
        # at |score| >= 2, so this is the sharp case: with the complaint counted
        # against the topic the score falls to 1 and the word vanishes entirely.
        self.seed(
            db_session,
            test_user.id,
            [
                ("sparse attention kernels", "accepted", None),
                ("sparse attention routing", "accepted", None),
                ("sparse attention benchmark", "rejected", "low_quality"),
            ],
        )
        profile = self.recompute(db_session, test_user.id)
        assert (
            profile.token_scores.get("sparse", 0) >= 2
        ), "a quality complaint must not cancel the subject you accepted"

    def test_rejecting_for_subject_still_teaches_the_words(self, db_session, test_user):
        self.seed(
            db_session,
            test_user.id,
            [
                ("blockchain governance token", "rejected", "off_topic"),
                ("blockchain governance council", "rejected", "off_topic"),
            ],
        )
        profile = self.recompute(db_session, test_user.id)
        assert profile.token_scores.get("blockchain", 0) < 0

    def test_a_rejection_recorded_before_reasons_existed_is_unchanged(
        self, db_session, test_user
    ):
        self.seed(
            db_session,
            test_user.id,
            [
                ("quantum annealing survey", "rejected", None),
                ("quantum annealing review", "rejected", None),
            ],
        )
        profile = self.recompute(db_session, test_user.id)
        assert profile.token_scores.get("quantum", 0) < 0


class TestTheApi:
    def make_item(self, db, user_id, key="api-1"):
        async def _seed():
            row = ResearchInboxItem(
                user_id=user_id,
                item_type="arxiv",
                item_key=key,
                title="Sparse attention",
                status="new",
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
            return row.id

        return asyncio.get_event_loop().run_until_complete(_seed())

    def test_the_vocabulary_is_served_rather_than_restated(self, client, auth_headers):
        response = client.get(f"{BASE}/rejection-reasons", headers=auth_headers)
        assert response.status_code == 200
        keys = {entry["key"] for entry in response.json()["reasons"]}
        assert keys == set(reasons.VALID_KEYS)

    def test_a_reason_is_stored_with_the_rejection(
        self, client, auth_headers, db_session, test_user
    ):
        item_id = self.make_item(db_session, test_user.id, "api-store")
        response = client.patch(
            f"{BASE}/{item_id}",
            headers=auth_headers,
            json={"status": "rejected", "rejection_reason": "low_quality"},
        )
        assert response.status_code == 200
        assert response.json()["rejection_reason"] == "low_quality"

    def test_an_unknown_reason_is_refused_rather_than_stored(
        self, client, auth_headers, db_session, test_user
    ):
        item_id = self.make_item(db_session, test_user.id, "api-bad")
        response = client.patch(
            f"{BASE}/{item_id}",
            headers=auth_headers,
            json={"status": "rejected", "rejection_reason": "because i said so"},
        )
        assert response.status_code == 422

    def test_accepting_an_item_drops_the_reason_it_was_rejected_for(
        self, client, auth_headers, db_session, test_user
    ):
        # Left behind, it would keep teaching the profile something nobody said.
        item_id = self.make_item(db_session, test_user.id, "api-flip")
        client.patch(
            f"{BASE}/{item_id}",
            headers=auth_headers,
            json={"status": "rejected", "rejection_reason": "off_topic"},
        )
        response = client.patch(
            f"{BASE}/{item_id}", headers=auth_headers, json={"status": "accepted"}
        )
        assert response.json()["rejection_reason"] is None

    def test_the_bulk_route_is_reachable(
        self, client, auth_headers, db_session, test_user
    ):
        # PATCH /bulk is registered after PATCH /{item_id}; if the parameterised
        # route shadows it, every bulk triage fails as "Invalid item id".
        item_id = self.make_item(db_session, test_user.id, "api-bulk")
        response = client.patch(
            f"{BASE}/bulk",
            headers=auth_headers,
            json={
                "item_ids": [str(item_id)],
                "status": "rejected",
                "rejection_reason": "off_topic",
            },
        )
        assert response.status_code != 400, response.json()
        assert response.status_code == 200
