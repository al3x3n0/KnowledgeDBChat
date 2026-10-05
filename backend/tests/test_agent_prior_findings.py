"""Recalling what earlier runs measured.

`agent_prior_findings` had no tests. It is the one path by which a run learns
what the corpus already knows, so what it declines to return matters as much
as what it returns.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


class TestAWithdrawnFindingIsNotRecalled:
    """A retracted finding is worse than an absent one.

    It arrives with a job id and a measurement source, reads as established,
    and the run citing it has no way to know it was taken back. Every
    retraction in this corpus was recorded because the number was wrong -- an
    inert prefetcher reported as a 0.78x regression, a sweep reporting a
    saturation it never measured -- and those are exactly the claims a later
    run would most want to reuse.
    """

    def test_the_ref_format_matches_what_retraction_records(self):
        """`<job_id>#<index>` on both sides, or the filter matches nothing."""
        from app.services import agent_retraction_service

        assert agent_retraction_service.finding_ref("abc", 3) == "abc#3"

    async def test_a_retracted_finding_is_skipped_and_counted(self, monkeypatch):
        from app.services import agent_prior_findings as pf

        job = type(
            "J",
            (),
            {
                "id": "job1",
                "job_type": "research",
                "created_at": None,
                "results": {
                    "findings": [
                        {"type": "mechanism_evaluation", "title": "withdrawn claim"},
                        {"type": "mechanism_evaluation", "title": "sound claim"},
                    ]
                },
            },
        )()

        async def fake_refs(db, user_id):
            return {"job1#0"}

        monkeypatch.setattr(pf, "_retracted_finding_refs", fake_refs)
        out = await pf.recall(
            db=_FakeDb([job]), user_id="u", finding_types=["mechanism_evaluation"]
        )
        titles = [f.get("title") for f in out["findings"]]
        assert titles == ["sound claim"]
        assert out["retracted_skipped"] == 1

    async def test_an_unreachable_retraction_table_does_not_empty_the_corpus(self):
        """Losing the corpus is a worse failure than surfacing one bad number,
        so the helper swallows its own errors and returns nothing withdrawn."""
        from app.services import agent_prior_findings as pf

        assert await pf._retracted_finding_refs(_BoomDb(), "u") == set()


class _FakeDb:
    def __init__(self, jobs):
        self._jobs = jobs

    async def execute(self, *_a, **_k):
        jobs = self._jobs

        class R:
            def scalars(self_inner):
                class S:
                    def all(self_s):
                        return jobs

                return S()

        return R()


class _BoomDb:
    async def execute(self, *_a, **_k):
        raise RuntimeError("unreachable")
