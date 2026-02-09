"""
Research monitor profile service.

Builds and persists lightweight token-score profiles from Research Inbox triage.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any, Optional, Tuple

from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from app.models.research_inbox import ResearchInboxItem
from app.models.research_monitor_profile import ResearchMonitorProfile


class ResearchMonitorProfileService:
    STOPWORDS = {
        "the","and","for","with","from","that","this","into","over","under","when","where","what","which","while",
        "your","you","are","our","their","they","them","then","than","also","only","just","more","most","less",
        "use","using","used","make","made","help","helps","via","can","could","should","would","may","might","will",
        "data","dataset","datasets","model","models","train","training","eval","evaluate","evaluation","assistant",
        "job","jobs","paper","papers","doc","docs","document","documents","research","monitor",
    }

    def tokenize(self, text: str) -> list[str]:
        raw = re.findall(r"[a-zA-Z0-9_\\-]+", (text or "").lower())
        out: list[str] = []
        for w in raw:
            w = w.strip("_-")
            if len(w) < 3:
                continue
            if w in self.STOPWORDS:
                continue
            out.append(w)
        return out

    async def recompute_profile(
        self,
        *,
        db: AsyncSession,
        user_id,
        customer: Optional[str],
        limit: int = 500,
    ) -> ResearchMonitorProfile:
        """
        Recompute token scores from accepted/rejected inbox items and upsert the profile.
        """
        stmt = (
            select(ResearchInboxItem.status, ResearchInboxItem.title, ResearchInboxItem.summary)
            .where(
                ResearchInboxItem.user_id == user_id,
                ResearchInboxItem.status.in_(["accepted", "rejected"]),
            )
            .order_by(ResearchInboxItem.updated_at.desc())
            .limit(int(max(50, min(limit, 2000))))
        )
        if customer:
            stmt = stmt.where(ResearchInboxItem.customer == customer)
        else:
            stmt = stmt.where(ResearchInboxItem.customer.is_(None))

        res = await db.execute(stmt)
        rows = res.all()

        pos = Counter()
        neg = Counter()
        for status, title, summary in rows:
            text = f"{title or ''} {summary or ''}"
            toks = self.tokenize(text)
            if not toks:
                continue
            if str(status) == "accepted":
                pos.update(toks)
            elif str(status) == "rejected":
                neg.update(toks)

        scores: dict[str, int] = {}
        for t, c in pos.items():
            scores[t] = scores.get(t, 0) + int(c)
        for t, c in neg.items():
            scores[t] = scores.get(t, 0) - int(c)

        # Keep only meaningful tokens to bound JSON size.
        pruned = {t: int(s) for t, s in scores.items() if abs(int(s)) >= 2}

        existing_res = await db.execute(
            select(ResearchMonitorProfile).where(
                and_(
                    ResearchMonitorProfile.user_id == user_id,
                    (ResearchMonitorProfile.customer == customer) if customer else ResearchMonitorProfile.customer.is_(None),
                )
            ).limit(1)
        )
        profile = existing_res.scalar_one_or_none()
        if profile:
            profile.token_scores = pruned
            profile.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(profile)
            return profile

        profile = ResearchMonitorProfile(
            user_id=user_id,
            customer=customer,
            token_scores=pruned,
            muted_tokens=[],
            muted_patterns=[],
            notes=None,
        )
        db.add(profile)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            # Race; load and update.
            existing_res = await db.execute(
                select(ResearchMonitorProfile).where(
                    and_(
                        ResearchMonitorProfile.user_id == user_id,
                        (ResearchMonitorProfile.customer == customer) if customer else ResearchMonitorProfile.customer.is_(None),
                    )
                ).limit(1)
            )
            profile = existing_res.scalar_one()
            profile.token_scores = pruned
            profile.updated_at = datetime.utcnow()
            await db.commit()
        await db.refresh(profile)
        return profile

    async def get_profile(
        self,
        *,
        db: AsyncSession,
        user_id,
        customer: Optional[str],
    ) -> Optional[ResearchMonitorProfile]:
        stmt = select(ResearchMonitorProfile).where(ResearchMonitorProfile.user_id == user_id)
        if customer:
            stmt = stmt.where(ResearchMonitorProfile.customer == customer)
        else:
            stmt = stmt.where(ResearchMonitorProfile.customer.is_(None))
        res = await db.execute(stmt.limit(1))
        return res.scalar_one_or_none()


research_monitor_profile_service = ResearchMonitorProfileService()

