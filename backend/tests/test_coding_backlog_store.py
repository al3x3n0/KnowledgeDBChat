"""Which coding backlog items a person is shown, and how many are read.

The list route asked the database for every item shared with anyone, decided
in Python which this caller could see, filtered in Python and sliced a page
out. These run the route itself, since the page, the total and the rule all
have to agree there.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import event

from app.api.endpoints import coding_backlog
from app.models.coding_backlog import CodingBacklogItem
from app.modules.coding_backlog.application import backlog_store
from tests.conftest import test_engine


def _item(owner, n: int, **fields) -> CodingBacklogItem:
    return CodingBacklogItem(
        id=uuid4(),
        user_id=owner,
        title=f"item {n:02d}",
        portfolio_goal="fix it",
        status=fields.pop("status", "draft"),
        updated_at=datetime(2026, 9, 1) + timedelta(minutes=n),
        **fields,
    )


async def _list(db, user, **overrides):
    params = dict(
        status_filter=None,
        visibility_scope="mine",
        assigned_user_id=None,
        limit=50,
        offset=0,
        current_user=user,
        db=db,
    )
    params.update(overrides)
    return await coding_backlog.list_coding_backlog_items(**params)


def _titles(listed):
    return [item.title for item in listed.items]


class _ItemSelects:
    def __enter__(self):
        self.statements = []

        def record(conn, cursor, statement, parameters, context, executemany):
            text = " ".join(statement.split()).lower()
            if text.startswith("select") and "coding_backlog_items.title" in text:
                self.statements.append(text)

        self._record = record
        event.listen(test_engine.sync_engine, "before_cursor_execute", record)
        return self

    def __exit__(self, *exc):
        event.remove(test_engine.sync_engine, "before_cursor_execute", self._record)


@pytest.mark.asyncio
async def test_your_own_items_are_paged_by_the_database(db_session, test_user):
    db_session.add_all([_item(test_user.id, n) for n in range(25)])
    db_session.add_all([_item(uuid4(), 100 + n) for n in range(5)])
    await db_session.commit()

    with _ItemSelects() as seen:
        listed = await _list(db_session, test_user, limit=10, offset=10)

    assert _titles(listed) == [f"item {n:02d}" for n in range(14, 4, -1)]
    assert listed.total == 25
    # The page itself is one limited query. (The other read of whole items
    # is the collaborator-name lookup, which is not this route's to change.)
    by_updated = [text for text in seen.statements if "order by" in text]
    assert len(by_updated) == 1 and " limit " in by_updated[0]


@pytest.mark.asyncio
async def test_shared_means_shared_with_you(db_session, test_user):
    owner = uuid4()
    db_session.add_all(
        [
            _item(
                owner, 1, visibility="shared", shared_with_user_ids=[str(test_user.id)]
            ),
            # Shared, but with somebody else: the list of people is what grants.
            _item(owner, 2, visibility="shared", shared_with_user_ids=[str(uuid4())]),
            _item(owner, 3, visibility="shared"),
            # Named in the list but not shared at all.
            _item(owner, 4, shared_with_user_ids=[str(test_user.id)]),
            _item(owner, 5, assigned_user_id=test_user.id),
            _item(owner, 6),
            _item(test_user.id, 7),
        ]
    )
    await db_session.commit()

    shared = await _list(db_session, test_user, visibility_scope="shared")
    everything = await _list(db_session, test_user, visibility_scope="all")
    mine = await _list(db_session, test_user)

    assert _titles(shared) == ["item 05", "item 01"]
    assert shared.total == 2
    assert _titles(everything) == ["item 07", "item 05", "item 01"]
    assert _titles(mine) == ["item 07"]


@pytest.mark.asyncio
async def test_filters_narrow_the_total_as_well_as_the_page(db_session, test_user):
    reviewer = uuid4()
    db_session.add_all(
        [
            _item(test_user.id, 1, status="active"),
            _item(test_user.id, 2, status="active", assigned_user_id=reviewer),
            _item(test_user.id, 3, assigned_user_id=reviewer),
            _item(test_user.id, 4),
        ]
    )
    await db_session.commit()

    active = await _list(db_session, test_user, status_filter="Active")
    assigned = await _list(db_session, test_user, assigned_user_id=reviewer)
    both = await _list(
        db_session, test_user, status_filter="active", assigned_user_id=reviewer
    )

    assert (_titles(active), active.total) == (["item 02", "item 01"], 2)
    assert (_titles(assigned), assigned.total) == (["item 03", "item 02"], 2)
    assert (_titles(both), both.total) == (["item 02"], 1)


@pytest.mark.asyncio
async def test_a_page_of_other_peoples_items_counts_only_what_you_may_see(
    db_session, test_user
):
    owner = uuid4()
    db_session.add_all(
        [
            _item(
                owner, n, visibility="shared", shared_with_user_ids=[str(test_user.id)]
            )
            for n in range(6)
        ]
        + [_item(owner, 50 + n, visibility="shared") for n in range(6)]
    )
    await db_session.commit()

    listed = await _list(
        db_session, test_user, visibility_scope="shared", limit=4, offset=4
    )

    assert _titles(listed) == ["item 01", "item 00"]
    assert listed.total == 6


@pytest.mark.asyncio
async def test_an_item_you_may_not_see_is_not_found(db_session, test_user):
    private = _item(uuid4(), 1)
    own = _item(test_user.id, 2)
    db_session.add_all([private, own])
    await db_session.commit()

    assert (
        await backlog_store.get_visible_item(db_session, own.id, test_user.id)
    ).id == own.id
    with pytest.raises(backlog_store.BacklogItemNotFound):
        await backlog_store.get_visible_item(db_session, private.id, test_user.id)
    with pytest.raises(backlog_store.BacklogItemNotFound):
        await backlog_store.get_visible_item(db_session, uuid4(), test_user.id)


@pytest.mark.asyncio
async def test_the_route_answers_404_for_one_you_may_not_see(db_session, test_user):
    from fastapi import HTTPException

    private = _item(uuid4(), 1)
    db_session.add(private)
    await db_session.commit()

    with pytest.raises(HTTPException) as refused:
        await coding_backlog._get_visible_backlog_item_or_404(
            db_session, private.id, test_user.id
        )

    assert refused.value.status_code == 404


@pytest.mark.asyncio
async def test_an_item_nobody_is_assigned_to_can_be_listed(db_session, test_user):
    # The attribute exists and is None, so `getattr(item, name, "")` returned
    # None and `str(None)` went to the response model as the id "None": every
    # unassigned item -- most of them -- made the whole list a 500.
    db_session.add(_item(test_user.id, 1))
    await db_session.commit()

    listed = await _list(db_session, test_user)

    summary = listed.items[0].collaboration_summary
    assert summary.assigned_user_id is None
    assert summary.assigned_by_user_id is None
