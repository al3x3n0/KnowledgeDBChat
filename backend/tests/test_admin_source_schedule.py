"""The admin page's next-run times for auto-synced sources.

`last_sync` is TIMESTAMP WITH TIME ZONE and Postgres returns it aware; the
endpoint compared it with a naive `utcnow()`, so the interval branch raised
and every next run came back None. A naive ISO string is also read by the
browser as local time, so the times are returned with their zone.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.document import DocumentSource

pytestmark = pytest.mark.unit


async def _source(db, config, last_sync=None):
    source = DocumentSource(
        name=f"src-{len(str(config))}-{last_sync}",
        source_type="web",
        config=config,
        is_active=True,
        last_sync=last_sync,
    )
    db.add(source)
    await db.commit()
    return source


@pytest.mark.parametrize(
    "last_sync",
    [
        datetime.now(timezone.utc) - timedelta(minutes=5),  # as Postgres returns it
        datetime.utcnow() - timedelta(minutes=5),  # as SQLite returns it
        None,
    ],
)
async def test_an_interval_source_has_a_zoned_next_run(
    client, admin_headers, db_session, last_sync
):
    source = await _source(
        db_session, {"auto_sync": True, "sync_interval_minutes": 60}, last_sync
    )

    items = client.get("/api/v1/admin/sources/next-run", headers=admin_headers).json()[
        "items"
    ]

    (mine,) = [i for i in items if i["source_id"] == str(source.id)]
    next_run = datetime.fromisoformat(mine["next_run"])
    assert next_run.tzinfo is not None
    assert next_run > datetime.now(timezone.utc)


async def test_a_cron_source_has_a_zoned_next_run(client, admin_headers, db_session):
    source = await _source(db_session, {"auto_sync": True, "cron": "0 * * * *"})

    items = client.get("/api/v1/admin/sources/next-run", headers=admin_headers).json()[
        "items"
    ]

    (mine,) = [i for i in items if i["source_id"] == str(source.id)]
    assert datetime.fromisoformat(mine["next_run"]).tzinfo is not None


def test_validate_cron_returns_a_zoned_time(client, admin_headers):
    body = client.post(
        "/api/v1/admin/validate-cron",
        params={"cron": "*/5 * * * *"},
        headers=admin_headers,
    ).json()
    assert body["valid"] is True
    assert datetime.fromisoformat(body["next_run"]).tzinfo is not None
