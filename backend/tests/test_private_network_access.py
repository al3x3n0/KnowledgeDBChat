"""Who may reach a private address: decided once, for both scrapers."""

import pytest

from app.models.document import DocumentSource
from app.services.web_scraper_service import (
    PRIVATE_NETWORK_REFUSAL,
    private_network_access,
)

pytestmark = pytest.mark.unit


@pytest.fixture
async def wiki(db_session):
    db_session.add(
        DocumentSource(
            name="Internal wiki",
            source_type="web",
            is_active=True,
            config={"base_urls": ["http://wiki.corp/start"]},
        )
    )
    await db_session.commit()


async def test_not_asking_allows_only_the_named_hosts(db_session, wiki):
    assert await private_network_access(
        db_session, "http://elsewhere/", asked=False, admin=False
    ) == (False, ["wiki.corp"], None)


async def test_an_admin_who_asks_may_reach_any_private_address(db_session, wiki):
    allow, hosts, refusal = await private_network_access(
        db_session, "http://10.0.0.5/", asked=True, admin=True
    )
    assert (allow, refusal) == (True, None)


async def test_anyone_may_ask_for_a_host_a_web_source_names(db_session, wiki):
    # Allowed per host, not wholesale: a link off the wiki stays blocked.
    assert await private_network_access(
        db_session, "http://docs.wiki.corp/page", asked=True, admin=False
    ) == (False, ["wiki.corp"], None)


async def test_a_non_admin_asking_for_any_other_host_is_refused(db_session, wiki):
    assert await private_network_access(
        db_session, "http://10.0.0.5/", asked=True, admin=False
    ) == (False, ["wiki.corp"], PRIVATE_NETWORK_REFUSAL)
