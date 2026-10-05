"""An admin can create an agent definition through the API.

The handler passed `routing_defaults` to a validator that takes no such
argument, so every create raised TypeError and answered 500 -- and the
routing defaults a caller sent were never stored either way.
"""

import pytest

pytestmark = pytest.mark.unit

PAYLOAD = {
    "name": "cache_expert",
    "display_name": "Cache Expert",
    "description": "Knows caches.",
    "system_prompt": "You answer questions about cache hierarchies.",
    "capabilities": ["general"],
    "tool_whitelist": ["search_documents"],
    "routing_defaults": {"tier": "deep"},
}


def test_an_admin_creates_an_agent(client, admin_headers):
    response = client.post("/api/v1/agent/agents", json=PAYLOAD, headers=admin_headers)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "cache_expert"
    assert body["is_system"] is False
    assert body["routing_defaults"] == {"tier": "deep"}


def test_an_unknown_tool_is_refused_by_name(client, admin_headers):
    response = client.post(
        "/api/v1/agent/agents",
        json={**PAYLOAD, "tool_whitelist": ["no_such_tool"]},
        headers=admin_headers,
    )

    assert response.status_code == 400
    assert "no_such_tool" in response.json()["detail"]


def test_a_regular_user_cannot(client, auth_headers):
    response = client.post("/api/v1/agent/agents", json=PAYLOAD, headers=auth_headers)

    assert response.status_code == 403
