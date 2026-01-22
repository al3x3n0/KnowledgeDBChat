"""
Tests for persona management endpoints.
"""

from uuid import UUID
from fastapi.testclient import TestClient


def test_admin_can_create_persona(client: TestClient, admin_headers: dict, auth_headers: dict):
    payload = {
        "name": "Alice",
        "platform_id": "user-1",
        "description": "Primary spokesperson",
        "is_active": True,
    }

    create_resp = client.post("/api/v1/personas/", json=payload, headers=admin_headers)
    assert create_resp.status_code == 201, create_resp.text
    persona = create_resp.json()
    assert persona["name"] == "Alice"
    assert persona["platform_id"] == "user-1"
    assert persona["id"]

    list_resp = client.get("/api/v1/personas/", headers=auth_headers)
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert body["total"] == 1
    assert body["items"][0]["name"] == "Alice"


def test_non_admin_cannot_create_persona(client: TestClient, auth_headers: dict):
    resp = client.post(
        "/api/v1/personas/",
        json={"name": "Bob"},
        headers=auth_headers,
    )
    assert resp.status_code == 403


def test_update_and_delete_persona(client: TestClient, admin_headers: dict):
    create_resp = client.post("/api/v1/personas/", json={"name": "Temp"}, headers=admin_headers)
    persona_id = create_resp.json()["id"]
    assert UUID(persona_id)

    update_resp = client.put(
        f"/api/v1/personas/{persona_id}",
        json={"description": "Updated", "name": "Temp 2"},
        headers=admin_headers,
    )
    assert update_resp.status_code == 200
    updated = update_resp.json()
    assert updated["description"] == "Updated"
    assert updated["name"] == "Temp 2"

    delete_resp = client.delete(f"/api/v1/personas/{persona_id}", headers=admin_headers)
    assert delete_resp.status_code == 204

    get_resp = client.get(f"/api/v1/personas/{persona_id}", headers=admin_headers)
    assert get_resp.status_code == 404
