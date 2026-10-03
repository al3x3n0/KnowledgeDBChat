"""The admin routes that curate AI Hub eval templates and dataset presets.

Two pairs of routes were copies of each other; both now go through one getter
and one setter. Only the feature-flag store is replaced.
"""

import pytest

from app.api.endpoints import admin as admin_endpoint

pytestmark = pytest.mark.unit

ROUTES = {
    "/api/v1/admin/ai-hub/evals/enabled": "ai_hub_enabled_eval_templates",
    "/api/v1/admin/ai-hub/datasets/presets/enabled": "ai_hub_enabled_dataset_presets",
}


@pytest.fixture
def flags(monkeypatch):
    store = {}

    async def get_feature_str(key):
        return store.get(key)

    async def set_feature_str(key, value):
        store[key] = value
        return True

    monkeypatch.setattr(admin_endpoint, "get_feature_str", get_feature_str)
    monkeypatch.setattr(admin_endpoint, "set_feature_str", set_feature_str)
    return store


@pytest.mark.parametrize("path,flag", ROUTES.items())
def test_a_list_is_stored_as_csv_and_read_back(
    client, admin_headers, flags, path, flag
):
    response = client.post(
        path, json={"enabled": [" a ", "", None, "b"]}, headers=admin_headers
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "enabled": ["a", "b"]}
    assert flags[flag] == "a,b"
    assert client.get(path, headers=admin_headers).json() == {
        "enabled": ["a", "b"],
        "raw": "a,b",
    }


@pytest.mark.parametrize("path", ROUTES)
def test_raw_csv_is_cleaned(client, admin_headers, flags, path):
    response = client.post(path, json={"raw": "a, ,b,"}, headers=admin_headers)
    assert response.json()["enabled"] == ["a", "b"]


@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize(
    "payload,detail",
    [({}, "Missing 'enabled' or 'raw'"), ({"enabled": "a"}, "Invalid payload")],
)
def test_a_payload_it_cannot_read_is_refused(
    client, admin_headers, flags, path, payload, detail
):
    response = client.post(path, json=payload, headers=admin_headers)
    assert response.status_code == 400
    assert response.json()["detail"] == detail


@pytest.mark.parametrize("path", ROUTES)
def test_only_an_admin_may_set_it(client, auth_headers, flags, path):
    response = client.post(path, json={"enabled": ["a"]}, headers=auth_headers)
    assert response.status_code == 403
    assert flags == {}
