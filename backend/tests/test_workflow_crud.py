"""Workflow create, read, update and delete through the API.

Covers the routes nothing tested: in particular that an update answers with
the graph it saved, not the one it replaced.
"""

import pytest

pytestmark = pytest.mark.unit


def _node(node_id, node_type="tool", **config):
    return {
        "node_id": node_id,
        "node_type": node_type,
        "builtin_tool": "search_documents" if node_type == "tool" else None,
        "config": config,
        "position_x": 0,
        "position_y": 0,
    }


def _create(client, auth_headers, name="wf"):
    response = client.post(
        "/api/v1/workflows",
        headers=auth_headers,
        json={
            "name": name,
            "nodes": [_node("start", "start"), _node("a")],
            "edges": [{"source_node_id": "start", "target_node_id": "a"}],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_create_answers_with_the_saved_graph(client, auth_headers, test_user):
    workflow = _create(client, auth_headers)
    assert workflow["user_id"] == str(test_user.id)
    assert sorted(n["node_id"] for n in workflow["nodes"]) == ["a", "start"]
    assert len(workflow["edges"]) == 1


def test_an_update_answers_with_the_graph_it_saved(client, auth_headers):
    workflow = _create(client, auth_headers)

    response = client.put(
        f"/api/v1/workflows/{workflow['id']}",
        headers=auth_headers,
        json={
            "name": "renamed",
            "nodes": [_node("start", "start"), _node("b"), _node("c")],
            "edges": [
                {"source_node_id": "start", "target_node_id": "b"},
                {"source_node_id": "b", "target_node_id": "c"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["name"] == "renamed"
    assert sorted(n["node_id"] for n in updated["nodes"]) == ["b", "c", "start"]
    assert len(updated["edges"]) == 2
    # And a fresh read agrees with what the update answered.
    again = client.get(
        f"/api/v1/workflows/{workflow['id']}", headers=auth_headers
    ).json()
    assert sorted(n["node_id"] for n in again["nodes"]) == ["b", "c", "start"]


def test_an_update_without_a_graph_keeps_the_graph(client, auth_headers):
    workflow = _create(client, auth_headers)
    response = client.put(
        f"/api/v1/workflows/{workflow['id']}",
        headers=auth_headers,
        json={"description": "only the description"},
    )
    assert response.status_code == 200, response.text
    assert sorted(n["node_id"] for n in response.json()["nodes"]) == ["a", "start"]


def test_another_users_workflow_is_not_found(client, auth_headers, admin_headers):
    workflow = _create(client, auth_headers)
    for method in ("get", "put", "delete"):
        kwargs = {"json": {"name": "x"}} if method == "put" else {}
        response = getattr(client, method)(
            f"/api/v1/workflows/{workflow['id']}", headers=admin_headers, **kwargs
        )
        assert response.status_code == 404, (method, response.text)


def test_delete_removes_it(client, auth_headers):
    workflow = _create(client, auth_headers)
    assert (
        client.delete(
            f"/api/v1/workflows/{workflow['id']}", headers=auth_headers
        ).status_code
        == 204
    )
    assert (
        client.get(
            f"/api/v1/workflows/{workflow['id']}", headers=auth_headers
        ).status_code
        == 404
    )


def test_list_counts_and_filters(client, auth_headers):
    _create(client, auth_headers, name="one")
    _create(client, auth_headers, name="two")
    listing = client.get("/api/v1/workflows", headers=auth_headers).json()
    assert listing["total"] == 2
    assert {w["name"] for w in listing["workflows"]} == {"one", "two"}
    assert all(w["node_count"] == 2 for w in listing["workflows"])
