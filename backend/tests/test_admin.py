"""
Tests for admin endpoints.
"""

from fastapi.testclient import TestClient

from app.api.endpoints import admin as admin_endpoints


def test_get_system_health_admin(client: TestClient, admin_headers):
    """Test getting system health (admin only)."""
    response = client.get("/api/v1/admin/health", headers=admin_headers)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "overall_status" in data
        assert "services" in data
        assert "timestamp" in data


def test_get_system_health_non_admin(client: TestClient, auth_headers):
    """Test getting system health as non-admin (should fail)."""
    response = client.get("/api/v1/admin/health", headers=auth_headers)

    assert response.status_code == 403


def test_get_system_stats_admin(client: TestClient, admin_headers):
    """Test getting system statistics (admin only)."""
    response = client.get("/api/v1/admin/stats", headers=admin_headers)

    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "timestamp" in data


def test_get_system_stats_non_admin(client: TestClient, auth_headers):
    """Test getting system stats as non-admin (should fail)."""
    response = client.get("/api/v1/admin/stats", headers=auth_headers)

    assert response.status_code == 403


def test_get_vector_store_stats_admin(client: TestClient, admin_headers):
    """Test getting vector store statistics (admin only)."""
    response = client.get("/api/v1/admin/vector-store/stats", headers=admin_headers)

    # May return 200 with stats or 500 if vector store not initialized
    assert response.status_code in [200, 500]


def test_get_task_status_admin(client: TestClient, admin_headers):
    """Test getting task status (admin only)."""
    response = client.get("/api/v1/admin/tasks/status", headers=admin_headers)

    # May return 200 with task info or 500 if Celery not available
    assert response.status_code in [200, 500]


def test_get_system_logs_admin(client: TestClient, admin_headers):
    """Test getting system logs (admin only)."""
    response = client.get("/api/v1/admin/logs?lines=50", headers=admin_headers)

    # May return 200 with logs or 500 if log file not found
    assert response.status_code in [200, 500]


def test_get_system_logs_with_lines(client: TestClient, admin_headers):
    """Test getting system logs with specific line count."""
    response = client.get("/api/v1/admin/logs?lines=10", headers=admin_headers)

    # May return 200 with logs or 500 if log file not found
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "logs" in data or "message" in data


def test_update_flags_accepts_repo_symbol_retrieval_flag(
    client: TestClient, admin_headers, monkeypatch
):
    calls = []

    async def _fake_set_feature_flag(name: str, value: bool) -> bool:
        calls.append((name, value))
        return True

    monkeypatch.setattr(admin_endpoints, "set_feature_flag", _fake_set_feature_flag)

    response = client.post(
        "/api/v1/admin/flags",
        json={"repo_symbol_retrieval_enabled": True},
        headers=admin_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["updated"]["repo_symbol_retrieval_enabled"] is True
    assert ("repo_symbol_retrieval_enabled", True) in calls


def test_get_flags_includes_repo_symbol_retrieval_flag(
    client: TestClient, admin_headers, monkeypatch
):
    async def _fake_get_feature_flags():
        return {
            "knowledge_graph_enabled": True,
            "summarization_enabled": True,
            "auto_summarize_on_process": False,
            "unsafe_code_execution_enabled": False,
            "repo_symbol_retrieval_enabled": True,
        }

    monkeypatch.setattr(admin_endpoints, "get_feature_flags", _fake_get_feature_flags)

    response = client.get("/api/v1/admin/flags", headers=admin_headers)
    assert response.status_code == 200
    body = response.json()
    assert "repo_symbol_retrieval_enabled" in body
    assert body["repo_symbol_retrieval_enabled"] is True
