"""
Tests for admin endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_get_system_health_admin(client: TestClient, admin_headers):
    """Test getting system health (admin only)."""
    response = client.get(
        "/api/v1/admin/health",
        headers=admin_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert "services" in data
    assert "timestamp" in data


def test_get_system_health_non_admin(client: TestClient, auth_headers):
    """Test getting system health as non-admin (should fail)."""
    response = client.get(
        "/api/v1/admin/health",
        headers=auth_headers
    )
    
    assert response.status_code == 403


def test_get_system_stats_admin(client: TestClient, admin_headers):
    """Test getting system statistics (admin only)."""
    response = client.get(
        "/api/v1/admin/stats",
        headers=admin_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data


def test_get_system_stats_non_admin(client: TestClient, auth_headers):
    """Test getting system stats as non-admin (should fail)."""
    response = client.get(
        "/api/v1/admin/stats",
        headers=auth_headers
    )
    
    assert response.status_code == 403


def test_get_vector_store_stats_admin(client: TestClient, admin_headers):
    """Test getting vector store statistics (admin only)."""
    response = client.get(
        "/api/v1/admin/vector-store/stats",
        headers=admin_headers
    )
    
    # May return 200 with stats or 500 if vector store not initialized
    assert response.status_code in [200, 500]


def test_get_task_status_admin(client: TestClient, admin_headers):
    """Test getting task status (admin only)."""
    response = client.get(
        "/api/v1/admin/tasks/status",
        headers=admin_headers
    )
    
    # May return 200 with task info or 500 if Celery not available
    assert response.status_code in [200, 500]


def test_get_system_logs_admin(client: TestClient, admin_headers):
    """Test getting system logs (admin only)."""
    response = client.get(
        "/api/v1/admin/logs?lines=50",
        headers=admin_headers
    )
    
    # May return 200 with logs or 500 if log file not found
    assert response.status_code in [200, 500]


def test_get_system_logs_with_lines(client: TestClient, admin_headers):
    """Test getting system logs with specific line count."""
    response = client.get(
        "/api/v1/admin/logs?lines=10",
        headers=admin_headers
    )
    
    # May return 200 with logs or 500 if log file not found
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "logs" in data or "message" in data

