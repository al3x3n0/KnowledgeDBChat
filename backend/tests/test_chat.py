"""
Tests for chat endpoints.
"""

import pytest
from uuid import uuid4
from fastapi.testclient import TestClient


def test_create_chat_session(client: TestClient, auth_headers):
    """Test creating a new chat session."""
    response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test Chat Session"},
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["title"] == "Test Chat Session"
    assert data["is_active"] is True


def test_create_chat_session_no_title(client: TestClient, auth_headers):
    """Test creating a chat session without title (auto-generated)."""
    response = client.post(
        "/api/v1/chat/sessions",
        json={},
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["title"] is not None


def test_create_chat_session_unauthorized(client: TestClient):
    """Test creating a chat session without authentication."""
    response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test"}
    )
    
    assert response.status_code == 401


def test_get_chat_sessions(client: TestClient, auth_headers):
    """Test getting all chat sessions for a user."""
    # Create a session first
    create_response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test Session"},
        headers=auth_headers
    )
    assert create_response.status_code == 200
    
    # Get all sessions
    response = client.get(
        "/api/v1/chat/sessions",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_get_chat_session(client: TestClient, auth_headers):
    """Test getting a specific chat session."""
    # Create a session first
    create_response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test Session"},
        headers=auth_headers
    )
    session_id = create_response.json()["id"]
    
    # Get the session
    response = client.get(
        f"/api/v1/chat/sessions/{session_id}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == session_id
    assert "messages" in data


def test_get_chat_session_not_found(client: TestClient, auth_headers):
    """Test getting a non-existent chat session."""
    fake_id = str(uuid4())
    response = client.get(
        f"/api/v1/chat/sessions/{fake_id}",
        headers=auth_headers
    )
    
    assert response.status_code == 404


def test_get_chat_session_unauthorized(client: TestClient):
    """Test getting a chat session without authentication."""
    fake_id = str(uuid4())
    response = client.get(f"/api/v1/chat/sessions/{fake_id}")
    
    assert response.status_code == 401


def test_send_message(client: TestClient, auth_headers):
    """Test sending a message in a chat session."""
    # Create a session first
    create_response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test Session"},
        headers=auth_headers
    )
    session_id = create_response.json()["id"]
    
    # Send a message
    response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"content": "Hello, this is a test message"},
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "Hello, this is a test message"
    assert data["role"] == "assistant"  # AI response


def test_send_message_empty_content(client: TestClient, auth_headers):
    """Test sending a message with empty content."""
    # Create a session first
    create_response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test Session"},
        headers=auth_headers
    )
    session_id = create_response.json()["id"]
    
    # Try to send empty message
    response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"content": ""},
        headers=auth_headers
    )
    
    assert response.status_code == 422  # Validation error


def test_send_message_invalid_session(client: TestClient, auth_headers):
    """Test sending a message to a non-existent session."""
    fake_id = str(uuid4())
    response = client.post(
        f"/api/v1/chat/sessions/{fake_id}/messages",
        json={"content": "Test message"},
        headers=auth_headers
    )
    
    assert response.status_code == 404


def test_delete_chat_session(client: TestClient, auth_headers):
    """Test deleting a chat session."""
    # Create a session first
    create_response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test Session"},
        headers=auth_headers
    )
    session_id = create_response.json()["id"]
    
    # Delete the session
    response = client.delete(
        f"/api/v1/chat/sessions/{session_id}",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    
    # Verify it's deleted
    get_response = client.get(
        f"/api/v1/chat/sessions/{session_id}",
        headers=auth_headers
    )
    assert get_response.status_code == 404


def test_delete_chat_session_not_found(client: TestClient, auth_headers):
    """Test deleting a non-existent chat session."""
    fake_id = str(uuid4())
    response = client.delete(
        f"/api/v1/chat/sessions/{fake_id}",
        headers=auth_headers
    )
    
    assert response.status_code == 404


def test_provide_feedback(client: TestClient, auth_headers):
    """Test providing feedback on a chat message."""
    # Create a session and send a message first
    create_response = client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test Session"},
        headers=auth_headers
    )
    session_id = create_response.json()["id"]
    
    send_response = client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"content": "Test message"},
        headers=auth_headers
    )
    message_id = send_response.json()["id"]
    
    # Provide feedback
    response = client.put(
        f"/api/v1/chat/messages/{message_id}/feedback",
        params={"rating": 5, "feedback": "Great response!"},
        headers=auth_headers
    )
    
    assert response.status_code == 200


def test_provide_feedback_invalid_rating(client: TestClient, auth_headers):
    """Test providing feedback with invalid rating."""
    fake_id = str(uuid4())
    response = client.put(
        f"/api/v1/chat/messages/{fake_id}/feedback",
        params={"rating": 10},  # Invalid rating (should be 1-5)
        headers=auth_headers
    )
    
    # Should fail validation or return 404
    assert response.status_code in [400, 404, 422]

