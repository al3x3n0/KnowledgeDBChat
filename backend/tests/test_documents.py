"""
Tests for document management endpoints.
"""

import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from io import BytesIO


def test_get_documents(client: TestClient, auth_headers):
    """Test getting list of documents."""
    response = client.get(
        "/api/v1/documents/",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_documents_with_pagination(client: TestClient, auth_headers):
    """Test getting documents with pagination."""
    response = client.get(
        "/api/v1/documents/?skip=0&limit=10",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10


def test_get_documents_with_search(client: TestClient, auth_headers):
    """Test getting documents with search query."""
    response = client.get(
        "/api/v1/documents/?search=test",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_document_not_found(client: TestClient, auth_headers):
    """Test getting a non-existent document."""
    fake_id = str(uuid4())
    response = client.get(
        f"/api/v1/documents/{fake_id}",
        headers=auth_headers
    )
    
    assert response.status_code == 404


def test_upload_document(client: TestClient, auth_headers):
    """Test uploading a document."""
    file_content = b"This is a test document content"
    files = {
        "file": ("test.txt", BytesIO(file_content), "text/plain")
    }
    data = {
        "title": "Test Document",
        "tags": "test, sample"
    }
    
    response = client.post(
        "/api/v1/documents/upload",
        files=files,
        data=data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    response_data = response.json()
    assert "document_id" in response_data
    assert "message" in response_data


def test_upload_document_invalid_file_type(client: TestClient, auth_headers):
    """Test uploading an invalid file type."""
    file_content = b"Invalid content"
    files = {
        "file": ("test.exe", BytesIO(file_content), "application/x-msdownload")
    }
    
    response = client.post(
        "/api/v1/documents/upload",
        files=files,
        headers=auth_headers
    )
    
    # Should fail validation
    assert response.status_code in [400, 422]


def test_upload_document_unauthorized(client: TestClient):
    """Test uploading a document without authentication."""
    file_content = b"Test content"
    files = {
        "file": ("test.txt", BytesIO(file_content), "text/plain")
    }
    
    response = client.post(
        "/api/v1/documents/upload",
        files=files
    )
    
    assert response.status_code == 401


def test_get_document_sources(client: TestClient, auth_headers):
    """Test getting document sources."""
    response = client.get(
        "/api/v1/documents/sources/",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_create_document_source_admin(client: TestClient, admin_headers):
    """Test creating a document source (admin only)."""
    response = client.post(
        "/api/v1/documents/sources/",
        json={
            "name": "Test Source",
            "source_type": "web",
            "config": {"url": "https://example.com"}
        },
        headers=admin_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Source"
    assert data["source_type"] == "web"


def test_create_document_source_non_admin(client: TestClient, auth_headers):
    """Test creating a document source as non-admin (should fail)."""
    response = client.post(
        "/api/v1/documents/sources/",
        json={
            "name": "Test Source",
            "source_type": "web",
            "config": {"url": "https://example.com"}
        },
        headers=auth_headers
    )
    
    assert response.status_code == 403


def test_sync_document_source(client: TestClient, admin_headers):
    """Test syncing a document source."""
    # First create a source
    create_response = client.post(
        "/api/v1/documents/sources/",
        json={
            "name": "Test Source",
            "source_type": "web",
            "config": {"url": "https://example.com"}
        },
        headers=admin_headers
    )
    source_id = create_response.json()["id"]
    
    # Sync the source
    response = client.post(
        f"/api/v1/documents/sources/{source_id}/sync",
        headers=admin_headers
    )
    
    assert response.status_code == 200


def test_delete_document_source(client: TestClient, admin_headers):
    """Test deleting a document source."""
    # First create a source
    create_response = client.post(
        "/api/v1/documents/sources/",
        json={
            "name": "Test Source",
            "source_type": "web",
            "config": {"url": "https://example.com"}
        },
        headers=admin_headers
    )
    source_id = create_response.json()["id"]
    
    # Delete the source
    response = client.delete(
        f"/api/v1/documents/sources/{source_id}",
        headers=admin_headers
    )
    
    assert response.status_code == 200

