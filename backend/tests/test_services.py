"""
Tests for core services.
"""

import pytest
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.document_service import DocumentService
from app.services.auth_service import AuthService
from app.models.document import Document, DocumentSource
from app.models.user import User


@pytest.mark.asyncio
async def test_document_service_get_documents(db_session: AsyncSession):
    """Test DocumentService.get_documents."""
    service = DocumentService()
    
    documents = await service.get_documents(
        skip=0,
        limit=10,
        db=db_session
    )
    
    assert isinstance(documents, list)


@pytest.mark.asyncio
async def test_document_service_get_document_not_found(db_session: AsyncSession):
    """Test DocumentService.get_document with non-existent ID."""
    service = DocumentService()
    
    fake_id = uuid4()
    document = await service.get_document(fake_id, db_session)
    
    assert document is None


@pytest.mark.asyncio
async def test_auth_service_create_user(db_session: AsyncSession):
    """Test AuthService.create_user."""
    service = AuthService()
    
    user = await service.create_user(
        username="testuser2",
        email="testuser2@example.com",
        password="password123",
        full_name="Test User 2",
        db=db_session
    )
    
    assert user is not None
    assert user.username == "testuser2"
    assert user.email == "testuser2@example.com"
    assert user.hashed_password != "password123"  # Should be hashed


@pytest.mark.asyncio
async def test_auth_service_authenticate_user(db_session: AsyncSession, test_user: User):
    """Test AuthService.authenticate_user."""
    service = AuthService()
    
    authenticated_user = await service.authenticate_user(
        username="testuser",
        password="testpassword123",
        db=db_session
    )
    
    assert authenticated_user is not None
    assert authenticated_user.id == test_user.id


@pytest.mark.asyncio
async def test_auth_service_authenticate_user_wrong_password(db_session: AsyncSession, test_user: User):
    """Test AuthService.authenticate_user with wrong password."""
    service = AuthService()
    
    authenticated_user = await service.authenticate_user(
        username="testuser",
        password="wrongpassword",
        db=db_session
    )
    
    assert authenticated_user is None


@pytest.mark.asyncio
async def test_auth_service_get_user_by_username(db_session: AsyncSession, test_user: User):
    """Test AuthService.get_user_by_username."""
    service = AuthService()
    
    user = await service.get_user_by_username("testuser", db_session)
    
    assert user is not None
    assert user.id == test_user.id


@pytest.mark.asyncio
async def test_auth_service_get_user_by_id(db_session: AsyncSession, test_user: User):
    """Test AuthService.get_user_by_id."""
    service = AuthService()
    
    user = await service.get_user_by_id(str(test_user.id), db_session)
    
    assert user is not None
    assert user.id == test_user.id

