"""
Test factories for creating test data.
"""

from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.models.document import Document, DocumentSource
from app.models.chat import ChatSession, ChatMessage
from app.services.auth_service import AuthService


async def create_test_user(
    db: AsyncSession,
    username: str = "testuser",
    email: str = "test@example.com",
    password: str = "testpassword123",
    role: str = "user"
) -> User:
    """Create a test user."""
    auth_service = AuthService()
    user = await auth_service.create_user(
        username=username,
        email=email,
        password=password,
        db=db
    )
    if role != "user":
        user.role = role
        await db.commit()
        await db.refresh(user)
    return user


async def create_test_document_source(
    db: AsyncSession,
    name: str = "Test Source",
    source_type: str = "web",
    config: dict = None
) -> DocumentSource:
    """Create a test document source."""
    if config is None:
        config = {"url": "https://example.com"}
    
    source = DocumentSource(
        name=name,
        source_type=source_type,
        config=config,
        is_active=True
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def create_test_document(
    db: AsyncSession,
    source: DocumentSource,
    title: str = "Test Document",
    content: str = "Test content"
) -> Document:
    """Create a test document."""
    import hashlib
    
    document = Document(
        title=title,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        source_id=source.id,
        source_identifier=str(uuid4()),
        is_processed=False
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    return document


async def create_test_chat_session(
    db: AsyncSession,
    user: User,
    title: str = "Test Session"
) -> ChatSession:
    """Create a test chat session."""
    session = ChatSession(
        title=title,
        user_id=user.id,
        is_active=True
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def create_test_chat_message(
    db: AsyncSession,
    session: ChatSession,
    content: str = "Test message",
    role: str = "user"
) -> ChatMessage:
    """Create a test chat message."""
    message = ChatMessage(
        session_id=session.id,
        content=content,
        role=role,
        message_type="text"
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message

