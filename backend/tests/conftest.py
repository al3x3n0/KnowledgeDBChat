"""
Pytest configuration and fixtures for backend tests.
"""

import asyncio
import importlib.machinery
import importlib.util
import sys
import types
from typing import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.models.user import User
from app.services.auth_service import AuthService

if "pptx" not in sys.modules:
    pptx_stub = types.ModuleType("pptx")
    pptx_stub.Presentation = object
    sys.modules["pptx"] = pptx_stub
    pptx_enum_stub = types.ModuleType("pptx.enum")
    pptx_shapes_stub = types.ModuleType("pptx.enum.shapes")
    pptx_shapes_stub.PP_PLACEHOLDER = object()
    pptx_shapes_stub.MSO_SHAPE = object()
    pptx_text_stub = types.ModuleType("pptx.enum.text")
    pptx_text_stub.PP_ALIGN = object()
    pptx_text_stub.MSO_ANCHOR = object()

    def _pptx_dimension(*args, **kwargs):
        return 0

    pptx_util_stub = types.ModuleType("pptx.util")
    pptx_util_stub.Inches = _pptx_dimension
    pptx_util_stub.Pt = _pptx_dimension
    pptx_util_stub.Emu = _pptx_dimension
    pptx_dml_stub = types.ModuleType("pptx.dml")
    pptx_dml_color_stub = types.ModuleType("pptx.dml.color")

    class _RGBColor:
        def __init__(self, *args, **kwargs):
            pass

    pptx_dml_color_stub.RGBColor = _RGBColor
    sys.modules["pptx.enum"] = pptx_enum_stub
    sys.modules["pptx.enum.shapes"] = pptx_shapes_stub
    sys.modules["pptx.enum.text"] = pptx_text_stub
    sys.modules["pptx.util"] = pptx_util_stub
    sys.modules["pptx.dml"] = pptx_dml_stub
    sys.modules["pptx.dml.color"] = pptx_dml_color_stub

if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def encode(self, texts, *args, **kwargs):
            if isinstance(texts, list):
                return [[0.0] * 4 for _ in texts]
            return [0.0] * 4

    sentence_transformers_stub.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers_stub


def _real_module_available(name: str) -> bool:
    """True if the real (non-stubbed) package is importable."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


if "bs4" not in sys.modules and not _real_module_available("bs4"):
    bs4_stub = types.ModuleType("bs4")
    bs4_stub.__spec__ = importlib.machinery.ModuleSpec("bs4", loader=None)

    class _DummyBeautifulSoup:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def find_all(self, *args, **kwargs):
            return []

        def __str__(self):
            return ""

    class _DummyNavigableString(str):
        pass

    bs4_stub.BeautifulSoup = _DummyBeautifulSoup
    bs4_stub.NavigableString = _DummyNavigableString
    sys.modules["bs4"] = bs4_stub

if "croniter" not in sys.modules and not _real_module_available("croniter"):
    croniter_stub = types.ModuleType("croniter")

    class _DummyCronIter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def get_next(self, *args, **kwargs):
            return 0

    croniter_stub.croniter = _DummyCronIter
    sys.modules["croniter"] = croniter_stub

if "mammoth" not in sys.modules:
    mammoth_stub = types.ModuleType("mammoth")

    class _DummyMammothResult:
        def __init__(self, value: str = "", messages=None):
            self.value = value
            self.messages = messages or []

    def _convert_to_html(*args, **kwargs):
        return _DummyMammothResult()

    mammoth_stub.convert_to_html = _convert_to_html
    sys.modules["mammoth"] = mammoth_stub

if "jsonpath_ng" not in sys.modules:
    jsonpath_ng_stub = types.ModuleType("jsonpath_ng")

    class _DummyJsonPathMatch:
        def __init__(self, value=None):
            self.value = value

    class _DummyJsonPath:
        def __init__(self, expression: str):
            self.expression = expression

        def find(self, data):
            return [_DummyJsonPathMatch(data)]

    def _parse_jsonpath(expression: str):
        return _DummyJsonPath(expression)

    jsonpath_ng_stub.parse = _parse_jsonpath
    sys.modules["jsonpath_ng"] = jsonpath_ng_stub


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(_element, _compiler, **_kwargs):
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid_sqlite(_element, _compiler, **_kwargs):
    return "CHAR(36)"


# Test database URL (in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Create test session factory
TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def _restore_current_event_loop(event_loop):
    """Keep Python 3.11 sync tests attached to pytest-asyncio's session loop."""
    asyncio.set_event_loop(event_loop)
    yield
    if not event_loop.is_closed():
        asyncio.set_event_loop(event_loop)


@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
def client(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Create a test client without contacting production infrastructure."""
    from app.services.storage_service import storage_service
    from app.services.vector_store import vector_store_service
    from app.utils.redis_subscriber import redis_subscriber
    from main import app

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(vector_store_service, "initialize", _noop)
    monkeypatch.setattr(storage_service, "initialize", _noop)
    monkeypatch.setattr(redis_subscriber, "start", _noop)
    monkeypatch.setattr(redis_subscriber, "stop", _noop)

    def override_get_db():
        return db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    auth_service = AuthService()

    user = await auth_service.create_user(
        username="testuser",
        email="test@example.com",
        password="testpassword123",
        full_name="Test User",
        db=db_session,
    )

    return user


@pytest.fixture
async def admin_user(db_session: AsyncSession) -> User:
    """Create an admin test user."""
    auth_service = AuthService()

    user = await auth_service.create_user(
        username="admin",
        email="admin@example.com",
        password="adminpassword123",
        full_name="Admin User",
        db=db_session,
    )

    # Set admin role
    user.role = "admin"
    await db_session.commit()
    await db_session.refresh(user)

    return user


@pytest.fixture
def auth_headers(client: TestClient, test_user: User) -> dict:
    """Get authentication headers for test user."""
    token = AuthService().create_access_token(test_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(client: TestClient, admin_user: User) -> dict:
    """Get authentication headers for admin user."""
    token = AuthService().create_access_token(admin_user.id)
    return {"Authorization": f"Bearer {token}"}
