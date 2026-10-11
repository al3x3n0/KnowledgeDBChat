"""A number a caller sends to size a query has a floor and a ceiling.

`limit: int = 20` on a route accepts -1, which is "no limit" to the database,
and 0, which is a divisor in every page count. Three routes took such a number
straight from the query string:

* `GET /export` divided by `page_size` (0 was a 500) and handed a negative
  one to the database as its limit (every row);
* `GET /api-keys/{id}/usage` turned `days` into a timedelta, which overflows;
* `GET /admin/logs` sliced with `lines`: 0 returned the whole file and a
  negative number dropped lines from the wrong end. It also read the file
  whole, on the event loop.

The source check below requires `ge` and `le` on such a parameter. A route
that clamps the number itself is listed in CLAMPED_BY_HAND, which may only
shrink.
"""

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.endpoints import admin as admin_endpoints

pytestmark = pytest.mark.unit

APP = Path(__file__).resolve().parents[1] / "app"

# Parameters that say how much to return or how far back to look. An offset or
# a page number is not here: too large a one returns nothing.
SIZES = {
    "limit",
    "page_size",
    "per_page",
    "lines",
    "days",
    "hours",
    "top_k",
    "max_results",
    "count",
    "size",
}

# Routes that bound the number in their own body or in the service they call.
CLAMPED_BY_HAND = {
    "get_ai_hub_recommendation_feedback_stats",
    "get_source_sync_logs",
    "export_source_sync_logs_csv",
    "list_autonomous_rnd_eval_launches",
    "list_autonomous_rnd_eval_runs",
    "get_chat_sessions",
    "get_documents",
    "list_latex_projects",
    "list_personas",
    "list_collaboration_users",
    "get_users",
}

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _routes():
    for path in sorted(APP.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.AsyncFunctionDef, ast.FunctionDef)):
                continue
            if any(
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr in HTTP_METHODS
                for d in fn.decorator_list
            ):
                yield path, fn


def _unbounded_sizes(fn):
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pairs = list(zip(positional, defaults)) + list(
        zip(args.kwonlyargs, args.kw_defaults)
    )
    for arg, default in pairs:
        if arg.arg not in SIZES or arg.annotation is None:
            continue
        if "int" not in ast.unparse(arg.annotation):
            continue
        keywords = (
            {k.arg for k in default.keywords}
            if isinstance(default, ast.Call)
            else set()
        )
        if not (keywords & {"ge", "gt"} and keywords & {"le", "lt"}):
            yield arg.arg


def test_a_size_parameter_has_a_floor_and_a_ceiling():
    examined = 0
    unbounded = {}
    for path, fn in _routes():
        examined += 1
        names = list(_unbounded_sizes(fn))
        if names:
            unbounded[fn.name] = (path.relative_to(APP), names)

    # A check that reads no routes passes for ever.
    assert examined > 500

    new = {k: v for k, v in unbounded.items() if k not in CLAMPED_BY_HAND}
    assert not new, "Give these Query(..., ge=..., le=...): " + "; ".join(
        f"{p}:{n}({', '.join(a)})" for n, (p, a) in sorted(new.items())
    )
    stale = CLAMPED_BY_HAND - set(unbounded)
    assert not stale, f"No longer unbounded, remove from CLAMPED_BY_HAND: {stale}"


@pytest.mark.parametrize("page_size", [0, -1, 101])
def test_export_jobs_refuses_a_page_size_out_of_range(
    client: TestClient, auth_headers, page_size
):
    response = client.get(f"/api/v1/export?page_size={page_size}", headers=auth_headers)
    assert response.status_code == 422


def test_export_jobs_refuses_page_zero(client: TestClient, auth_headers):
    response = client.get("/api/v1/export?page=0", headers=auth_headers)
    assert response.status_code == 422


def test_export_jobs_still_lists(client: TestClient, auth_headers):
    response = client.get("/api/v1/export?page_size=5", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["page_size"] == 5


@pytest.mark.parametrize("days", [0, -3, 10**12])
def test_api_key_usage_refuses_days_out_of_range(
    client: TestClient, auth_headers, days
):
    response = client.get(
        f"/api/v1/api-keys/00000000-0000-0000-0000-000000000001/usage?days={days}",
        headers=auth_headers,
    )
    assert response.status_code == 422


@pytest.mark.parametrize("lines", [0, -5, admin_endpoints.MAX_LOG_LINES + 1])
def test_logs_refuse_a_line_count_out_of_range(
    client: TestClient, admin_headers, lines
):
    response = client.get(f"/api/v1/admin/logs?lines={lines}", headers=admin_headers)
    assert response.status_code == 422


def test_logs_return_the_end_of_the_file(
    client: TestClient, admin_headers, tmp_path, monkeypatch
):
    from app.core.config import settings

    log = tmp_path / "app.log"
    log.write_text("".join(f"line {i}\n" for i in range(1, 51)))
    monkeypatch.setattr(settings, "LOG_FILE", str(log))

    response = client.get("/api/v1/admin/logs?lines=3", headers=admin_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["logs"] == ["line 48", "line 49", "line 50"]
    assert body["total_lines"] == 50
    assert body["returned_lines"] == 3


def test_the_tail_of_a_short_file_is_the_file(tmp_path):
    log = tmp_path / "short.log"
    log.write_text("a\nb\n")

    assert admin_endpoints._tail_of_file(str(log), 100) == (["a\n", "b\n"], 2)


# --------------------------------------------------------------------------
# MCP: the same number arrives from a model's tool call, past any route
# --------------------------------------------------------------------------


class _Key:
    def __init__(self, scopes):
        self.scopes = scopes

    def has_scope(self, scope):
        return scope in self.scopes


async def _sources(db, count):
    from uuid import uuid4

    from app.models.document import DocumentSource

    for i in range(count):
        db.add(
            DocumentSource(
                name=f"src-{i}-{uuid4().hex[:6]}",
                source_type="file",
                config={},
                is_active=True,
            )
        )
    await db.commit()


@pytest.mark.parametrize("limit", [-1, 0, None, "many"])
async def test_mcp_list_sources_never_runs_without_a_limit(
    db_session, test_user, limit
):
    from app.mcp.auth import MCPAuthContext
    from app.mcp.tools.documents import DocumentsTool

    await _sources(db_session, 3)
    auth = MCPAuthContext(user=test_user, api_key=_Key(["read"]), scopes=["read"])

    result = await DocumentsTool().list_sources(auth=auth, db=db_session, limit=limit)

    # -1 was LIMIT -1, every row; 0 and the rest fall back to a real limit.
    expected = 1 if limit in (-1, 0) else 3
    assert "error" not in result
    assert len(result["sources"]) == expected


async def test_mcp_list_documents_bounds_limit_and_offset(db_session, test_user):
    import hashlib
    from uuid import uuid4

    from app.mcp.auth import MCPAuthContext
    from app.mcp.tools.documents import DocumentsTool
    from app.models.document import Document, DocumentSource

    source = DocumentSource(name=f"s-{uuid4().hex[:6]}", source_type="file", config={})
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    for i in range(3):
        db_session.add(
            Document(
                title=f"doc {i}",
                content="body",
                content_hash=hashlib.sha256(f"{i}".encode()).hexdigest(),
                source_id=source.id,
                source_identifier=f"test:{uuid4().hex}",
            )
        )
    await db_session.commit()
    auth = MCPAuthContext(user=test_user, api_key=_Key(["read"]), scopes=["read"])

    result = await DocumentsTool().list_documents(
        auth=auth, db=db_session, limit=-1, offset=-5
    )

    assert result["total"] == 3
    assert len(result["documents"]) == 1


async def test_mcp_list_sources_describes_each_source(db_session, test_user):
    """It read four attributes the model lacks and answered with no sources."""
    import hashlib
    import json
    from datetime import datetime, timezone
    from uuid import uuid4

    from app.mcp.auth import MCPAuthContext
    from app.mcp.tools.documents import DocumentsTool
    from app.models.document import Document, DocumentSource

    synced = DocumentSource(
        name="a-synced",
        source_type="web",
        config={"description": "The handbook"},
        is_active=True,
        last_sync=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    failed = DocumentSource(
        name="b-failed",
        source_type="gitlab",
        config={},
        is_active=True,
        last_error="401",
    )
    hidden = DocumentSource(
        name="c-inactive", source_type="file", config={}, is_active=False
    )
    db_session.add_all([synced, failed, hidden])
    await db_session.commit()
    for i in range(2):
        db_session.add(
            Document(
                title=f"page {i}",
                content="body",
                content_hash=hashlib.sha256(f"p{i}".encode()).hexdigest(),
                source_id=synced.id,
                source_identifier=f"test:{uuid4().hex}",
            )
        )
    await db_session.commit()
    auth = MCPAuthContext(user=test_user, api_key=_Key(["read"]), scopes=["read"])

    result = await DocumentsTool().list_sources(auth=auth, db=db_session)

    json.dumps(result)
    by_name = {s["name"]: s for s in result["sources"]}
    assert set(by_name) == {"a-synced", "b-failed"}
    assert by_name["a-synced"]["description"] == "The handbook"
    assert by_name["a-synced"]["document_count"] == 2
    assert by_name["a-synced"]["sync_status"] == "synced"
    assert by_name["a-synced"]["last_sync_at"].startswith("2026-10-01")
    assert by_name["b-failed"]["document_count"] == 0
    assert by_name["b-failed"]["sync_status"] == "error"
    assert by_name["b-failed"]["last_sync_at"] is None


async def test_mcp_get_document_returns_the_stored_metadata(db_session, test_user):
    """`doc.metadata` is the table registry, not the document's metadata."""
    import hashlib
    import json
    from uuid import uuid4

    from app.mcp.auth import MCPAuthContext
    from app.mcp.tools.documents import DocumentsTool
    from app.models.document import Document, DocumentSource

    source = DocumentSource(name=f"s-{uuid4().hex[:6]}", source_type="file", config={})
    db_session.add(source)
    await db_session.commit()
    doc = Document(
        title="Report",
        content="body",
        content_hash=hashlib.sha256(b"report").hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        extra_metadata={"pages": 3},
    )
    db_session.add(doc)
    await db_session.commit()
    auth = MCPAuthContext(user=test_user, api_key=_Key(["read"]), scopes=["read"])

    result = await DocumentsTool().get_document(
        auth=auth, db=db_session, document_id=str(doc.id)
    )

    json.dumps(result)
    assert result["metadata"] == {"pages": 3}
    assert result["content"] == "body"
