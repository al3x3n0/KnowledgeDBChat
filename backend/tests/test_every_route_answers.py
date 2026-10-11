"""Every route answers a request it can be sent, without a server error.

About four in five route handlers are named in no test. Each time one of
them was first run -- the backlog list, both inline arXiv ingests, template
upload, the chat workflow tool -- it turned out never to have worked, and had
been answering 500 or failing inside `except Exception` since it was written.

This does not test what a route returns. It sends each one the request a
client could send with nothing set up -- an id that names nothing, an empty
body -- and requires an answer that is not a 500. That is enough to find a
route that cannot run at all: the first pass found ten, among them the
workflow editor's list of built-in tools.

A POST route with no path parameter gets a body built from its own declared
schema -- every required field, the smallest value its type allows -- so the
handler really runs. For those the network is closed and task dispatch is
recorded, not sent: a route that needs the model, arXiv or object storage
must then say so with a 503, which is a different answer from "this server
is broken". That second walk found nine more, one of them a setting that
could never be saved.

Left out: the few routes in `NEEDS_A_SERVICE`, which wait on something a
test run does not have. A built body that validation refuses (422) tells
nothing about the handler; `test_most_built_bodies_reach_their_handler`
keeps that share from growing unnoticed.
"""

from __future__ import annotations

import json
import re
import socket
import uuid

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.services.auth_service import AuthService
from tests.conftest import TEST_ON_POSTGRES

#: (method, path) -> why it cannot answer here. Each waits for another
#: process; the wait itself is what would be measured.
NEEDS_A_SERVICE = {
    ("POST", "/api/v1/admin/sources/{source_id}/dry-run"): (
        "waits up to a minute for a Celery worker to run the dry run"
    ),
}


def _request_path(route: APIRoute) -> str:
    def value(match):
        name = match.group(1).split(":")[0]
        return str(uuid.uuid4()) if name.endswith("id") else "x"

    return re.sub(r"\{([^}]+)\}", value, route.path)


def _requests(app):
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        has_parameter = "{" in route.path
        for method in sorted(route.methods):
            if method == "GET" or (
                has_parameter and method in {"DELETE", "PUT", "PATCH", "POST"}
            ):
                if (method, route.path) not in NEEDS_A_SERVICE:
                    yield method, route.path, _request_path(route)


@pytest.mark.skipif(
    TEST_ON_POSTGRES,
    reason="the HTTP client cannot share a Postgres session across loops",
)
def test_no_route_answers_with_a_server_error(db_session, admin_user, monkeypatch):
    from app.services.storage_service import storage_service
    from app.services.vector_store import vector_store_service
    from app.utils.redis_subscriber import redis_subscriber
    from main import app

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(type(vector_store_service), "initialize", _noop)
    monkeypatch.setattr(type(storage_service), "initialize", _noop)
    monkeypatch.setattr(type(redis_subscriber), "start", _noop)
    monkeypatch.setattr(type(redis_subscriber), "stop", _noop)
    app.dependency_overrides[get_db] = lambda: db_session
    token = AuthService().create_access_token(admin_user.id)
    headers = {"Authorization": f"Bearer {token}"}

    broken, asked = [], 0
    try:
        # A handler that raises is reported as a 500, like one that answers it.
        with TestClient(app, raise_server_exceptions=False) as client:
            for method, declared, path in _requests(app):
                body = {} if method in {"POST", "PUT", "PATCH"} else None
                answer = client.request(method, path, headers=headers, json=body)
                asked += 1
                if answer.status_code == 500:
                    broken.append(f"{method} {declared}: {answer.text[:160]}")
                # What a failed request left in the shared session must not
                # be the next request's problem.
                client.portal.call(db_session.rollback)
    finally:
        app.dependency_overrides.clear()

    assert asked > 400  # the routes were really walked
    assert broken == [], "\n".join(broken)


def test_the_exceptions_are_real_routes():
    from main import app

    declared = {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    assert set(NEEDS_A_SERVICE) <= declared


# --- POST routes, with a body built from the schema -------------------------


def _value(schema, spec, name="", depth=0):
    """The smallest thing `schema` accepts, required fields only."""
    if depth > 6 or not isinstance(schema, dict):
        return None
    if "$ref" in schema:
        target = spec["components"]["schemas"][schema["$ref"].split("/")[-1]]
        return _value(target, spec, name, depth + 1)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [o for o in schema[key] if o.get("type") != "null"]
            return _value(options[0], spec, name, depth + 1) if options else None
    if "allOf" in schema:
        return _value(schema["allOf"][0], spec, name, depth + 1)
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        required = schema.get("required") or []
        return {
            field: _value(sub, spec, field, depth + 1)
            for field, sub in (schema.get("properties") or {}).items()
            if field in required
        }
    if kind == "array":
        item = _value(schema.get("items", {}), spec, name, depth + 1)
        return [] if item is None else [item] * max(schema.get("minItems", 0), 1)
    if kind == "string":
        fmt = schema.get("format")
        if fmt == "uuid" or name == "id" or name.endswith("_id"):
            return str(uuid.uuid4())
        if fmt == "binary":
            return _FILE
        if fmt == "date-time":
            return "2026-01-01T00:00:00"
        if fmt == "date":
            return "2026-01-01"
        if fmt == "email" or "email" in name:
            return "probe@example.com"
        if fmt in ("uri", "url") or "url" in name:
            return "https://example.com/probe"
        return ("probe " * 10)[: max(schema.get("minLength", 0), 11)].strip()
    if kind == "integer":
        return int(schema.get("minimum", schema.get("exclusiveMinimum", 0) + 1))
    if kind == "number":
        return float(schema.get("minimum", 1))
    if kind == "boolean":
        return False
    return {}


_FILE = object()


def _post_requests(spec):
    for path, item in sorted(spec["paths"].items()):
        operation = item.get("post")
        if not operation or "{" in path or ("POST", path) in NEEDS_A_SERVICE:
            continue
        content = (operation.get("requestBody") or {}).get("content") or {}
        sent = {}
        if "application/json" in content:
            sent["json"] = _value(content["application/json"].get("schema", {}), spec)
        else:
            form = next((content[kind] for kind in content if "form" in kind), None)
            if form is not None:
                fields = _value(form.get("schema", {}), spec) or {}
                files = {
                    name: ("probe.txt", b"probe file\n", "text/plain")
                    for name, value in fields.items()
                    if value is _FILE
                }
                sent["data"] = {
                    name: json.dumps(value)
                    if isinstance(value, (dict, list))
                    else str(value)
                    for name, value in fields.items()
                    if value is not _FILE
                }
                if files:
                    sent["files"] = files
        sent["params"] = {
            parameter["name"]: _value(
                parameter.get("schema", {}), spec, parameter["name"], 1
            )
            for parameter in operation.get("parameters", [])
            if parameter.get("in") == "query" and parameter.get("required")
        }
        yield path, sent


@pytest.fixture
def _posted(db_session, admin_user, monkeypatch):
    """Every such route posted to once; (path, status, text) for each."""
    from celery.app.task import Task

    from app.services.llm_service import LLMService
    from app.services.storage_service import storage_service
    from app.services.vector_store import vector_store_service
    from app.utils.exceptions import LLMServiceError
    from app.utils.redis_subscriber import redis_subscriber
    from main import app

    async def _noop(*args, **kwargs):
        return None

    async def _no_model(self, *args, **kwargs):
        raise LLMServiceError("no model in a test run")

    def _no_network(self, address):
        raise ConnectionRefusedError(f"network closed in a test run: {address}")

    class _Queued:
        id = "probe-task"

        def get(self, *args, **kwargs):
            raise TimeoutError("no worker in a test run")

    monkeypatch.setattr(type(vector_store_service), "initialize", _noop)
    monkeypatch.setattr(type(storage_service), "initialize", _noop)
    monkeypatch.setattr(type(redis_subscriber), "start", _noop)
    monkeypatch.setattr(type(redis_subscriber), "stop", _noop)
    monkeypatch.setattr(LLMService, "generate_response", _no_model)
    monkeypatch.setattr(LLMService, "generate_structured", _no_model)
    monkeypatch.setattr(Task, "apply_async", lambda self, *a, **k: _Queued())
    app.dependency_overrides[get_db] = lambda: db_session
    token = AuthService().create_access_token(admin_user.id)
    headers = {"Authorization": f"Bearer {token}"}
    spec = app.openapi()

    answers = []
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            # Closed only once the app is up, and for these requests alone.
            monkeypatch.setattr(socket.socket, "connect", _no_network, raising=False)
            for path, sent in _post_requests(spec):
                answer = client.post(path, headers=headers, **sent)
                answers.append((path, answer.status_code, answer.text[:160]))
                client.portal.call(db_session.rollback)
            monkeypatch.undo()
    finally:
        app.dependency_overrides.clear()
    return answers


@pytest.mark.skipif(
    TEST_ON_POSTGRES,
    reason="the HTTP client cannot share a Postgres session across loops",
)
def test_no_post_route_answers_with_a_server_error(_posted):
    broken = [f"POST {path}: {text}" for path, status, text in _posted if status == 500]

    assert len(_posted) > 120  # the routes were really walked
    assert broken == [], "\n".join(broken)


@pytest.mark.skipif(
    TEST_ON_POSTGRES,
    reason="the HTTP client cannot share a Postgres session across loops",
)
def test_most_built_bodies_reach_their_handler(_posted):
    # A 422 means the built body was refused and the handler never ran. Some
    # schemas need more than their types say; that share must not grow until
    # this walk is checking nothing.
    refused = [path for path, status, _ in _posted if status == 422]

    assert len(refused) <= 20, refused
