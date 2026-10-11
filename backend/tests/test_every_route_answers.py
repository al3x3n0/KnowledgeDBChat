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

Left out: POST routes with no path parameter (an empty body there is
refused by validation before the handler runs, or creates things), and the
few routes in `NEEDS_A_SERVICE`, which wait on something a test run does not
have.
"""

from __future__ import annotations

import re
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
