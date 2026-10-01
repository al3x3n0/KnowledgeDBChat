"""Every route knows who is calling, unless it is on a short list that says why.

Twelve knowledge-graph routes and the document editor's read and write
declared no user at all. They were written like their neighbours, minus one
parameter, and nothing failed: an endpoint without an auth dependency is not
broken, it is public. `GET /kg/stats` answered an anonymous request with the
entity count; `PUT /documents/{id}/edit` would have overwritten a document for
anyone who knew its id.

So the absence is what is tested. A route with no auth dependency must be
named below with its reason, and a WebSocket -- which cannot use the HTTP
dependency -- must authenticate in its own handler.
"""

import inspect

import pytest

pytestmark = pytest.mark.unit

#: HTTP routes that are meant to answer anyone, and why.
PUBLIC = {
    ("GET", "/"): "landing response",
    ("GET", "/health"): "liveness probe",
    ("POST", "/api/v1/auth/login"): "this is how a caller gets a token",
    ("POST", "/api/v1/auth/register"): "there is no account yet",
    ("POST", "/api/v1/auth/refresh"): "authenticates by the refresh token it carries",
    ("POST", "/api/v1/auth/logout"): "nothing to protect; clears the caller's state",
    ("GET", "/api/v1/mcp/info"): "server identity for MCP discovery",
    (
        "POST",
        "/api/v1/external-agents/compops-webhooks/{subscription_id}",
    ): "authenticated by the webhook's signature, checked in the handler",
}

#: What a WebSocket handler must call. The HTTP dependency reads a Request,
#: which a socket does not have, so each handler does this itself.
WEBSOCKET_AUTH_CALLS = (
    "authenticate_websocket",
    "require_websocket_auth",
    "authorize_owner",
    # The agent-job stream takes its authenticator as an injected parameter.
    "authenticate_token",
)

#: Sockets that authenticate another way, and how.
WEBSOCKET_OTHERWISE = {
    "/api/v1/dashboard/ws": "waits for an auth message and verifies its token",
}


def _dependency_names(dependant, seen=None):
    seen = set() if seen is None else seen
    seen.add(getattr(dependant.call, "__name__", "") if dependant.call else "")
    for child in dependant.dependencies:
        _dependency_names(child, seen)
    return seen


def _knows_its_caller(names) -> bool:
    return any(
        "current_user" in n or n in ("require_admin", "get_mcp_auth") for n in names
    )


@pytest.fixture(scope="module")
def routes():
    from main import app

    found = [r for r in app.routes if hasattr(r, "dependant")]
    assert len(found) > 300, "too few routes; this guard would pass vacuously"
    return found


def test_no_http_route_is_public_by_omission(routes):
    unexplained = []
    for route in routes:
        methods = getattr(route, "methods", None)
        if not methods:
            continue
        if _knows_its_caller(_dependency_names(route.dependant)):
            continue
        for method in sorted(methods - {"HEAD", "OPTIONS"}):
            if (method, route.path) not in PUBLIC:
                unexplained.append(f"{method} {route.path}")
    assert not unexplained, (
        "These routes take no user. Add an auth dependency, or list the route "
        "in PUBLIC with the reason it answers anyone:\n"
        + "\n".join(f"  - {r}" for r in sorted(unexplained))
    )


def test_the_public_list_names_only_routes_that_exist_and_are_public(routes):
    """A list that outlives its routes hides the next omission behind a name."""
    actual = {
        (method, route.path): route
        for route in routes
        for method in (getattr(route, "methods", None) or ())
    }
    for key in PUBLIC:
        assert key in actual, f"{key} is listed as public but is not a route"
        assert not _knows_its_caller(
            _dependency_names(actual[key].dependant)
        ), f"{key} now takes a user; remove it from PUBLIC"


def test_every_websocket_authenticates_in_its_handler(routes):
    sockets = [r for r in routes if not getattr(r, "methods", None)]
    assert len(sockets) >= 10, "too few sockets; this guard would pass vacuously"
    unauthenticated = []
    for route in sockets:
        if route.path in WEBSOCKET_OTHERWISE:
            continue
        source = inspect.getsource(route.endpoint)
        if not any(call in source for call in WEBSOCKET_AUTH_CALLS):
            unauthenticated.append(route.path)
    assert (
        not unauthenticated
    ), "These WebSocket handlers never check who connected:\n" + "\n".join(
        f"  - {p}" for p in sorted(unauthenticated)
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/kg/stats",
        "/api/v1/kg/entities",
        "/api/v1/kg/global/graph",
        "/api/v1/kg/types",
        "/api/v1/documents/00000000-0000-0000-0000-000000000000/edit",
        "/api/v1/repo-reports/sections",
        "/api/v1/mcp-config/tools",
    ],
)
def test_a_route_that_answered_anyone_now_refuses(client, path):
    assert client.get(path).status_code in (401, 403)


def test_a_document_cannot_be_overwritten_without_a_token(client):
    response = client.put(
        "/api/v1/documents/00000000-0000-0000-0000-000000000000/edit",
        json={"html_content": "<p>replaced</p>"},
    )
    assert response.status_code in (401, 403)


def test_a_graph_cannot_be_rebuilt_without_a_token(client):
    response = client.post(
        "/api/v1/kg/document/00000000-0000-0000-0000-000000000000/rebuild"
    )
    assert response.status_code in (401, 403)


def test_a_signed_in_user_can_still_read_the_graph(client, auth_headers):
    assert client.get("/api/v1/kg/stats", headers=auth_headers).status_code == 200


class _Socket:
    def __init__(self):
        self.closed = None

    async def close(self, code=1000, reason=""):
        self.closed = (code, reason)


class TestWatchingSomeoneElsesJob:
    """Three progress streams accepted any connection that knew a job id."""

    async def _authorize(self, monkeypatch, caller, owner_id):
        from app.utils import websocket_auth

        async def fake_authenticate(_websocket, token=None):
            return caller

        monkeypatch.setattr(websocket_auth, "authenticate_websocket", fake_authenticate)
        socket = _Socket()
        user = await websocket_auth.authorize_owner(socket, owner_id)
        return user, socket.closed

    async def test_no_token_is_refused(self, monkeypatch):
        user, closed = await self._authorize(monkeypatch, None, "owner")
        assert user is None and closed[0] == 4001

    async def test_a_stranger_is_told_it_does_not_exist(self, monkeypatch):
        from types import SimpleNamespace

        stranger = SimpleNamespace(id="stranger", is_admin=lambda: False)
        user, closed = await self._authorize(monkeypatch, stranger, "owner")
        # 4004, not a "forbidden": confirming the id is real is what is withheld.
        assert user is None and closed == (4004, "Job not found")

    async def test_the_owner_and_an_admin_are_let_in(self, monkeypatch):
        from types import SimpleNamespace

        owner = SimpleNamespace(id="owner", is_admin=lambda: False)
        admin = SimpleNamespace(id="someone-else", is_admin=lambda: True)
        for caller in (owner, admin):
            user, closed = await self._authorize(monkeypatch, caller, "owner")
            assert user is caller and closed is None

    async def test_a_job_with_no_owner_is_nobodys_to_watch(self, monkeypatch):
        from types import SimpleNamespace

        caller = SimpleNamespace(id="anyone", is_admin=lambda: False)
        user, closed = await self._authorize(monkeypatch, caller, None)
        assert user is None and closed[0] == 4004
