"""A viewer reads, and changes nothing shared (decided 2026-10-14).

The role came from the directory's group mapping and nothing enforced it,
so a viewer could upload, delete, start agent jobs and run tools. Each of the
three ways in is checked where every request passes: the authentication
dependency (HTTP), the tool policy engine (chat, workflow and MCP tools), and
the MCP key's scopes.
"""

import uuid

import pytest
from fastapi.routing import APIRoute

from app.mcp.auth import MCPAuthContext, validate_api_key
from app.models.user import User
from app.services import viewer_access
from app.services.auth_service import (
    AuthService,
    get_current_user,
    get_current_user_optional,
    require_admin,
)
from app.services.tool_policy_engine import evaluate_tool_policy

pytestmark = pytest.mark.unit

_AUTH = {get_current_user, get_current_user_optional, require_admin}


@pytest.fixture
async def viewer(db_session):
    user = await AuthService().create_user(
        username="vera",
        email="vera@example.com",
        password="viewer-password-1",
        full_name="Vera Viewer",
        db=db_session,
    )
    user.role = "viewer"
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def viewer_headers(client, viewer):
    return {"Authorization": f"Bearer {AuthService().create_access_token(viewer.id)}"}


def _calls(dependant):
    yield dependant.call
    for sub in dependant.dependencies:
        yield from _calls(sub)


def _write_routes():
    from main import app

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods - viewer_access.SAFE_METHODS):
            yield route, method


def _needs_a_user(route):
    return any(call in _AUTH for call in _calls(route.dependant))


def _url(path):
    # Any value fills a path parameter: the refusal comes before the handler.
    import re

    return re.sub(r"\{[^}:]+(:[^}]+)?\}", str(uuid.uuid4()), path)


class TestHttp:
    def test_every_write_a_viewer_may_not_make_is_refused(self, client, viewer_headers):
        allowed_and_answered = []
        not_refused = []
        for route, method in _write_routes():
            if not _needs_a_user(route):
                continue
            response = client.request(
                method, _url(route.path), headers=viewer_headers, json={}
            )
            if (method, route.path) in viewer_access.VIEWER_MAY:
                if response.status_code == 403 and (
                    response.json().get("detail") == viewer_access.REFUSAL
                ):
                    allowed_and_answered.append((method, route.path))
                continue
            if not (
                response.status_code == 403
                and response.json().get("detail") == viewer_access.REFUSAL
            ):
                not_refused.append((method, route.path, response.status_code))

        assert not_refused == []
        # And none of the writes a viewer may make is refused as one.
        assert allowed_and_answered == []

    def test_the_walk_covered_the_platform(self):
        """A walk that found nothing would pass for ever."""
        routes = [r for r, _m in _write_routes() if _needs_a_user(r)]
        assert len(routes) > 250

    def test_every_allowed_write_is_a_real_route(self):
        real = {(m, r.path) for r, m in _write_routes()}

        assert viewer_access.VIEWER_MAY - real == set()

    def test_a_viewer_reads(self, client, viewer_headers):
        response = client.get("/api/v1/documents/", headers=viewer_headers)

        assert response.status_code != 403

    def test_a_viewer_chats(self, client, viewer_headers):
        response = client.post(
            "/api/v1/chat/sessions", headers=viewer_headers, json={"title": "x"}
        )

        assert response.status_code in (200, 201), response.text

    def test_a_user_is_not_refused_the_same_write(self, client, auth_headers):
        """The control: the refusal is about the role, not the route."""
        response = client.post(
            "/api/v1/searches", headers=auth_headers, json={"name": "s", "query": "q"}
        )

        assert response.status_code != 403

    async def test_an_optional_user_route_does_not_treat_a_viewer_as_anonymous(
        self, viewer, db_session
    ):
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        from starlette.requests import Request

        token = AuthService().create_access_token(viewer.id)
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/v1/documents/upload",
                "headers": [],
                "query_string": b"",
            }
        )

        with pytest.raises(HTTPException) as refused:
            await get_current_user_optional(
                HTTPAuthorizationCredentials(scheme="Bearer", credentials=token),
                None,
                db_session,
                request,
            )
        assert refused.value.status_code == 403


class TestTools:
    async def test_a_viewer_may_call_a_read_only_tool(self, viewer, db_session):
        decision = await evaluate_tool_policy(
            db=db_session, tool_name="search_documents", user=viewer
        )

        assert decision.allowed

    async def test_but_not_one_that_changes_things(self, viewer, db_session):
        decision = await evaluate_tool_policy(
            db=db_session, tool_name="delete_document", user=viewer
        )

        assert not decision.allowed
        assert "read-only" in decision.denied_reason

    async def test_nor_one_that_declares_nothing(self, viewer, db_session):
        decision = await evaluate_tool_policy(
            db=db_session, tool_name=f"user_tool:{uuid.uuid4()}", user=viewer
        )

        assert not decision.allowed

    async def test_a_user_may_call_the_tool_a_viewer_may_not(
        self, test_user, db_session
    ):
        decision = await evaluate_tool_policy(
            db=db_session, tool_name="delete_document", user=test_user
        )

        assert decision.allowed


class TestMcpKeys:
    async def _key(self, db_session, owner, scopes):
        from app.mcp.auth import hash_api_key
        from app.models.api_key import APIKey

        raw = f"kdbc_{uuid.uuid4().hex}"
        key = APIKey(
            user_id=owner.id,
            name="k",
            key_hash=hash_api_key(raw),
            key_prefix=raw[:12],
            scopes=scopes,
            is_active=True,
        )
        db_session.add(key)
        await db_session.commit()
        return raw, key

    async def test_a_viewer_s_key_loses_the_write_scope_it_was_given(
        self, viewer, db_session
    ):
        raw, key = await self._key(db_session, viewer, ["read", "write", "chat"])

        assert await validate_api_key(raw, db_session, required_scope="write") is None
        context = await validate_api_key(raw, db_session, required_scope="read")
        assert context is not None and context.has_scope("chat")

    async def test_a_key_with_no_scopes_is_still_read_only_for_a_viewer(
        self, viewer, db_session
    ):
        _raw, key = await self._key(db_session, viewer, [])

        context = MCPAuthContext(user=viewer, api_key=key, scopes=[])

        assert context.has_scope("read") and not context.has_scope("write")

    async def test_a_user_s_key_keeps_its_scopes(self, test_user, db_session):
        raw, _key = await self._key(db_session, test_user, ["read", "write"])

        assert await validate_api_key(raw, db_session, required_scope="write")


def test_the_role_name_is_the_one_the_directory_assigns():
    from app.services.ldap_service import ldap_service

    assert viewer_access.VIEWER_ROLE == "viewer"
    # map_role's answer for the viewer groups, spelled where it is decided.
    import inspect

    assert '"viewer"' in inspect.getsource(ldap_service.map_role)


def test_a_none_user_is_no_viewer():
    assert not viewer_access.is_viewer(None)
    assert not viewer_access.is_viewer(User(role="user"))
