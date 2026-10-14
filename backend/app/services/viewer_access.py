"""What a viewer may do: read, and nothing that changes the shared platform.

The directory's group mapping (``LDAP_VIEWER_GROUP_DNS``) assigned the
``viewer`` role and nothing read it, so a viewer could upload, delete, start
agent jobs and run tools like anyone else. Decided 2026-10-14: a viewer is
read-only. Three entry points, each checked in the one place it passes
through:

* **HTTP**: any non-GET request is refused in the authentication dependency,
  except the writes listed in :data:`VIEWER_MAY` -- signing in and out, a
  viewer's own conversations, settings and notifications, and a few POST
  routes that only compute an answer.
* **Tools** (chat, workflow tool nodes, MCP, custom tools all go through
  ``tool_policy_engine.evaluate_tool_policy``): only tools declared
  ``effects: read``. A user tool or plugin tool has no declaration the
  engine can see, so a viewer may not call one.
* **MCP API keys**: a viewer's key has the ``read`` and ``chat`` scopes and
  no others, whatever was granted when the key was made.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException, status

VIEWER_ROLE = "viewer"

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: The writes a viewer may make, by method and route template. Each is about
#: the viewer's own session, conversation, settings or notifications, or
#: computes an answer and stores nothing shared.
VIEWER_MAY = frozenset(
    {
        # Signing in and out.
        ("POST", "/api/v1/auth/login"),
        ("POST", "/api/v1/auth/logout"),
        ("POST", "/api/v1/auth/refresh"),
        # Their own conversations. A chat turn's tools are checked
        # separately, by the policy engine.
        ("POST", "/api/v1/chat/sessions"),
        ("PATCH", "/api/v1/chat/sessions/{session_id}"),
        ("DELETE", "/api/v1/chat/sessions/{session_id}"),
        ("POST", "/api/v1/chat/sessions/{session_id}/messages"),
        ("PUT", "/api/v1/chat/messages/{message_id}/feedback"),
        ("POST", "/api/v1/agent/chat"),
        ("POST", "/api/v1/agent/conversations/{conversation_id}/messages"),
        ("POST", "/api/v1/export/chat/{session_id}"),
        # Their own settings.
        ("PUT", "/api/v1/users/me/password"),
        ("PUT", "/api/v1/users/me/preferences"),
        ("PUT", "/api/v1/memory/preferences"),
        ("PUT", "/api/v1/notifications/preferences"),
        # Their own notifications.
        ("PUT", "/api/v1/notifications/read-all"),
        ("PUT", "/api/v1/notifications/{notification_id}/read"),
        ("PUT", "/api/v1/notifications/{notification_id}/dismiss"),
        ("DELETE", "/api/v1/notifications/{notification_id}"),
        # Reads that happen to be POSTs.
        ("POST", "/api/v1/memory/search"),
        ("POST", "/api/v1/plugins/me/views/{slug}/{view_id}/data"),
        ("POST", "/api/v1/research/arxiv/translate-query"),
        ("POST", "/api/v1/agent-pipelines/check"),
    }
)

#: MCP scopes a viewer's API key keeps.
VIEWER_SCOPES = frozenset({"read", "chat"})

REFUSAL = (
    "Viewers have read-only access: reading, searching and chat. Ask an "
    "administrator for the user role to change anything."
)


def is_viewer(user: Optional[Any]) -> bool:
    return str(getattr(user, "role", "") or "").strip().lower() == VIEWER_ROLE


def _route_template(request: Any) -> str:
    route = (getattr(request, "scope", None) or {}).get("route")
    return str(getattr(route, "path", "") or getattr(request.url, "path", "") or "")


def refuse_writes(user: Optional[Any], request: Optional[Any]) -> None:
    """A 403 when a viewer makes a request that is not theirs to make."""
    if request is None or not is_viewer(user):
        return
    method = str(getattr(request, "method", "") or "").upper()
    if method in SAFE_METHODS:
        return
    if (method, _route_template(request)) in VIEWER_MAY:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=REFUSAL)


def tool_refusal(user: Optional[Any], tool_name: str, effects: Optional[str]) -> str:
    """Why a viewer may not call this tool, or "" if they may."""
    if not is_viewer(user):
        return ""
    if effects == "read":
        return ""
    if effects is None:
        return (
            f"Viewers may call only tools declared read-only, and '{tool_name}' "
            "declares nothing this check can see."
        )
    return f"Viewers may call only read-only tools; '{tool_name}' changes things."
