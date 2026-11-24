"""REST operations granted to scoped general-purpose API keys."""

from fastapi import HTTPException, Request

from app.models.api_key import APIKey
from app.models.user import User
from app.services import viewer_access

_WORKFLOW_ACTIONS = frozenset(
    {
        "/workflows/{workflow_id}/execute",
        "/workflows/{workflow_id}/execute/async",
        "/workflows/executions/{execution_id}/cancel",
    }
)


def require_api_key_access(key: APIKey, user: User, request: Request | None) -> None:
    route = request.scope.get("route") if request is not None else None
    path = getattr(route, "path", "").removeprefix("/api/v1")
    family = path.split("/")[1] if path.startswith("/") else ""
    method = request.method if request is not None else ""
    read_only = method in viewer_access.SAFE_METHODS

    # Unmapped operations (including credentials, policies and agentic tool
    # execution) require an unrestricted key. Account role checks still apply.
    scopes = ("admin",)
    if family in {"documents", "document-folders", "upload"}:
        if read_only and family != "upload":
            scopes = ("read", "documents")
        elif not path.startswith("/documents/sources/"):
            scopes = ("write", "documents")
        else:
            scopes = ("documents",)
    elif (family == "searches" and read_only) or path == "/auth/me":
        scopes = ("read",)
    elif family == "chat" or (family == "agent" and read_only):
        scopes = ("chat",)
    elif family == "agent" and path == "/agent/confirm-delete/{document_id}":
        scopes = ("write", "documents")
    elif family == "workflows" and (
        read_only or (method == "POST" and path in _WORKFLOW_ACTIONS)
    ):
        scopes = ("workflows",)

    if viewer_access.is_viewer(user):
        scopes = tuple(
            scope for scope in scopes if scope in viewer_access.VIEWER_SCOPES
        )
    if scopes and key.has_any_scope(list(scopes)):
        return
    raise HTTPException(403, "API key scope does not permit this operation")
