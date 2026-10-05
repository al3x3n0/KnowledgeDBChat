"""
WebSocket authentication utilities.
"""

from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect, status
from jose import JWTError, jwt
from loguru import logger

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.services.auth_service import AuthService


async def authenticate_websocket(
    websocket: WebSocket, token: Optional[str] = None
) -> Optional[User]:
    """
    Authenticate a WebSocket connection using JWT token.

    Args:
        websocket: WebSocket connection
        token: JWT token (can be in query params or headers)

    Returns:
        Authenticated User object or None if authentication fails
    """
    # Try to get token from query parameters first
    if not token:
        token = websocket.query_params.get("token")

    # Try to get token from headers
    if not token:
        token = websocket.headers.get("Authorization", "").replace("Bearer ", "")

    if not token:
        logger.warning("WebSocket connection attempted without token")
        return None

    try:
        # Decode JWT token
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")

        if user_id is None:
            logger.warning("WebSocket token missing user ID")
            return None

        # Get user from database
        async with AsyncSessionLocal() as db:
            auth_service = AuthService()
            user = await auth_service.get_user_by_id(user_id, db)

            if user is None:
                logger.warning(f"WebSocket user not found: {user_id}")
                return None

            if not user.is_active:
                logger.warning(f"WebSocket connection from inactive user: {user_id}")
                return None

            return user

    except JWTError as e:
        logger.warning(f"WebSocket JWT validation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None


async def authorize_owner(
    websocket: WebSocket, owner_id: object, *, what: str = "Job"
) -> Optional[User]:
    """The caller, if they may watch something owned by ``owner_id``.

    For a stream about one user's job. Three progress streams accepted any
    connection: knowing a job id -- which appears in URLs and logs -- was
    enough to watch someone else's work as it ran. Closes the socket and
    returns None otherwise; the caller only has to stop.

    A stranger is told the thing does not exist rather than that it is
    forbidden, the same answer the HTTP endpoints give: confirming that an id
    is real is itself the thing being withheld.

    The connection must already be accepted.
    """
    user = await authenticate_websocket(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Authentication required")
        return None
    if owner_id is None:
        await websocket.close(code=4004, reason=f"{what} not found")
        return None
    if str(owner_id) != str(user.id) and not user.is_admin():
        await websocket.close(code=4004, reason=f"{what} not found")
        return None
    return user


async def require_websocket_auth(websocket: WebSocket) -> User:
    """
    Require WebSocket authentication, reject connection if not authenticated.

    Args:
        websocket: WebSocket connection

    Returns:
        Authenticated User object

    Raises:
        WebSocketDisconnect: If authentication fails
    """
    # Must accept the WebSocket connection before doing anything else in ASGI
    await websocket.accept()

    user = await authenticate_websocket(websocket)

    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketDisconnect(code=1008, reason="Authentication required")

    return user
