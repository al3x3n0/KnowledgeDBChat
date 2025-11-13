"""
WebSocket authentication utilities.
"""

from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect, status
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.user import User
from app.services.auth_service import AuthService


async def authenticate_websocket(
    websocket: WebSocket,
    token: Optional[str] = None
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
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
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
    user = await authenticate_websocket(websocket)
    
    if user is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        raise WebSocketDisconnect(code=1008, reason="Authentication required")
    
    return user

