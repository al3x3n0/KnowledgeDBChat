"""
Rate limiting configuration.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Create limiter instance
limiter = Limiter(key_func=get_remote_address)


def get_user_identifier(request: Request) -> str:
    """
    Get user identifier for rate limiting.
    Uses user ID if authenticated, otherwise uses IP address.
    """
    # Try to get user from request state (set by auth middleware)
    if hasattr(request.state, 'user') and request.state.user:
        return f"user:{request.state.user.id}"
    
    # Fall back to IP address
    return get_remote_address(request)


# Rate limit configurations
# Format: "count/period" where period can be second(s), minute(s), hour(s), day(s)

# Default rate limits
DEFAULT_LIMIT = "100/minute"  # 100 requests per minute for authenticated users
ANONYMOUS_LIMIT = "20/minute"  # 20 requests per minute for anonymous users

# Endpoint-specific limits
AUTH_LIMIT = "5/minute"  # Login/register endpoints
CHAT_LIMIT = "30/minute"  # Chat endpoints
UPLOAD_LIMIT = "10/hour"  # File upload endpoints
ADMIN_LIMIT = "200/minute"  # Admin endpoints


def get_rate_limit_for_endpoint(path: str) -> str:
    """
    Get appropriate rate limit for an endpoint based on its path.
    
    Args:
        path: Request path
        
    Returns:
        Rate limit string
    """
    if "/auth/login" in path or "/auth/register" in path:
        return AUTH_LIMIT
    elif "/chat" in path:
        return CHAT_LIMIT
    elif "/upload" in path:
        return UPLOAD_LIMIT
    elif "/admin" in path:
        return ADMIN_LIMIT
    else:
        return DEFAULT_LIMIT

