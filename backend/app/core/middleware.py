"""
Custom middleware for the FastAPI application.
"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import log_request, log_response, set_correlation_id, get_correlation_id


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging with correlation IDs."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log with correlation ID."""
        # Set correlation ID
        correlation_id = set_correlation_id()
        
        # Log request
        log_request(request, request.method, request.url.path, correlation_id=correlation_id)
        
        # Process request
        start_time = time.time()
        response = None
        status_code = 500  # Default to 500 in case of exception
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log response
            log_response(status_code, response_time_ms, correlation_id)
            
            # Add correlation ID to response headers (only if response exists)
            if response is not None and hasattr(response, 'headers'):
                response.headers["X-Correlation-ID"] = correlation_id
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add correlation ID if available
        correlation_id = get_correlation_id()
        if correlation_id:
            response.headers["X-Correlation-ID"] = correlation_id
        
        return response
