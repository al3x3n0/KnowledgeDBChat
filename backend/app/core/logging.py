"""
Structured logging utilities for the application.
"""

import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from loguru import logger
from fastapi import Request

# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set correlation ID in context.
    
    Args:
        cid: Correlation ID to set. If None, generates a new UUID.
        
    Returns:
        The correlation ID that was set.
    """
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid


def log_request(
    request: Request,
    method: str,
    path: str,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log incoming HTTP request with correlation ID.
    
    Args:
        request: FastAPI request object
        method: HTTP method
        path: Request path
    """
    if correlation_id is None:
        correlation_id = get_correlation_id() or set_correlation_id()
    
    logger.info(
        "Incoming request",
        extra={
            "correlation_id": correlation_id,
            "method": method,
            "path": path,
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
    )


def log_response(
    status_code: int,
    response_time_ms: float,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log HTTP response.
    
    Args:
        status_code: HTTP status code
        response_time_ms: Response time in milliseconds
        correlation_id: Correlation ID (uses context if not provided)
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    
    log_level = "error" if status_code >= 500 else "warning" if status_code >= 400 else "info"
    
    getattr(logger, log_level)(
        "Response sent",
        extra={
            "correlation_id": correlation_id,
            "status_code": status_code,
            "response_time_ms": round(response_time_ms, 2),
        }
    )


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log an error with context.
    
    Args:
        error: Exception object
        context: Additional context dictionary
        correlation_id: Correlation ID (uses context if not provided)
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    
    extra = {
        "correlation_id": correlation_id,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
    }
    
    if context:
        extra.update(context)
    
    logger.error(
        "Error occurred",
        extra=extra,
        exc_info=error
    )


def log_service_call(
    service_name: str,
    method_name: str,
    duration_ms: float,
    success: bool = True,
    correlation_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log a service method call.
    
    Args:
        service_name: Name of the service
        method_name: Name of the method
        duration_ms: Duration in milliseconds
        success: Whether the call was successful
        correlation_id: Correlation ID (uses context if not provided)
        **kwargs: Additional context
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    
    log_level = "error" if not success else "info"
    
    extra = {
        "correlation_id": correlation_id,
        "service": service_name,
        "method": method_name,
        "duration_ms": round(duration_ms, 2),
        "success": success,
    }
    extra.update(kwargs)
    
    getattr(logger, log_level)(
        f"Service call: {service_name}.{method_name}",
        extra=extra
    )
