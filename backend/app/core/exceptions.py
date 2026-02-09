"""
Global exception handlers for FastAPI application.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from loguru import logger
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from app.utils.exceptions import (
    KnowledgeDBException,
    DocumentNotFoundError,
    VectorStoreError,
    LLMServiceError,
    AuthenticationError,
    ValidationError as CustomValidationError,
)
from app.utils.formatters import format_error_response


async def knowledge_db_exception_handler(request: Request, exc: KnowledgeDBException) -> JSONResponse:
    """Handle custom KnowledgeDB exceptions."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Map exception types to status codes
    if isinstance(exc, DocumentNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, AuthenticationError):
        status_code = status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, CustomValidationError):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, (VectorStoreError, LLMServiceError)):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    error_response = format_error_response(exc, status_code)
    
    logger.error(f"KnowledgeDB exception: {exc.message}", exc_info=exc)
    
    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error.get("loc", []))
        errors.append({
            "field": field,
            "message": error.get("msg"),
            "type": error.get("type")
        })
    
    error_response = {
        "error": "ValidationError",
        "detail": "Request validation failed",
        "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "errors": errors
    }
    
    logger.warning(f"Validation error: {errors}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions."""
    if isinstance(exc, SQLAlchemyTimeoutError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "ServiceUnavailable",
                "detail": "Database is busy (connection pool exhausted). Please retry in a moment.",
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
            },
            headers={"Retry-After": "3"},
        )

    error_response = {
        "error": exc.__class__.__name__,
        "detail": str(exc),
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
    }
    
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response
    )
