"""
Global exception handlers for FastAPI application.
"""

from uuid import uuid4

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from app.core.config import settings
from app.utils.exceptions import (
    AuthenticationError,
    DocumentNotFoundError,
    KnowledgeDBException,
    LLMServiceError,
)
from app.utils.exceptions import ValidationError as CustomValidationError
from app.utils.exceptions import VectorStoreError
from app.utils.formatters import format_error_response

GENERIC_ERROR = "Internal server error"


def internal_error_detail(exc: BaseException, what: str = GENERIC_ERROR) -> str:
    """What a 500 response may say about the exception behind it.

    ``what`` names the operation ("Failed to render the diagram"). With
    ``EXPOSE_ERROR_DETAILS`` the exception's text follows it, as every such
    response used to carry unconditionally -- SQL, file paths and upstream
    replies included, to any caller. Otherwise the client gets a reference,
    and the exception is logged under it so the report can be matched.
    """
    if settings.EXPOSE_ERROR_DETAILS:
        return str(exc) if what == GENERIC_ERROR else f"{what}: {exc}"
    reference = uuid4().hex[:8]
    logger.error(f"[error {reference}] {what}: {exc!r}")
    return f"{what} (reference {reference})"


#: Libraries whose connection and timeout errors mean "that service is away".
_TRANSPORT_MODULES = {"urllib3", "httpx", "httpcore", "aiohttp", "redis", "requests"}
_TRANSPORT_WORDS = ("Connect", "Timeout", "MaxRetry", "Transport", "Network")


def is_dependency_failure(exc: BaseException) -> bool:
    """Whether `exc` says something this server depends on did not answer.

    The model, the vector store, object storage: when one is away the honest
    answer is 503, "try again", not 500, "this server is broken". The typed
    errors already map to 503 in the handler below, but a route that wraps
    its body in `except Exception` and raises its own 500 never lets them
    get that far. Causes are followed, since services wrap what they catch.
    """
    seen = 0
    current: BaseException | None = exc
    while current is not None and seen < 8:
        if isinstance(current, (LLMServiceError, VectorStoreError)):
            return True
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        kind = type(current)
        if kind.__module__.split(".")[0] in _TRANSPORT_MODULES and any(
            word in kind.__name__ for word in _TRANSPORT_WORDS
        ):
            return True
        current = current.__cause__ or current.__context__
        seen += 1
    return False


def failure_status(exc: BaseException) -> int:
    """503 when a dependency did not answer, 500 for a fault of our own."""
    return (
        status.HTTP_503_SERVICE_UNAVAILABLE
        if is_dependency_failure(exc)
        else status.HTTP_500_INTERNAL_SERVER_ERROR
    )


async def knowledge_db_exception_handler(
    request: Request, exc: KnowledgeDBException
) -> JSONResponse:
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

    return JSONResponse(status_code=status_code, content=error_response)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error.get("loc", []))
        errors.append(
            {"field": field, "message": error.get("msg"), "type": error.get("type")}
        )

    error_response = {
        "error": "ValidationError",
        "detail": "Request validation failed",
        "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "errors": errors,
    }

    logger.warning(f"Validation error: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_response
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

    logger.exception(f"Unhandled exception: {exc}")
    error_response = {
        "error": exc.__class__.__name__
        if settings.EXPOSE_ERROR_DETAILS
        else "InternalServerError",
        "detail": internal_error_detail(exc),
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response
    )
