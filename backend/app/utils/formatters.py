"""
Data formatting utilities.
"""

from typing import Any, Dict


def format_error_response(error: Exception, status_code: int = 500) -> Dict[str, Any]:
    """
    Format error response for API.

    Args:
        error: Exception object
        status_code: HTTP status code

    Returns:
        Formatted error response dictionary
    """
    response = {
        "error": error.__class__.__name__,
        "detail": str(error),
        "status_code": status_code,
    }

    # Add additional details for custom exceptions
    if hasattr(error, "detail") and error.detail:
        response["detail"] = error.detail

    if hasattr(error, "field"):
        response["field"] = error.field

    return response
