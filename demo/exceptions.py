from fastapi import HTTPException, status
from typing import Any, Dict, Optional


class BaseError(HTTPException):
    """Base exception class for custom API errors."""

    def __init__(
        self,
        status_code: int,
        detail: Optional[str] = None,
        error_code: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        self.error_details = error_details
        super().__init__(status_code=status_code, detail=detail)


class BadRequestError(BaseError):
    """Exception raised when client sends invalid data."""

    def __init__(
        self,
        detail: str = "Invalid request data",
        error_code: str = "BAD_REQUEST",
        error_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code,
            error_details=error_details,
        )


class NotFoundError(BaseError):
    """Exception raised when requested resource is not found."""

    def __init__(
        self,
        detail: str = "Resource not found",
        error_code: str = "NOT_FOUND",
        error_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code=error_code,
            error_details=error_details,
        )


class UnauthorizedError(BaseError):
    """Exception raised when authentication fails."""

    def __init__(
        self,
        detail: str = "Authentication required",
        error_code: str = "UNAUTHORIZED",
        error_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
            error_details=error_details,
        )


class ForbiddenError(BaseError):
    """Exception raised when user lacks permission for the requested action."""

    def __init__(
        self,
        detail: str = "Permission denied",
        error_code: str = "FORBIDDEN",
        error_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code=error_code,
            error_details=error_details,
        )


class ValidationError(BaseError):
    """Exception raised when input validation fails."""

    def __init__(
        self,
        detail: str = "Validation error",
        error_code: str = "VALIDATION_ERROR",
        error_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code,
            error_details=error_details,
        )


class ConflictError(BaseError):
    """Exception raised when there's a conflict with the current state."""

    def __init__(
        self,
        detail: str = "Resource conflict",
        error_code: str = "CONFLICT",
        error_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
            error_details=error_details,
        )


class ServerError(BaseError):
    """Exception raised for server-side errors."""

    def __init__(
        self,
        detail: str = "Internal server error",
        error_code: str = "SERVER_ERROR",
        error_details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code,
            error_details=error_details,
        )


# External API Errors


class ExternalApiError(BaseError):
    """Base exception class for all external API errors."""

    def __init__(
        self,
        detail: str = "External API error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "EXTERNAL_API_ERROR",
        error_details: Optional[Dict[str, Any]] = None,
        source: str = "external_api",
        original_error: Optional[str] = None,
    ):
        self.source = source
        self.original_error = original_error
        # Keep track of the original status code for internal logging
        self.internal_status_code = status_code

        # All external API errors use 500 status code in responses
        # to avoid leaking implementation details
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=error_code,
            error_details=error_details,
        )


class ExternalApiClientError(ExternalApiError):
    """Exception raised when an external API returns a 4xx client error."""

    def __init__(
        self,
        detail: str = "External API client error",
        error_code: str = "EXTERNAL_API_CLIENT_ERROR",
        error_details: Optional[Dict[str, Any]] = None,
        source: str = "external_api",
        original_error: Optional[str] = None,
    ):
        super().__init__(
            detail=detail,
            # This is tracked for internal logging but not used in responses
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code=error_code,
            error_details=error_details,
            source=source,
            original_error=original_error,
        )


class ExternalApiServerError(ExternalApiError):
    """Exception raised when an external API returns a 5xx server error."""

    def __init__(
        self,
        detail: str = "External API server error",
        error_code: str = "EXTERNAL_API_SERVER_ERROR",
        error_details: Optional[Dict[str, Any]] = None,
        source: str = "external_api",
        original_error: Optional[str] = None,
    ):
        super().__init__(
            detail=detail,
            # This is tracked for internal logging but not used in responses
            status_code=status.HTTP_502_BAD_GATEWAY,
            error_code=error_code,
            error_details=error_details,
            source=source,
            original_error=original_error,
        )


class ExternalApiConnectionError(ExternalApiError):
    """Exception raised when there are connectivity issues with an external API."""

    def __init__(
        self,
        detail: str = "Could not connect to external service",
        error_code: str = "EXTERNAL_API_CONNECTION_ERROR",
        error_details: Optional[Dict[str, Any]] = None,
        source: str = "external_api",
        original_error: Optional[str] = None,
    ):
        super().__init__(
            detail=detail,
            # This is tracked for internal logging but not used in responses
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=error_code,
            error_details=error_details,
            source=source,
            original_error=original_error,
        )


class ExternalApiTimeoutError(ExternalApiError):
    """Exception raised when an external API request times out."""

    def __init__(
        self,
        detail: str = "External service timed out",
        error_code: str = "EXTERNAL_API_TIMEOUT_ERROR",
        error_details: Optional[Dict[str, Any]] = None,
        source: str = "external_api",
        original_error: Optional[str] = None,
    ):
        super().__init__(
            detail=detail,
            # This is tracked for internal logging but not used in responses
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code=error_code,
            error_details=error_details,
            source=source,
            original_error=original_error,
        )
