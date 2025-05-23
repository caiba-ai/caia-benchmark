import httpx
from datetime import datetime
from typing import Any, Dict, Optional, ClassVar, TypeVar, Self
import asyncio
import logging
import time
import random

from demo.exceptions import (
    ExternalApiClientError,
    ExternalApiServerError,
    ExternalApiConnectionError,
    ExternalApiTimeoutError,
)

logger = logging.getLogger(__name__)

# logging.getLogger("httpx").setLevel(
#     logging.WARNING
# )  # or logging.ERROR for even less output


class RateLimiter:
    def __init__(self, max_requests: int, time_window: float):
        """
        Initialize a sliding window rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        # Store timestamps of requests to implement sliding window
        self.request_timestamps = []

    def acquire(self) -> bool:
        """
        Try to acquire a permission to make a request.

        Returns:
            bool: True if request is allowed, False otherwise
        """
        now = time.time()

        # Remove timestamps that are outside the window
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts <= self.time_window]

        # If we have fewer requests than max_requests in the current window, allow the request
        if len(self.request_timestamps) < self.max_requests:
            self.request_timestamps.append(now)
            return True

        return False

    async def wait(self):
        """Wait until a request is allowed."""
        while True:
            # If we can acquire, return immediately
            if self.acquire():
                return

            # Calculate time until oldest request expires from window
            now = time.time()
            if self.request_timestamps:
                oldest = min(self.request_timestamps)
                # Time until the oldest request exits the window
                wait_time = max(0.1, oldest + self.time_window - now)
                await asyncio.sleep(min(wait_time, 0.5))  # Cap at 0.5s for responsiveness
            else:
                # Should never happen, but just in case
                await asyncio.sleep(0.1)


T = TypeVar("T", bound="BaseHttpClient")


class BaseHttpClient:
    """
    Base asynchronous HTTP client that provides common functionality for making API requests.
    This class can be inherited by specific API clients to ensure consistent behavior.

    All subclasses must use the get_instance() method to obtain an instance. Direct instantiation
    is prevented to enforce the singleton pattern.
    """

    # Class variable to hold instances of derived classes
    _instances: ClassVar[Dict[str, "BaseHttpClient"]] = {}

    # Track if we're in the get_instance method
    _instantiating_from_get_instance = False

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Override __new__ to strictly enforce the singleton pattern through get_instance().

        Any direct instantiation attempt will raise a RuntimeError, forcing users to use
        the get_instance() method instead.
        """
        # Check if this is being called from get_instance
        if not cls._instantiating_from_get_instance:
            raise RuntimeError(
                f"Direct instantiation of {cls.__name__} is not allowed. Use {cls.__name__}.get_instance() instead."
            )

        # Use cls.__name__ as key to support multiple derived classes
        key = cls.__name__

        # If this instance already exists, return it
        if key in cls._instances:
            return cls._instances[key]  # type: ignore

        # If we're here, this is the first instantiation from get_instance
        instance = super().__new__(cls)
        cls._instances[key] = instance
        return instance

    @classmethod
    def get_instance(cls, *args: Any, **kwargs: Any) -> Self:
        """
        Get or create a singleton instance of the HTTP client.

        This is the ONLY way to get an instance of this class.

        Returns:
            BaseHttpClient: The singleton instance
        """
        try:
            # Set flag to allow instantiation
            cls._instantiating_from_get_instance = True

            # Create new instance or get existing one
            instance = cls(*args, **kwargs)
            return instance
        finally:
            # Reset flag to disallow direct instantiation again
            cls._instantiating_from_get_instance = False

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        cache_ttl: int = 60,
        headers: Optional[Dict] = None,
        max_requests_per_min: Optional[int] = None,  # max requests per minute
    ):
        """
        Initialize the BaseHttpClient.

        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_ttl: Cache time-to-live in seconds
            headers: Optional custom headers
        """
        # Skip initialization if this instance is already initialized
        if hasattr(self, "base_url"):
            return

        self.base_url = base_url
        self.client = None
        self.timeout = timeout
        self.max_retries = max_retries
        self._cache = {}
        self._cache_ttl = cache_ttl
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        if headers:
            self.default_headers.update(headers)

        # Initialize rate limiter
        self.rate_limiter = None
        if max_requests_per_min:
            self.rate_limiter = RateLimiter(max_requests_per_min, 60)

    async def __aenter__(self) -> "BaseHttpClient":
        """Setup the client session when used as a context manager."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the client session when exiting the context."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """
        Ensure an active httpx session exists.

        Returns:
            httpx.AsyncClient: The active HTTP client instance
        """
        if self.client is None:
            timeout = httpx.Timeout(connect=10.0, read=self.timeout, write=self.timeout, pool=self.timeout)
            self.client = httpx.AsyncClient(
                timeout=timeout,
                headers=self.default_headers,
                follow_redirects=True,
                http2=True,  # Explicitly enable HTTP/2
            )
        return self.client

    def _validate_response_data(self, data: Optional[Dict[str, Any]]) -> None:
        """
        Validate response data.

        Args:
            data: Response data to validate

        Raises:
            ValueError: If response data is None
        """
        if data is None:
            raise ValueError("Response data is None")

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
        method: str = "GET",
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an asynchronous request to the API with retry logic."""
        # This provides better backward compatibility if the class is extended in a way that doesn't initialize the rate limiter
        if hasattr(self, "rate_limiter") and self.rate_limiter:
            await self.rate_limiter.wait()
        if params is None:
            params = {}

        url = f"{self.base_url}{endpoint}"

        # Generate cache key for GET requests
        if method.upper() == "GET":
            # Convert params to a sorted tuple of items for consistent cache keys
            params_tuple = tuple(sorted(params.items()))
            cache_key = f"{url}:{params_tuple}"

            # Check cache
            if cache_key in self._cache:
                cache_data = self._cache[cache_key]
                current_time = datetime.now().timestamp()

                # Return cached data if not expired
                if current_time - cache_data["timestamp"] < self._cache_ttl:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cache_data["data"]
                else:
                    # Remove expired cache entry
                    del self._cache[cache_key]
        # POST requests are not cached and always sent directly

        retry_count = 0
        last_exception = None

        while retry_count < self.max_retries:
            try:
                client = await self._ensure_client()
                if client is None:
                    raise ExternalApiConnectionError(
                        detail="Failed to initialize HTTP client",
                        error_code="API_CLIENT_INIT_ERROR",
                        source=self.__class__.__name__,
                    )

                if method.upper() == "GET":
                    response = await client.get(url, params=params, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=body, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Check status codes and handle appropriately
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    # Retry on rate limiting or server errors
                    if retry_count < self.max_retries - 1:
                        # Add jitter to prevent retry storms
                        wait_time = (2**retry_count) + random.random()
                        logger.warning(
                            f"Request to {url} failed with status {response.status_code}. "
                            f"Retrying ({retry_count + 1}/{self.max_retries}) in {wait_time:.2f} seconds..."
                        )
                        retry_count += 1
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        # Max retries reached for server errors
                        error_text = response.text
                        logger.error(f"Max retries reached. HTTP error: {response.status_code} - {error_text}")
                        raise ExternalApiServerError(
                            detail="External data provider is experiencing issues",
                            error_code="API_SERVER_ERROR",
                            source=self.__class__.__name__,
                            original_error=error_text,
                        )
                elif 400 <= response.status_code < 500:
                    # Client errors - don't retry
                    error_text = response.text
                    logger.error(f"HTTP client error: {response.status_code} - {error_text}")
                    raise ExternalApiClientError(
                        detail="Invalid request to external data provider",
                        error_code="API_CLIENT_ERROR",
                        source=self.__class__.__name__,
                        original_error=error_text,
                    )
                elif response.status_code not in [200, 201]:
                    # Any other non-200/201 status code
                    error_text = response.text
                    logger.error(f"Unexpected HTTP status: {response.status_code} - {error_text}")
                    raise ExternalApiServerError(
                        detail=f"Unexpected status code {response.status_code}",
                        error_code="API_UNEXPECTED_STATUS",
                        source=self.__class__.__name__,
                        original_error=error_text,
                    )

                # Process successful response
                try:
                    data = response.json()
                    self._validate_response_data(data)

                    # Cache successful GET responses
                    if method.upper() == "GET":
                        self._cache[cache_key] = {
                            "data": data,
                            "timestamp": datetime.now().timestamp(),
                        }
                        await self._cleanup_cache_if_needed()

                    return data

                except ValueError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    raise ExternalApiServerError(
                        detail="Invalid JSON response from server",
                        error_code="API_INVALID_JSON",
                        source=self.__class__.__name__,
                        original_error=str(e),
                    )

            except httpx.TimeoutException as e:
                last_exception = e
                if retry_count < self.max_retries - 1:
                    # Add jitter to prevent retry storms
                    wait_time = (2**retry_count) + random.random()
                    logger.warning(
                        f"Timeout error for {url} (attempt {retry_count + 1}/{self.max_retries}): {str(e)}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    retry_count += 1
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for {url} after timeout errors")
                    raise ExternalApiTimeoutError(
                        detail="External data provider timed out",
                        error_code="API_TIMEOUT_ERROR",
                        source=self.__class__.__name__,
                    )

            except httpx.RequestError as e:
                last_exception = e
                if retry_count < self.max_retries - 1:
                    wait_time = (2**retry_count) + random.random()
                    logger.warning(
                        f"Connection error for {url} (attempt {retry_count + 1}/{self.max_retries}): {str(e)}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    retry_count += 1
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries reached for {url} after connection errors")
                    raise ExternalApiConnectionError(
                        detail="Could not connect to external data provider",
                        error_code="API_CONNECTION_ERROR",
                        source=self.__class__.__name__,
                        original_error=str(e),
                    )

        # If we get here, all retries were exhausted but we didn't raise a specific exception
        if last_exception:
            logger.error(f"All {self.max_retries} retries failed for {url}.")
            raise ExternalApiConnectionError(
                detail="All retry attempts failed",
                error_code="API_MAX_RETRIES_ERROR",
                source=self.__class__.__name__,
                original_error=str(last_exception),
            )

        # This line should never be reached due to the retry logic and exception handling above
        # but we add it to satisfy the type checker
        raise ExternalApiConnectionError(
            detail="Unexpected error occurred",
            error_code="API_UNEXPECTED_ERROR",
            source=self.__class__.__name__,
        )

    async def _cleanup_cache_if_needed(self):
        """Clean up expired cache entries if cache size exceeds 100 items."""
        if len(self._cache) > 100:
            current_time = datetime.now().timestamp()
            self._cache = {k: v for k, v in self._cache.items() if current_time - v["timestamp"] < self._cache_ttl}
