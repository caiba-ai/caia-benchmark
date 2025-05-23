import logging
import os
from typing import Dict, Any, List
from demo.async_http_client import BaseHttpClient
from demo.exceptions import (
    ExternalApiClientError,
    ExternalApiServerError,
    ExternalApiConnectionError,
    ExternalApiTimeoutError,
)
import httpx

logger = logging.getLogger(__name__)


class AsyncSerpClient(BaseHttpClient):
    """
    Asynchronous client for interacting with the SERP API.
    
    This client provides async methods to perform web searches using
    the SERP API.
    """

    def __init__(self):
        """Initialize the AsyncSerpClient."""
        super().__init__(
            base_url="https://google.serper.dev",
            timeout=30,
            max_retries=3,
            cache_ttl=300,  # 5 minutes default TTL for API responses
            max_requests_per_min=60,  # Adjust based on API limits
        )
        self.scrape_base_url = "https://scrape.serper.dev"
        self.api_key = os.getenv("SERPER_API_KEY", None)
        if not self.api_key:
            raise ValueError("SERPER_API_KEY environment variable is not set")

    async def quick_search(self, query: str, tbs: str = "qdr:w") -> Dict[str, Any]:
        """
        Perform a quick web search using the SERP API.

        Args:
            query: Search query string

        Returns:
            Dict containing search results

        Raises:
            ExternalApiClientError: For 400-level HTTP errors
            ExternalApiServerError: For 500-level HTTP errors
            ExternalApiConnectionError: For connection issues
            ExternalApiTimeoutError: For timeout issues
        """
        try:
            endpoint = "/search"
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }
            params = {
                "q": query,
                "tbs": tbs
            }

            response_data = await self._make_request(endpoint, params, headers=headers)
            return response_data
        except ExternalApiClientError as e:
            logger.error(f"Client error during SERP search: {e}")
            raise
        except ExternalApiServerError as e:
            logger.error(f"Server error during SERP search: {e}")
            raise
        except ExternalApiConnectionError as e:
            logger.error(f"Connection error during SERP search: {e}")
            raise
        except ExternalApiTimeoutError as e:
            logger.error(f"Timeout error during SERP search: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during SERP search: {e}")
            raise ExternalApiServerError(f"Unexpected error: {str(e)}")

    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape content from multiple URLs using SERP API.

        Args:
            urls: List of URLs to scrape

        Returns:
            List of dictionaries containing scraped content and metadata

        Raises:
            ExternalApiClientError: For 400-level HTTP errors
            ExternalApiServerError: For 500-level HTTP errors
            ExternalApiConnectionError: For connection issues
            ExternalApiTimeoutError: For timeout issues
        """
        import asyncio

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        async def scrape_single_url(url: str) -> Dict[str, Any]:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        self.scrape_base_url,
                        json={"url": url},
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error scraping URL {url}: {e.response.text}")
                return {
                    "error": f"{e.response.status_code}: {e.response.text}",
                    "url": url,
                    "text": f"Failed to fetch the content of the URL, Error: {e.response.text}",
                    "metadata": {}
                }
            except Exception as e:
                logger.error(f"Unexpected error scraping URL {url}: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "text": f"Failed to fetch the content of the URL, Error: {e}",
                    "metadata": {}
                }

        results = await asyncio.gather(*[scrape_single_url(url) for url in urls])
        return [r for r in results if r is not None]


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        client = AsyncSerpClient.get_instance()
        try:
            # Search using SERP and save to JSON
            search_response = await client.quick_search("cyber site:cryptorank.io OR site:rootdata.com")
            
            # Save search results to JSON file
            import json
            import os
            
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
            # Save to JSON file with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/search_results_{timestamp}.json"
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(search_response, f, ensure_ascii=False, indent=2)
                
            print(f"Search results saved to: {output_file}")

            # Test URL scraping
            # urls = ["https://wristmart.in/cryptocurrency-news-may-13-2025/"]
            # scrape_results = await client.scrape_urls(urls)
            # print("Scrape results:", scrape_results)
        except (
            ExternalApiClientError,
            ExternalApiServerError,
            ExternalApiConnectionError,
            ExternalApiTimeoutError,
        ) as e:
            print(f"Error: {e}")

    asyncio.run(main())
