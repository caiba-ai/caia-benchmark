import logging
from typing import List
from demo.async_http_client import BaseHttpClient
import httpx

logger = logging.getLogger(__name__)

class AsyncJinaClient(BaseHttpClient):
    """
    Asynchronous client for interacting with the Jina API (r.jina.ai).
    This client provides async methods to scrape web content using Jina.
    """

    def __init__(self):
        """Initialize the AsyncJinaClient."""
        super().__init__(
            base_url="https://r.jina.ai",
            timeout=30,
            max_retries=3,
            cache_ttl=300,  # 5 minutes default TTL for API responses
            max_requests_per_min=60,  # Adjust based on API limits
        )

    async def scrape_urls(self, urls: List[str]) -> List[str]:
        """
        Scrape content from multiple URLs using Jina API.

        Args:
            urls: List of URLs to scrape (can be original URLs or already r.jina.ai URLs)

        Returns:
            List of text content (str) for each URL. If an error occurs, the error message is returned as the text.
        """
        import asyncio

        def to_jina_url(url: str) -> str:
            # Already a full r.jina.ai URL
            if url.startswith("https://r.jina.ai/"):
                return url
            # Starts with r.jina.ai/ but missing schema
            if url.startswith("r.jina.ai/"):
                return f"https://{url}"
            # Otherwise, treat as normal URL
            return f"https://r.jina.ai/{url}"

        jina_urls = [to_jina_url(url) for url in urls]

        async def fetch(url: str) -> str:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(url)
                    try:
                        data = resp.json()
                        return data.get("text", str(data))
                    except Exception:
                        return resp.text
            except Exception as e:
                logger.error(f"Error scraping URL {url}: {e}")
                return f"Failed to fetch the content of the URL, Error: {e}"

        results = await asyncio.gather(*[fetch(url) for url in jina_urls])
        return results

if __name__ == "__main__":
    import asyncio
    client = AsyncJinaClient.get_instance()
    print(asyncio.run(client.scrape_urls(["https://cryptorank.io/price/cyber/team"]))[0])