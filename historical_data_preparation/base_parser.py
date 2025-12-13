"""
Abstract base class for news parsers.
Provides common interface and utilities for different news sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import requests
from loguru import logger


@dataclass
class ParsedNews:
    """Standardized news data structure."""
    url: str
    title: str
    text: str
    published_date: str  # ISO format
    published_timestamp: int
    source: str  # e.g., "smartlab", "investing.com"
    raw_html: Optional[str] = None  # For debugging


class BaseNewsParser(ABC):
    """
    Abstract base class for news parsers.

    Subclasses must implement:
    - fetch_news_list_page(): Get URLs from a listing page
    - fetch_single_news(): Parse individual news article
    - parse_date(): Convert date string to datetime
    """

    def __init__(self, config: Dict, source_name: str):
        """
        Initialize parser with configuration.

        Args:
            config: Configuration dictionary for this parser
            source_name: Name of the news source (e.g., "smartlab")
        """
        self.config = config
        self.source_name = source_name
        self.request_delay = config.get('request_delay_seconds', 1.0)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay_seconds', 5)
        self.timeout = config.get('timeout_seconds', 30)
        self.headers = config.get('headers', {})

        # Track last request time for rate limiting
        self._last_request_time = 0

        logger.info(f"Initialized {source_name} parser")

    def _wait_for_rate_limit(self) -> None:
        """Ensure minimum delay between requests."""
        if self.request_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_delay:
                sleep_time = self.request_delay - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)

    def _make_request(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic.

        Args:
            url: URL to fetch
            **kwargs: Additional arguments for requests.get()

        Returns:
            Response object or None on failure
        """
        self._wait_for_rate_limit()

        # Merge default headers with kwargs
        request_headers = {**self.headers, **kwargs.get('headers', {})}
        kwargs['headers'] = request_headers

        # Set timeout
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Requesting: {url} (attempt {attempt + 1}/{self.max_retries})")

                response = requests.get(url, **kwargs)
                self._last_request_time = time.time()

                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logger.warning(f"404 Not Found: {url}")
                    return None
                elif e.response.status_code == 429:
                    logger.warning(f"Rate limited (429), waiting longer...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {url}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout: {url}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        logger.error(f"Failed to fetch after {self.max_retries} attempts: {url}")
        return None

    @abstractmethod
    def fetch_news_list_page(self, ticker: str, page_index: int) -> Optional[List[str]]:
        """
        Fetch list of news URLs from a listing page.

        Args:
            ticker: Stock ticker symbol
            page_index: Page number (0-indexed or 1-indexed depending on site)

        Returns:
            List of news article URLs, or None on error
        """
        pass

    @abstractmethod
    def fetch_single_news(self, url: str) -> Optional[ParsedNews]:
        """
        Fetch and parse a single news article.

        Args:
            url: URL of the news article

        Returns:
            ParsedNews object or None on error
        """
        pass

    @abstractmethod
    def parse_date(self, date_str: str) -> datetime:
        """
        Parse date string to datetime object.

        Args:
            date_str: Date string from the website

        Returns:
            datetime object

        Raises:
            ValueError: If date cannot be parsed
        """
        pass

    def fetch_news_batch(
        self,
        ticker: str,
        page_index: int
    ) -> List[ParsedNews]:
        """
        Fetch a batch of news from one listing page.

        Args:
            ticker: Stock ticker symbol
            page_index: Page number

        Returns:
            List of ParsedNews objects (may be empty on error)
        """
        # Get list of URLs
        urls = self.fetch_news_list_page(ticker, page_index)
        if not urls:
            logger.warning(f"No URLs found for {ticker} page {page_index}")
            return []

        logger.info(f"Found {len(urls)} news URLs on page {page_index}")

        # Fetch each article
        news_items = []
        for i, url in enumerate(urls, 1):
            try:
                news = self.fetch_single_news(url)
                if news:
                    news_items.append(news)
                    logger.debug(f"  [{i}/{len(urls)}] ✓ {news.title[:60]}...")
                else:
                    logger.warning(f"  [{i}/{len(urls)}] ✗ Failed to parse: {url}")
            except Exception as e:
                logger.error(f"  [{i}/{len(urls)}] ✗ Error parsing {url}: {e}")

        logger.info(f"Successfully parsed {len(news_items)}/{len(urls)} articles")
        return news_items

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(source={self.source_name})>"
