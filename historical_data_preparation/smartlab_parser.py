"""
Smart-Lab news parser implementation.
Scrapes news from smart-lab.ru for specific tickers.
"""

from typing import List, Optional
from datetime import datetime
from lxml import html
from loguru import logger

from base_parser import BaseNewsParser, ParsedNews
from config import get_config


class SmartLabParser(BaseNewsParser):
    """Parser for Smart-Lab website (smart-lab.ru)."""

    # Russian month names to numbers
    MONTHS = {
        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
        'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
        'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize Smart-Lab parser."""
        config = get_config(config_path)

        # Convert config to dict format expected by BaseNewsParser
        parser_config = {
            'request_delay_seconds': config.smartlab.request_delay_seconds,
            'max_retries': config.smartlab.max_retries,
            'retry_delay_seconds': config.smartlab.retry_delay_seconds,
            'timeout_seconds': config.smartlab.timeout_seconds,
            'headers': config.smartlab.headers
        }

        super().__init__(parser_config, source_name="smartlab")

        self.base_url = config.smartlab.base_url

        logger.info(f"SmartLabParser initialized (base_url={self.base_url})")

    def fetch_news_list_page(self, ticker: str, page_index: int) -> Optional[List[str]]:
        """
        Fetch list of news URLs from a Smart-Lab ticker page.

        Args:
            ticker: Stock ticker (e.g., "SBER", "POSI")
            page_index: Page number (1-indexed)

        Returns:
            List of full news URLs
        """
        url = f"{self.base_url}/forum/news/{ticker}/page{page_index}/"

        logger.debug(f"Fetching news list: {url}")

        response = self._make_request(url)
        if not response:
            return None

        try:
            tree = html.fromstring(response.content)

            # Extract links from news list
            # XPath: //ul[@class="temp_headers temp_headers--have-numbers"]//a/@href
            links = tree.xpath('//ul[@class="temp_headers temp_headers--have-numbers"]//a/@href')

            if not links:
                logger.warning(f"No news links found on {url}")
                return []

            # Convert relative URLs to full URLs
            # /blog/1118401.php -> https://smart-lab.ru/blog/news/1118401.php
            full_links = []
            for link in links:
                # Extract ID from link (e.g., /blog/1118401.php -> 1118401)
                parts = link.split('/')
                if len(parts) >= 3:
                    news_id = parts[2]  # e.g., "1118401.php"
                    full_url = f"{self.base_url}/blog/news/{news_id}"
                    full_links.append(full_url)

            logger.debug(f"Found {len(full_links)} news links")
            return full_links

        except Exception as e:
            logger.error(f"Error parsing news list from {url}: {e}")
            return None

    def fetch_single_news(self, url: str) -> Optional[ParsedNews]:
        """
        Fetch and parse a single Smart-Lab news article.

        Args:
            url: Full URL of the news article

        Returns:
            ParsedNews object or None on error
        """
        response = self._make_request(url)
        if not response:
            return None

        try:
            tree = html.fromstring(response.content)

            # Extract date
            date_elements = tree.xpath('//li[@class="date"]/text()')
            if not date_elements:
                logger.warning(f"No date found in {url}")
                return None

            date_str = date_elements[0].strip()
            published_date = self.parse_date(date_str)

            # Extract title
            title_elements = tree.xpath('//h1[@class="title "]//span/text()')
            if not title_elements:
                logger.warning(f"No title found in {url}")
                return None

            title = title_elements[0].strip()

            # Extract content
            content_parts = tree.xpath('//div[@class="content"]//text()[normalize-space()]')
            content = ' '.join([part.strip() for part in content_parts])

            # Extract tags
            tags = tree.xpath('//ul[@class="tags"]//a/text()')

            # Combine all text
            full_text = f"{title}. {content}. {' '.join(tags)}"

            return ParsedNews(
                url=url,
                title=title,
                text=full_text,
                published_date=published_date.isoformat(),
                published_timestamp=int(published_date.timestamp()),
                source="smartlab"
            )

        except Exception as e:
            logger.error(f"Error parsing news from {url}: {e}")
            return None

    def parse_date(self, date_str: str) -> datetime:
        """
        Parse Smart-Lab date string to datetime.

        Args:
            date_str: Date string like "13 декабря, 2024 15:30"

        Returns:
            datetime object

        Raises:
            ValueError: If date cannot be parsed
        """
        try:
            # Remove comma and split: "13 декабря, 2024 15:30" -> ["13", "декабря", "2024", "15:30"]
            parts = date_str.replace(',', '').split()

            if len(parts) < 4:
                raise ValueError(f"Invalid date format: {date_str}")

            day = int(parts[0])
            month = self.MONTHS.get(parts[1])
            if month is None:
                raise ValueError(f"Unknown month: {parts[1]}")

            year = int(parts[2])
            time_parts = parts[3].split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1])

            return datetime(year, month, day, hour, minute)

        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse date '{date_str}': {e}")


if __name__ == "__main__":
    # Test the parser
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    parser = SmartLabParser()

    print("\n" + "="*50)
    print("TESTING SMARTLAB PARSER:")
    print("="*50)

    # Test fetching news list
    print("\n1. Fetching news list for SBER, page 1...")
    urls = parser.fetch_news_list_page("SBER", 1)

    if urls:
        print(f"   Found {len(urls)} URLs")
        print(f"   First URL: {urls[0]}")

        # Test fetching single news
        print(f"\n2. Fetching single news: {urls[0]}")
        news = parser.fetch_single_news(urls[0])

        if news:
            print(f"   Title: {news.title}")
            print(f"   Date: {news.published_date}")
            print(f"   Text preview: {news.text[:200]}...")
        else:
            print("   Failed to fetch news")
    else:
        print("   Failed to fetch news list")
