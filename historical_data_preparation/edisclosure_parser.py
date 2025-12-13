"""
E-Disclosure parser implementation.
Scrapes news from e-disclosure.ru for specific companies.
"""

from typing import List, Optional
from datetime import datetime
from lxml import html
from loguru import logger

from base_parser import BaseNewsParser, ParsedNews
from edisclosure_client import EDisclosureClient
from config import get_config


# Mapping of tickers to e-disclosure company IDs
TICKER_TO_COMPANY_IDS = {
    "POSI": [38196, 38538],
    "SBER": [3043],
    "ROSN": [6505],
    "YDEX": [39059]
}


class EDisclosureParser(BaseNewsParser):
    """Parser for E-Disclosure website (e-disclosure.ru)."""

    def __init__(self, config_path: Optional[str] = None, headless: bool = True):
        """
        Initialize E-Disclosure parser.

        Args:
            config_path: Path to config file
            headless: Run browser in headless mode
        """
        config = get_config(config_path)

        # Convert config to dict format expected by BaseNewsParser
        parser_config = {
            'request_delay_seconds': 2.0,  # Longer delay for browser-based scraping
            'max_retries': 2,
            'retry_delay_seconds': 5,
            'timeout_seconds': 60,
            'headers': {}
        }

        super().__init__(parser_config, source_name="edisclosure")

        self.headless = headless
        self.base_url = "https://www.e-disclosure.ru"
        self.client: Optional[EDisclosureClient] = None

        logger.info(f"EDisclosureParser initialized (headless={headless})")

    def _ensure_client(self):
        """Ensure browser client is running."""
        if self.client is None:
            self.client = EDisclosureClient(headless=self.headless)
            self.client._start_browser()

    def fetch_news_list_page(self, ticker: str, page_index: int) -> Optional[List[str]]:
        """
        Fetch list of news URLs for a ticker and year.

        Note: page_index is interpreted as year for e-disclosure.

        Args:
            ticker: Stock ticker (e.g., "SBER", "POSI")
            page_index: Year to fetch (e.g., 2025)

        Returns:
            List of event IDs (pseudoGUIDs)
        """
        self._ensure_client()

        company_ids = TICKER_TO_COMPANY_IDS.get(ticker, [])
        if not company_ids:
            logger.warning(f"No company IDs found for ticker {ticker}")
            return None

        all_event_ids = []

        # Fetch events for each company ID
        for company_id in company_ids:
            logger.debug(f"Fetching events for company {company_id}, year {page_index}")

            events = self.client.get_events_by_year(company_id, page_index)

            if events:
                # Extract event IDs and store with company_id
                event_ids = [
                    f"{event['pseudoGUID']}|{company_id}"
                    for event in events
                ]
                all_event_ids.extend(event_ids)
                logger.debug(f"Got {len(event_ids)} events from company {company_id}")

        if all_event_ids:
            logger.info(f"Found {len(all_event_ids)} total events for {ticker} in {page_index}")
            return all_event_ids
        else:
            logger.info(f"No events found for {ticker} in {page_index}")
            return []

    def fetch_single_news(self, url: str) -> Optional[ParsedNews]:
        """
        Fetch and parse a single e-disclosure event.

        Args:
            url: Event ID in format "pseudoGUID|company_id"

        Returns:
            ParsedNews object or None on error
        """
        self._ensure_client()

        # Parse the combined ID
        try:
            event_id, company_id_str = url.split('|')
            company_id = int(company_id_str)
        except ValueError:
            logger.error(f"Invalid event ID format: {url}")
            return None

        # Get HTML content
        html_content = self.client.get_event_html(event_id, company_id)

        if not html_content:
            return None

        try:
            tree = html.fromstring(html_content)

            # Parse date: '12.12.2025 16:17'
            date_elements = tree.xpath('//div[@class="time left"]/span[@class="date"]/text()')
            if not date_elements:
                logger.warning(f"No date found for event {event_id}")
                return None

            date_str = date_elements[0].strip()
            published_date = self.parse_date(date_str)

            # Parse title
            title_elements = tree.xpath('//h4/text()')
            if not title_elements:
                logger.warning(f"No title found for event {event_id}")
                return None

            title = title_elements[0].strip()

            # Parse content
            content_elements = tree.xpath(
                '//div[@style="word-break: break-word; word-wrap: break-word; white-space: pre-wrap;"]/text()'
            )
            content = content_elements[0].strip() if content_elements else ""

            # Combine text
            full_text = f"TITLE:\n{title}\n\nCONTENT:\n{content}"

            # Create full URL for storage
            full_url = f"{self.base_url}/portal/event.aspx?EventId={event_id}"

            return ParsedNews(
                url=full_url,
                title=title,
                text=full_text,
                published_date=published_date.isoformat(),
                published_timestamp=int(published_date.timestamp()),
                source="edisclosure"
            )

        except Exception as e:
            logger.error(f"Error parsing event {event_id}: {e}")
            return None

    def parse_date(self, date_str: str) -> datetime:
        """
        Parse e-disclosure date string to datetime.

        Args:
            date_str: Date string like "12.12.2025 16:17"

        Returns:
            datetime object

        Raises:
            ValueError: If date cannot be parsed
        """
        try:
            # Split date and time: "12.12.2025 16:17"
            parts = date_str.strip().split()

            if len(parts) < 2:
                raise ValueError(f"Invalid date format: {date_str}")

            # Parse date part: "12.12.2025"
            date_parts = parts[0].split('.')
            day = int(date_parts[0])
            month = int(date_parts[1])
            year = int(date_parts[2])

            # Parse time part: "16:17"
            time_parts = parts[1].split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1])

            return datetime(year, month, day, hour, minute)

        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse date '{date_str}': {e}")

    def fetch_news_by_years(
        self,
        ticker: str,
        start_year: int,
        end_year: int
    ) -> List[ParsedNews]:
        """
        Fetch news for ticker across multiple years.

        Args:
            ticker: Stock ticker
            start_year: Start year (inclusive)
            end_year: End year (inclusive)

        Returns:
            List of ParsedNews objects
        """
        all_news = []

        for year in range(start_year, end_year + 1):
            logger.info(f"Fetching {ticker} news for year {year}")

            # Get event IDs for this year
            event_ids = self.fetch_news_list_page(ticker, year)

            if not event_ids:
                continue

            # Fetch each event
            for event_id in event_ids:
                try:
                    news = self.fetch_single_news(event_id)
                    if news:
                        all_news.append(news)
                except Exception as e:
                    logger.error(f"Error fetching event {event_id}: {e}")

        logger.info(f"Total fetched: {len(all_news)} news items for {ticker}")
        return all_news

    def close(self):
        """Close browser client."""
        if self.client:
            self.client.close()
            self.client = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


if __name__ == "__main__":
    # Test the parser
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    print("\n" + "="*60)
    print("TESTING E-DISCLOSURE PARSER")
    print("="*60)

    parser = EDisclosureParser(headless=True)

    try:
        # Test fetching news for YDEX in 2025
        print("\n1. Fetching news list for YDEX, year 2025...")
        event_ids = parser.fetch_news_list_page("YDEX", 2025)

        if event_ids:
            print(f"✓ Found {len(event_ids)} events")

            # Test fetching first event
            print(f"\n2. Fetching details of first event...")
            news = parser.fetch_single_news(event_ids[0])

            if news:
                print(f"✓ Successfully parsed event:")
                print(f"  Title: {news.title}")
                print(f"  Date: {news.published_date}")
                print(f"  URL: {news.url}")
                print(f"  Text preview: {news.text[:200]}...")
            else:
                print("✗ Failed to parse event")
        else:
            print("✗ No events found")

    finally:
        parser.close()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
