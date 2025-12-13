"""
E-Disclosure browser client using Playwright.
Handles browser automation for bypassing anti-bot protection.
"""

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
import time
import json
from typing import List, Dict, Any, Optional
from loguru import logger


class EDisclosureClient:
    """Client for working with e-disclosure.ru through browser automation."""

    def __init__(self, headless: bool = True):
        """
        Initialize Playwright browser client.

        Args:
            headless: Run browser in headless mode
        """
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.base_url = "https://www.e-disclosure.ru"

        logger.debug(f"EDisclosureClient initialized (headless={headless})")

    def __enter__(self):
        self._start_browser()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _start_browser(self):
        """Start Playwright browser."""
        try:
            self.playwright = sync_playwright().start()

            self.browser = self.playwright.chromium.launch(
                headless=self.headless,
                args=['--disable-blink-features=AutomationControlled']
            )

            self.context = self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
                locale='ru-RU'
            )

            self.page = self.context.new_page()

            # Hide automation
            self.page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            logger.info("✓ Browser started successfully")

        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            raise

    def get_events_by_year(self, company_id: int, year: int) -> List[Dict[str, Any]]:
        """
        Get list of events for company by year.

        Args:
            company_id: Company ID on e-disclosure
            year: Year to fetch

        Returns:
            List of event dictionaries
        """
        if not self.page:
            self._start_browser()

        events_data = []

        # Intercept API response
        def handle_response(response):
            if '/api/events/page' in response.url and response.status == 200:
                try:
                    data = response.json()
                    events_data.extend(data)
                except Exception as e:
                    logger.debug(f"Failed to parse response: {e}")

        self.page.on('response', handle_response)

        # Visit company page
        company_url = f"{self.base_url}/portal/company.aspx?id={company_id}"
        logger.debug(f"Loading company page: {company_url}")

        try:
            self.page.goto(company_url, wait_until='domcontentloaded', timeout=30000)

            # Wait for anti-bot
            time.sleep(3)

            # Wait for network idle
            try:
                self.page.wait_for_load_state('networkidle', timeout=10000)
            except:
                pass

            time.sleep(2)

            # If events not intercepted, try via JavaScript
            if not events_data:
                logger.debug("Trying to fetch events via JavaScript...")
                try:
                    api_url = f"{self.base_url}/api/events/page?companyId={company_id}&year={year}"
                    events_json = self.page.evaluate(f"""
                        fetch('{api_url}', {{
                            headers: {{
                                'X-Requested-With': 'XMLHttpRequest'
                            }}
                        }})
                        .then(r => r.json())
                        .then(data => JSON.stringify(data))
                    """)

                    if events_json:
                        events_data = json.loads(events_json)
                except Exception as e:
                    logger.warning(f"JS request failed: {e}")

            self.page.remove_listener('response', handle_response)

            if events_data:
                logger.info(f"✓ Got {len(events_data)} events for year {year}")
                return events_data
            else:
                logger.warning(f"No events found for year {year}")
                return []

        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []

    def get_event_html(self, event_id: str, company_id: Optional[int] = None) -> Optional[str]:
        """
        Get HTML content of event page.

        Args:
            event_id: Event pseudoGUID
            company_id: Optional company ID to visit first

        Returns:
            HTML content or None on error
        """
        if not self.page:
            self._start_browser()

        try:
            # Visit company page first if provided
            if company_id:
                company_url = f"{self.base_url}/portal/company.aspx?id={company_id}"
                logger.debug(f"Visiting company page: {company_url}")
                self.page.goto(company_url, wait_until='domcontentloaded', timeout=30000)
                time.sleep(2)

            # Go to event page
            event_url = f"{self.base_url}/portal/event.aspx?EventId={event_id}"
            logger.debug(f"Loading event page: {event_url}")
            self.page.goto(event_url, wait_until='domcontentloaded', timeout=30000)

            # Wait for page to load (bypass anti-bot)
            logger.debug("Waiting for page load...")
            max_wait = 15
            start = time.time()

            while time.time() - start < max_wait:
                html = self.page.content()

                # Check if real page loaded
                if 'servicepipe.ru' not in html or len(html) > 10000:
                    if 'id_spinner' not in html or len(html) > 10000:
                        logger.debug(f"✓ Page loaded in {time.time() - start:.1f}s")
                        break

                time.sleep(1)

            # Additional pause
            time.sleep(2)
            html = self.page.content()

            # Verify result
            if 'servicepipe.ru' in html and len(html) < 5000:
                logger.warning("Anti-bot protection may not be bypassed")
            else:
                logger.debug(f"✓ Got HTML ({len(html)} chars)")

            return html

        except Exception as e:
            logger.error(f"Error fetching event HTML: {e}")
            return None

    def close(self):
        """Close browser and cleanup."""
        try:
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            logger.debug("✓ Browser closed")
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")


if __name__ == "__main__":
    # Test the client
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    company_id = 39059  # YDEX
    year = 2025

    print("\n" + "="*60)
    print("TESTING E-DISCLOSURE CLIENT")
    print("="*60)

    with EDisclosureClient(headless=True) as client:
        print(f"\n1. Fetching events for company {company_id}, year {year}...")
        events = client.get_events_by_year(company_id, year)

        if events:
            print(f"✓ Found {len(events)} events")
            print(f"\nFirst event:")
            event = events[0]
            print(f"  Name: {event.get('eventName', 'N/A')}")
            print(f"  Date: {event.get('eventDate', 'N/A')}")
            print(f"  ID: {event.get('pseudoGUID', 'N/A')}")

            print(f"\n2. Fetching event details...")
            event_id = event['pseudoGUID']
            html = client.get_event_html(event_id, company_id)

            if html:
                print(f"✓ Got HTML ({len(html)} chars)")
                if 'servicepipe.ru' not in html and len(html) > 10000:
                    print("✓✓✓ SUCCESS! Page loaded correctly")
                else:
                    print("⚠ Warning: May be blocked by anti-bot")
        else:
            print("✗ No events found")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
