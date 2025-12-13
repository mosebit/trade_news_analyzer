"""
News collection pipeline orchestrator.
Coordinates parsing, enrichment, duplicate detection, and storage.
"""

from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import sys
from loguru import logger

from config import get_config
from base_parser import BaseNewsParser, ParsedNews
from smartlab_parser import SmartLabParser
from news_enricher import NewsEnricher
from news_database_chroma import NewsDatabase


class NewsPipeline:
    """
    Main pipeline for collecting and processing news.

    Flow:
    1. Parse news from website
    2. Enrich with LLM
    3. Check for duplicates (RAG + LLM)
    4. Store in database
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with all components."""
        self.config = get_config(config_path)

        # Setup logging
        self._setup_logging()

        # Initialize components
        logger.info("Initializing pipeline components...")

        self.parser = SmartLabParser(config_path)
        self.enricher = NewsEnricher(config_path)
        self.database = NewsDatabase(config_path=config_path)

        logger.info("✓ Pipeline initialized successfully")

    def _setup_logging(self) -> None:
        """Configure logging based on config."""
        # Remove default handler
        logger.remove()

        # Add console handler
        logger.add(
            sys.stderr,
            level=self.config.logging.level,
            format=self.config.logging.format
        )

        # Add file handler if enabled
        if self.config.logging.log_to_file:
            log_path = Path(self.config.logging.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                log_path,
                level=self.config.logging.level,
                format=self.config.logging.format,
                rotation=self.config.logging.rotation,
                retention=self.config.logging.retention
            )

            logger.info(f"Logging to file: {log_path}")

    def process_single_news(
        self,
        news: ParsedNews,
        check_duplicates: bool = True
    ) -> bool:
        """
        Process a single news item through the pipeline.

        Args:
            news: Parsed news item
            check_duplicates: Whether to check for duplicates

        Returns:
            True if news was saved, False otherwise
        """
        logger.debug(f"Processing: {news.title[:60]}...")

        # Enrich with LLM
        enriched_data = self.enricher.enrich_news(news.text)

        if not enriched_data:
            logger.warning(f"Failed to enrich news: {news.url}")
            return False

        # Filter by impact level
        impact = enriched_data.get('level_of_potential_impact_on_price')
        if impact not in ["low", "medium", "high"]:
            logger.debug(f"Skipping news with impact={impact}: {news.title[:60]}...")
            return False

        # Check for duplicates
        if check_duplicates:
            similar_news = self.database.find_similar_news_by_text(
                query_text=enriched_data.get('clean_description'),
                limit=5
            )

            if similar_news:
                logger.debug(f"Found {len(similar_news)} similar news in database")

                # Check with LLM
                candidate_texts = [item['clean_description'] for item in similar_news]
                duplicate_result = self.enricher.find_duplicate(news.text, candidate_texts)

                if duplicate_result:
                    # Found duplicate
                    duplicate_idx = duplicate_result['index']
                    duplicate_news = similar_news[duplicate_idx]

                    logger.info(f"Duplicate detected: {news.url}")
                    logger.info(f"  Original: {duplicate_news['url']}")

                    # Keep the earlier news
                    if news.published_timestamp < duplicate_news['date_timestamp']:
                        logger.info("  New news is earlier, replacing old one")

                        # Delete old news
                        self.database.delete_news(duplicate_news['url'])

                        # Save new news with reference to old URL
                        self.database.save_news(
                            url=news.url,
                            title=news.title,
                            original_text=news.text,
                            enriched_data=enriched_data,
                            published_date=news.published_date,
                            published_timestamp=news.published_timestamp,
                            other_urls=[duplicate_news['url']]
                        )
                        return True
                    else:
                        logger.info("  Existing news is earlier, skipping new one")
                        return False

        # Save news
        result = self.database.save_news(
            url=news.url,
            title=news.title,
            original_text=news.text,
            enriched_data=enriched_data,
            published_date=news.published_date,
            published_timestamp=news.published_timestamp
        )

        return result is not None

    def process_ticker_page(
        self,
        ticker: str,
        page_index: int
    ) -> Dict:
        """
        Process one page of news for a ticker.

        Args:
            ticker: Stock ticker
            page_index: Page number

        Returns:
            Dictionary with statistics:
            - news_count: Total news on page
            - saved_count: News saved to database
            - min_timestamp: Earliest news timestamp
        """
        logger.info(f"Processing {ticker} page {page_index}...")

        # Fetch news batch
        news_items = self.parser.fetch_news_batch(ticker, page_index)

        if not news_items:
            logger.warning(f"No news found for {ticker} page {page_index}")
            return {
                'news_count': 0,
                'saved_count': 0,
                'min_timestamp': None
            }

        # Process each news item
        saved_count = 0
        min_timestamp = float('inf')

        for news in news_items:
            try:
                if self.process_single_news(news):
                    saved_count += 1

                # Track earliest timestamp
                if news.published_timestamp < min_timestamp:
                    min_timestamp = news.published_timestamp

            except Exception as e:
                logger.error(f"Error processing news {news.url}: {e}")
                if not self.config.pipeline.continue_on_error:
                    raise

        logger.info(f"✓ {ticker} page {page_index}: saved {saved_count}/{len(news_items)}")

        return {
            'news_count': len(news_items),
            'saved_count': saved_count,
            'min_timestamp': min_timestamp if min_timestamp != float('inf') else None
        }

    def collect_news_until_date(
        self,
        target_date: str,
        tickers: Optional[List[str]] = None
    ) -> Dict:
        """
        Collect news for tickers until reaching target date.

        Args:
            target_date: ISO format date string (e.g., "2023-01-01T00:00:00")
            tickers: List of tickers (uses config if not provided)

        Returns:
            Statistics dictionary
        """
        # Parse target date
        target_dt = datetime.fromisoformat(target_date)
        target_timestamp = int(target_dt.timestamp())

        # Use config tickers if not provided
        if tickers is None:
            tickers = self.config.get_ticker_list()

        logger.info("="*60)
        logger.info("STARTING NEWS COLLECTION")
        logger.info(f"Target date: {target_date}")
        logger.info(f"Tickers: {tickers}")
        logger.info("="*60)

        # Statistics
        total_stats = {
            'tickers_processed': 0,
            'total_pages': 0,
            'total_news': 0,
            'total_saved': 0,
            'by_ticker': {}
        }

        # Process each ticker
        for ticker in tickers:
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING TICKER: {ticker}")
            logger.info(f"{'='*60}\n")

            ticker_stats = {
                'pages': 0,
                'news_count': 0,
                'saved_count': 0
            }

            page_index = 1

            try:
                while True:
                    # Process page
                    page_stats = self.process_ticker_page(ticker, page_index)

                    # Update statistics
                    ticker_stats['pages'] += 1
                    ticker_stats['news_count'] += page_stats['news_count']
                    ticker_stats['saved_count'] += page_stats['saved_count']

                    # Check if we should continue
                    if page_stats['min_timestamp'] is None:
                        logger.info(f"No more news for {ticker}")
                        break

                    if page_stats['min_timestamp'] < target_timestamp:
                        logger.info(f"Reached target date for {ticker}")
                        break

                    # Print stats periodically
                    if page_index % self.config.pipeline.save_stats_interval == 0:
                        logger.info(f"\n{ticker} Progress: {ticker_stats}")

                    page_index += 1

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                if not self.config.pipeline.continue_on_error:
                    raise

            # Update total statistics
            total_stats['tickers_processed'] += 1
            total_stats['total_pages'] += ticker_stats['pages']
            total_stats['total_news'] += ticker_stats['news_count']
            total_stats['total_saved'] += ticker_stats['saved_count']
            total_stats['by_ticker'][ticker] = ticker_stats

            logger.info(f"\n✓ {ticker} complete: {ticker_stats}")

        # Final statistics
        logger.info("\n" + "="*60)
        logger.info("COLLECTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Tickers processed: {total_stats['tickers_processed']}")
        logger.info(f"Total pages: {total_stats['total_pages']}")
        logger.info(f"Total news found: {total_stats['total_news']}")
        logger.info(f"Total news saved: {total_stats['total_saved']}")
        logger.info("\nBy ticker:")
        for ticker, stats in total_stats['by_ticker'].items():
            logger.info(f"  {ticker}: {stats['saved_count']}/{stats['news_count']} saved")

        # Database statistics
        db_stats = self.database.get_stats()
        logger.info(f"\nDatabase total: {db_stats['total_news']} news")
        logger.info(f"By ticker: {db_stats['by_ticker']}")
        logger.info("="*60)

        return total_stats

    def close(self):
        """Clean up resources."""
        self.database.close()
        logger.info("Pipeline closed")


def main():
    """Main entry point for running the pipeline."""
    # Initialize pipeline
    pipeline = NewsPipeline()

    # Run collection
    try:
        stats = pipeline.collect_news_until_date(
            target_date=pipeline.config.pipeline.target_date
        )

        logger.info("\n✓ Pipeline completed successfully")

    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
