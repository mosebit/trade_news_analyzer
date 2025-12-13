"""
Example usage of the refactored news collection system.
Run this after installing dependencies and configuring .env and config.yaml
"""

from news_pipeline import NewsPipeline
from news_database_chroma import NewsDatabase
from smartlab_parser import SmartLabParser
from news_enricher import NewsEnricher
from loguru import logger
import sys

# Setup simple logging for examples
logger.remove()
logger.add(sys.stderr, level="INFO")


def example_1_full_pipeline():
    """Example 1: Run full pipeline to collect news."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Full Pipeline")
    print("="*60)

    # Initialize pipeline
    pipeline = NewsPipeline()

    # Collect news for specific tickers and date range
    # This will run until reaching the target date in config.yaml
    stats = pipeline.collect_news_until_date(
        target_date="2025-12-10T00:00:00",  # Override config
        tickers=["SBER"]  # Just one ticker for example
    )

    print(f"\nCollected {stats['total_saved']} news items")

    pipeline.close()


def example_2_test_parser():
    """Example 2: Test parser independently."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Test Parser")
    print("="*60)

    parser = SmartLabParser()

    # Fetch first page for SBER
    print("\nFetching SBER news (page 1)...")
    news_items = parser.fetch_news_batch("SBER", 1)

    print(f"Found {len(news_items)} news items")

    if news_items:
        # Show first news
        news = news_items[0]
        print(f"\nFirst news:")
        print(f"  Title: {news.title}")
        print(f"  Date: {news.published_date}")
        print(f"  URL: {news.url}")
        print(f"  Text preview: {news.text[:200]}...")


def example_3_test_enricher():
    """Example 3: Test news enrichment."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Test Enricher")
    print("="*60)

    enricher = NewsEnricher()

    # Sample news
    sample_news = """
    Сбербанк объявил о рекордной прибыли за третий квартал 2024 года.
    Чистая прибыль банка составила 400 млрд рублей, что на 25% выше
    показателей прошлого года.
    """

    print(f"\nOriginal text:\n{sample_news}")

    # Enrich
    enriched = enricher.enrich_news(sample_news)

    if enriched:
        print(f"\nEnriched data:")
        print(f"  Clean description: {enriched['clean_description']}")
        print(f"  Sentiment: {enriched['sentiment']}")
        print(f"  Tickers: {enriched['tickers_of_interest']}")
        print(f"  Impact: {enriched['level_of_potential_impact_on_price']}")


def example_4_search_database():
    """Example 4: Search existing database."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Search Database")
    print("="*60)

    db = NewsDatabase()

    # Get statistics
    stats = db.get_stats()
    print(f"\nDatabase statistics:")
    print(f"  Total news: {stats['total_news']}")
    print(f"  By ticker: {stats['by_ticker']}")
    print(f"  Hybrid search: {stats['hybrid_search']}")
    print(f"  Reranking: {stats['reranking']}")

    if stats['total_news'] > 0:
        # Search for similar news
        query = "прибыль банка"
        print(f"\nSearching for: '{query}'")

        similar = db.find_similar_news_by_text(
            query_text=query,
            limit=3
        )

        print(f"\nFound {len(similar)} similar news:")
        for i, news in enumerate(similar, 1):
            print(f"\n{i}. Score: {news['score']:.3f}")
            print(f"   {news['clean_description'][:100]}...")
            print(f"   Tickers: {news['tickers']}")
            print(f"   Date: {news['published_date']}")

    db.close()


def example_5_get_ticker_news():
    """Example 5: Get all news for specific ticker."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Get Ticker News")
    print("="*60)

    db = NewsDatabase()

    # Get SBER news
    ticker = "SBER"
    print(f"\nGetting {ticker} news...")

    news_list = db.get_news_by_ticker(ticker, limit=5)

    print(f"\nFound {len(news_list)} {ticker} news items:")
    for i, news in enumerate(news_list, 1):
        print(f"\n{i}. {news['title']}")
        print(f"   Impact: {news['impact_level']}")
        print(f"   Sentiment: {news['sentiment']}")
        print(f"   Date: {news['published_date']}")

    # Get only high-impact news
    print(f"\n\nHigh-impact {ticker} news:")
    high_impact = db.get_news_by_ticker(ticker, limit=5, min_impact="high")

    print(f"Found {len(high_impact)} high-impact news items:")
    for news in high_impact:
        print(f"  - {news['clean_description'][:80]}...")

    db.close()


def example_6_process_single_page():
    """Example 6: Process just one page of news."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Process Single Page")
    print("="*60)

    pipeline = NewsPipeline()

    # Process just page 1 for POSI
    print("\nProcessing POSI page 1...")

    stats = pipeline.process_ticker_page("POSI", page_index=1)

    print(f"\nResults:")
    print(f"  News found: {stats['news_count']}")
    print(f"  News saved: {stats['saved_count']}")
    print(f"  Earliest timestamp: {stats['min_timestamp']}")

    pipeline.close()


def example_7_check_duplicates():
    """Example 7: Test duplicate detection."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Duplicate Detection")
    print("="*60)

    enricher = NewsEnricher()

    # Sample news
    news_to_check = "Сбербанк показал прибыль 400 млрд за Q3"

    # Candidate duplicates
    candidates = [
        "Яндекс запустил новый сервис такси",
        "Сбер объявил прибыль в 400 миллиардов за третий квартал",
        "Роснефть увеличила добычу нефти"
    ]

    print(f"\nChecking: '{news_to_check}'")
    print(f"\nAgainst candidates:")
    for i, cand in enumerate(candidates):
        print(f"  [{i}] {cand}")

    # Check for duplicate
    result = enricher.find_duplicate(news_to_check, candidates)

    if result:
        print(f"\n✓ Duplicate found!")
        print(f"  Index: {result['index']}")
        print(f"  Text: {result['news']}")
    else:
        print(f"\n✗ No duplicate found")


if __name__ == "__main__":
    # Choose which example to run
    import sys

    examples = {
        "1": ("Full Pipeline", example_1_full_pipeline),
        "2": ("Test Parser", example_2_test_parser),
        "3": ("Test Enricher", example_3_test_enricher),
        "4": ("Search Database", example_4_search_database),
        "5": ("Get Ticker News", example_5_get_ticker_news),
        "6": ("Process Single Page", example_6_process_single_page),
        "7": ("Duplicate Detection", example_7_check_duplicates),
    }

    print("\n" + "="*60)
    print("NEWS COLLECTION SYSTEM - EXAMPLES")
    print("="*60)
    print("\nAvailable examples:")
    for num, (name, _) in examples.items():
        print(f"  {num}. {name}")

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\nUsage: python example_usage.py <example_number>")
        print("Example: python example_usage.py 4")
        print("\nRunning example 4 by default...\n")
        choice = "4"

    if choice in examples:
        name, func = examples[choice]
        try:
            func()
        except Exception as e:
            logger.error(f"Error running example: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Invalid choice: {choice}")
        print(f"Available: {', '.join(examples.keys())}")
