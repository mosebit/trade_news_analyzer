# Trade News Analyzer - Historical Data Preparation

Refactored and enhanced news collection system with hybrid search and modular architecture.

## 🎯 Features

### Enhanced Similarity Search
- **Hybrid Search**: Combines BM25 keyword search with vector embeddings
- **Cross-Encoder Reranking**: Improves accuracy by reranking top candidates
- **Adaptive Configuration**: Easy to tune via YAML config

### Clean Architecture
- **Modular Design**: Separated parsers, enrichers, database, and pipeline
- **Extensible**: Easy to add new news sources by extending `BaseNewsParser`
- **Proper Error Handling**: Retry logic, rate limiting, comprehensive logging
- **YAML Configuration**: All settings in one place

## 📁 File Structure

```
historical_data_preparation/
├── config.yaml              # Main configuration file
├── config.py                # Configuration loader
├── base_parser.py           # Abstract parser base class
├── smartlab_parser.py       # Smart-Lab website parser
├── news_enricher.py         # LLM enrichment and duplicate detection
├── news_database_chroma.py  # Enhanced ChromaDB with hybrid search
├── news_pipeline.py         # Main pipeline orchestrator
├── future_price_change.py   # MOEX price data fetcher (unchanged)
├── migrate_to_chromadb.py   # Migration script (legacy)
└── logs/                    # Log files directory
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file:

```env
LLM_MODEL=your-model-name
LLM_API_KEY=your-api-key
BASE_URL=https://api-llm.ml.ptsecurity.ru/v1
```

### 3. Adjust Configuration (Optional)

Edit `config.yaml` to customize:
- Database paths
- Similarity search parameters
- LLM settings
- Tickers to track
- Logging preferences

### 4. Run the Pipeline

```bash
python news_pipeline.py
```

The pipeline will:
1. Parse news from Smart-Lab for configured tickers
2. Enrich each news with LLM analysis
3. Check for duplicates using hybrid search + LLM
4. Store in ChromaDB
5. Continue until reaching the target date (configured in `config.yaml`)

## 🔧 Configuration Guide

### Key Settings in `config.yaml`

#### Similarity Search
```yaml
similarity_search:
  vector_top_k: 20              # Candidates for vector search
  bm25_enabled: true            # Enable keyword search
  bm25_weight: 0.3              # Balance between BM25 and vectors
  reranking_enabled: true       # Use cross-encoder reranking
  reranking_top_k: 5            # Final results after reranking
  duplicate_threshold: 0.75     # Similarity threshold for duplicates
```

**Tips:**
- Higher `bm25_weight` = more importance to exact keywords
- Lower `duplicate_threshold` = stricter duplicate detection
- Disable `reranking_enabled` to save compute (slight accuracy drop)

#### Pipeline Settings
```yaml
pipeline:
  target_date: "2023-01-01T00:00:00"  # Stop when reaching this date
  save_stats_interval: 10              # Print stats every N pages
  continue_on_error: true              # Keep going if one ticker fails
```

## 📊 Usage Examples

### Testing Individual Components

**Test Parser:**
```bash
python smartlab_parser.py
```

**Test Enricher:**
```bash
python news_enricher.py
```

**Test Database:**
```bash
python news_database_chroma.py
```

### Programmatic Usage

```python
from news_pipeline import NewsPipeline

# Initialize pipeline
pipeline = NewsPipeline()

# Collect news for specific tickers and date range
stats = pipeline.collect_news_until_date(
    target_date="2024-01-01T00:00:00",
    tickers=["SBER", "POSI"]
)

print(f"Collected {stats['total_saved']} news items")

# Close resources
pipeline.close()
```

### Querying the Database

```python
from news_database_chroma import NewsDatabase

db = NewsDatabase()

# Find similar news using hybrid search
similar = db.find_similar_news_by_text(
    query_text="Сбербанк объявил рекордную прибыль",
    limit=5
)

for news in similar:
    print(f"Score: {news['score']:.3f} - {news['clean_description']}")

# Get news by ticker
sber_news = db.get_news_by_ticker("SBER", limit=10, min_impact="medium")

# Statistics
stats = db.get_stats()
print(stats)
```

## 🆕 Adding New News Sources

1. Create a new parser class extending `BaseNewsParser`:

```python
from base_parser import BaseNewsParser, ParsedNews

class MyCustomParser(BaseNewsParser):
    def __init__(self):
        config = {'request_delay_seconds': 1.0, ...}
        super().__init__(config, source_name="mycustom")

    def fetch_news_list_page(self, ticker: str, page_index: int):
        # Implement URL list fetching
        pass

    def fetch_single_news(self, url: str):
        # Implement article parsing
        pass

    def parse_date(self, date_str: str):
        # Implement date parsing
        pass
```

2. Update `news_pipeline.py` to use your new parser:

```python
self.parser = MyCustomParser()  # Instead of SmartLabParser()
```

3. Add parser-specific config to `config.yaml`

## 🔍 How Similarity Search Works

1. **Initial Retrieval** (Hybrid):
   - **Vector Search**: Semantic similarity using embeddings
   - **BM25 Search**: Keyword matching
   - Combine scores with weighted average

2. **Reranking** (Optional):
   - Use cross-encoder model for more accurate scoring
   - Much slower but better quality
   - Only rerank top-K candidates

3. **Duplicate Detection**:
   - Find similar news above threshold
   - Ask LLM to verify if truly duplicates
   - Keep earlier news if duplicate found

## 📈 Performance Tuning

### Speed Optimization
- Set `reranking_enabled: false` (loses ~5% accuracy)
- Reduce `vector_top_k` to 10
- Increase `request_delay_seconds` to reduce load

### Accuracy Optimization
- Increase `reranking_top_k` to 10
- Lower `duplicate_threshold` to 0.6
- Increase `bm25_weight` if you have many keyword-specific queries

### Cost Optimization
- Keep current embedding model (small and fast)
- Use reranking only for final duplicate check
- Batch LLM calls if possible

## 🐛 Troubleshooting

### "No module named 'rank_bm25'"
```bash
pip install rank-bm25
```

### "Failed to fetch after 3 attempts"
- Check internet connection
- Increase `retry_delay_seconds` in config
- Website might be blocking you (check User-Agent)

### "BM25 index is empty"
- Database has no documents yet
- BM25 is lazy-loaded on first search
- Check database path in config

### High memory usage
- Reduce `vector_top_k` and `bm25_top_k`
- Process fewer tickers at once
- Clear ChromaDB cache periodically

## 📝 Migration from Old System

If you have data in the old SQLite database:

```bash
python migrate_to_chromadb.py old_news_data.db --chroma-path ./chroma_db_new
```

Then update `database.path` in `config.yaml` to point to the new database.

## 🔮 Future Enhancements

Potential improvements:
- [ ] Add more news sources (Investing.com, RBC, etc.)
- [ ] Implement incremental updates (only new news)
- [ ] Add sentiment analysis visualization
- [ ] Correlation analysis with price movements
- [ ] REST API for querying
- [ ] Multi-threaded parsing for faster collection

## 📄 License

Internal use only.
