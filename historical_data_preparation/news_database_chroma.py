"""
Enhanced ChromaDB news database with hybrid search and reranking.
Supports BM25 keyword search + vector search + cross-encoder reranking.
"""

# Workaround for systems with old SQLite version
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
import json
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from loguru import logger
import numpy as np
from rank_bm25 import BM25Okapi
from dataclasses import dataclass

from config import get_config


@dataclass
class SearchResult:
    """Single search result with all metadata."""
    url: str
    title: str
    clean_description: str
    sentiment: str
    published_date: str
    date_timestamp: int
    tickers: List[str]
    distance: float
    score: float  # Final hybrid/reranked score


class NewsDatabase:
    """Enhanced news database with hybrid search capabilities."""

    def __init__(self, path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize ChromaDB database with hybrid search support.

        Args:
            path: Override database path from config
            config_path: Path to config file (optional)
        """
        self.config = get_config(config_path)

        # Use provided path or config path
        db_path = path or self.config.database.path

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            self.config.database.collection_name,
            metadata={"hnsw:space": self.config.database.similarity_metric}
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.config.database.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.database.embedding_model)

        # Initialize reranking model if enabled
        self.reranker = None
        if self.config.similarity_search.reranking_enabled:
            logger.info(f"Loading reranking model: {self.config.similarity_search.reranking_model}")
            self.reranker = CrossEncoder(self.config.similarity_search.reranking_model)

        # BM25 index (lazy-loaded on first search)
        self._bm25_index = None
        self._bm25_urls = None

        logger.info(f"✓ NewsDatabase initialized: {db_path}")
        logger.info(f"  - Collection: {self.config.database.collection_name}")
        logger.info(f"  - Documents: {self.collection.count()}")
        logger.info(f"  - Hybrid search: {'enabled' if self.config.similarity_search.bm25_enabled else 'disabled'}")
        logger.info(f"  - Reranking: {'enabled' if self.config.similarity_search.reranking_enabled else 'disabled'}")

    def create_embedding(self, enriched_data: Dict) -> np.ndarray:
        """Create embedding from enriched news data."""
        text = (
            f"news description: {enriched_data.get('clean_description', '')} "
            f"sentiment: {enriched_data.get('sentiment', '')} "
            f"impact: {enriched_data.get('level_of_potential_impact_on_price', '')} "
            f"tickers: {', '.join(enriched_data.get('tickers_of_interest', []))}"
        )
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def create_embedding_from_text(self, text: str) -> np.ndarray:
        """Create embedding from raw text."""
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in database."""
        if not self.config.similarity_search.bm25_enabled:
            return

        logger.info("Building BM25 index...")

        # Get all documents
        results = self.collection.get(
            limit=100000,  # Large number to get all
            include=['documents', 'metadatas']
        )

        if not results['ids']:
            logger.warning("No documents found for BM25 indexing")
            return

        # Tokenize documents (simple word splitting)
        tokenized_corpus = [
            doc.lower().split() for doc in results['documents']
        ]

        # Build BM25 index
        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._bm25_urls = results['ids']

        logger.info(f"✓ BM25 index built with {len(self._bm25_urls)} documents")

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Perform BM25 keyword search.

        Returns:
            List of (url, score) tuples
        """
        if self._bm25_index is None:
            self._build_bm25_index()

        if self._bm25_index is None:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Return (url, score) pairs
        return [(self._bm25_urls[idx], scores[idx]) for idx in top_indices]

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Perform vector similarity search.

        Returns:
            List of result dictionaries with metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )

        search_results = []
        for i, url in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            search_results.append({
                'url': url,
                'title': metadata.get('title', ''),
                'clean_description': results['documents'][0][i],
                'sentiment': metadata.get('sentiment', ''),
                'published_date': metadata.get('published_date', ''),
                'date_timestamp': metadata.get('timestamp', 0),
                'tickers': metadata.get('tickers', '').split(',') if metadata.get('tickers') else [],
                'distance': results['distances'][0][i],
                'score': 1.0 - results['distances'][0][i]  # Convert distance to similarity
            })

        return search_results

    def _hybrid_search(self, query_text: str, query_embedding: np.ndarray) -> List[Dict]:
        """
        Combine BM25 and vector search results.

        Args:
            query_text: Text query for BM25
            query_embedding: Vector embedding for similarity search

        Returns:
            List of hybrid search results
        """
        # Get BM25 results
        bm25_results = {}
        if self.config.similarity_search.bm25_enabled:
            bm25_hits = self._bm25_search(query_text, self.config.similarity_search.bm25_top_k)
            # Normalize BM25 scores to [0, 1]
            if bm25_hits:
                max_bm25_score = max(score for _, score in bm25_hits)
                if max_bm25_score > 0:
                    bm25_results = {
                        url: score / max_bm25_score
                        for url, score in bm25_hits
                    }

        # Get vector search results
        vector_results = self._vector_search(
            query_embedding,
            self.config.similarity_search.vector_top_k
        )

        # Combine results
        combined = {}
        for result in vector_results:
            url = result['url']
            vector_score = result['score']
            bm25_score = bm25_results.get(url, 0.0)

            # Weighted combination
            bm25_weight = self.config.similarity_search.bm25_weight
            hybrid_score = (1 - bm25_weight) * vector_score + bm25_weight * bm25_score

            result['score'] = hybrid_score
            combined[url] = result

        # Add BM25-only results that weren't in vector search
        for url, bm25_score in bm25_results.items():
            if url not in combined:
                # Fetch full data for this document
                doc_data = self.get_news(url)
                if doc_data:
                    combined[url] = {
                        'url': url,
                        'title': doc_data['title'],
                        'clean_description': doc_data['clean_description'],
                        'sentiment': doc_data['sentiment'],
                        'published_date': doc_data['published_date'],
                        'date_timestamp': doc_data.get('date_timestamp', 0),
                        'tickers': doc_data['tickers'],
                        'distance': 1.0,  # Max distance
                        'score': self.config.similarity_search.bm25_weight * bm25_score
                    }

        # Sort by hybrid score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return sorted_results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using cross-encoder model.

        Args:
            query: Original query text
            results: List of search results

        Returns:
            Reranked list of results
        """
        if not self.reranker or not results:
            return results

        # Prepare pairs for reranking
        pairs = [
            [query, result['clean_description']]
            for result in results
        ]

        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Update scores and sort
        for i, result in enumerate(results):
            result['score'] = float(rerank_scores[i])

        reranked = sorted(results, key=lambda x: x['score'], reverse=True)

        return reranked[:self.config.similarity_search.reranking_top_k]

    def find_similar_news_by_text(
        self,
        enriched_data: Optional[Dict] = None,
        query_text: Optional[str] = None,
        limit: Optional[int] = None,
        days_back: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Find similar news using hybrid search and reranking.

        Args:
            enriched_data: Enriched news data (alternative to query_text)
            query_text: Raw text query
            limit: Maximum number of results (uses config if not provided)
            days_back: Not implemented yet
            threshold: Minimum similarity threshold

        Returns:
            List of similar news items
        """
        # Determine query text and embedding
        if enriched_data:
            query_text = enriched_data.get('clean_description', '')
            query_embedding = self.create_embedding(enriched_data)
        elif query_text:
            query_embedding = self.create_embedding_from_text(query_text)
        else:
            logger.warning("Neither enriched_data nor query_text provided")
            return []

        # Use configured threshold if not provided
        if threshold is None:
            threshold = self.config.similarity_search.duplicate_threshold

        # Perform hybrid search
        if self.config.similarity_search.bm25_enabled:
            results = self._hybrid_search(query_text, query_embedding)
        else:
            # Vector-only search
            results = self._vector_search(
                query_embedding,
                self.config.similarity_search.vector_top_k
            )

        # Apply reranking if enabled
        if self.config.similarity_search.reranking_enabled and results:
            logger.debug(f"Reranking {len(results)} results")
            results = self._rerank_results(query_text, results)

        # Apply threshold filter
        filtered_results = [
            r for r in results
            if r['score'] >= threshold
        ]

        # Apply limit
        if limit:
            filtered_results = filtered_results[:limit]

        logger.info(f"Found {len(filtered_results)} similar news (threshold={threshold:.2f})")

        return filtered_results

    def save_news(
        self,
        url: str,
        title: str,
        original_text: str,
        enriched_data: Dict,
        published_date: str,
        published_timestamp: int,
        other_urls: List[str] = []
    ) -> Optional[str]:
        """Save news to database."""
        try:
            # Check for existing
            existing = self.collection.get(ids=[url])
            if existing['ids']:
                logger.warning(f"News with URL {url} already exists")
                return url

            # Create embedding
            embedding = self.create_embedding(enriched_data)

            # Prepare metadata
            tickers = enriched_data.get('tickers_of_interest', [])
            impact_level = enriched_data.get('level_of_potential_impact_on_price', 'none')

            metadata = {
                'title': title or '',
                'original_text': (original_text or '')[:3500],
                'tickers': ','.join(tickers),
                'sentiment': enriched_data.get('sentiment', 'neutral'),
                'impact': impact_level,
                'published_date': published_date or '',
                'timestamp': published_timestamp or 0,
                'enriched_json': json.dumps(enriched_data, ensure_ascii=False)
            }

            if other_urls:
                metadata['other_urls'] = json.dumps(other_urls)

            # Add ticker-specific impact fields
            for ticker in tickers:
                metadata[f'{ticker}_impact'] = impact_level

            # Save to ChromaDB
            self.collection.add(
                ids=[url],
                embeddings=[embedding.tolist()],
                documents=[enriched_data.get('clean_description', '')],
                metadatas=[metadata]
            )

            # Invalidate BM25 index (will be rebuilt on next search)
            self._bm25_index = None

            logger.info(f"✓ News saved: {url} (tickers={tickers})")
            return url

        except Exception as e:
            logger.error(f"✗ Error saving news: {e}")
            return None

    def get_news(self, url: str) -> Optional[Dict]:
        """Get full news information by URL."""
        result = self.collection.get(
            ids=[url],
            include=['metadatas', 'documents']
        )

        if not result['ids']:
            return None

        metadata = result['metadatas'][0]

        return {
            'url': url,
            'title': metadata.get('title', ''),
            'original_text': metadata.get('original_text', ''),
            'clean_description': result['documents'][0],
            'enriched_data': json.loads(metadata.get('enriched_json', '{}')),
            'tickers': metadata.get('tickers', '').split(',') if metadata.get('tickers') else [],
            'sentiment': metadata.get('sentiment', 'neutral'),
            'published_date': metadata.get('published_date', ''),
            'date_timestamp': metadata.get('timestamp', 0)
        }

    def delete_news(self, url: str) -> bool:
        """Delete news from database."""
        try:
            existing = self.collection.get(ids=[url])
            if not existing['ids']:
                logger.warning(f"News with URL {url} not found")
                return False

            self.collection.delete(ids=[url])

            # Invalidate BM25 index
            self._bm25_index = None

            logger.info(f"✓ News deleted: {url}")
            return True

        except Exception as e:
            logger.error(f"✗ Error deleting news: {e}")
            return False

    def get_news_by_ticker(
        self,
        ticker: str,
        limit: int = 100,
        min_impact: Optional[str] = None
    ) -> List[Dict]:
        """Get news by ticker with optional impact filter."""
        ticker_field = f"{ticker}_impact"

        if min_impact:
            impact_levels = {
                'low': ['low', 'medium', 'high'],
                'medium': ['medium', 'high'],
                'high': ['high']
            }
            where = {ticker_field: {"$in": impact_levels.get(min_impact, ['low', 'medium', 'high'])}}
        else:
            where = {ticker_field: {"$in": ['none', 'low', 'medium', 'high']}}

        results = self.collection.get(
            where=where,
            limit=limit,
            include=['metadatas', 'documents']
        )

        news_list = []
        for i, url in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            news_list.append({
                'url': url,
                'title': metadata.get('title', ''),
                'clean_description': results['documents'][i],
                'sentiment': metadata.get('sentiment', ''),
                'impact_level': metadata.get('impact', ''),
                'published_date': metadata.get('published_date', ''),
                'tickers': metadata.get('tickers', '').split(',') if metadata.get('tickers') else []
            })

        return news_list

    def get_stats(self) -> Dict:
        """Get database statistics."""
        total = self.collection.count()

        # Count by tickers
        ticker_counts = {}
        for ticker in self.config.get_ticker_list():
            ticker_field = f"{ticker}_impact"
            results = self.collection.get(
                where={ticker_field: {"$in": ['none', 'low', 'medium', 'high']}},
                limit=100000
            )
            ticker_counts[ticker] = len(results['ids'])

        return {
            'total_news': total,
            'by_ticker': ticker_counts,
            'hybrid_search': self.config.similarity_search.bm25_enabled,
            'reranking': self.config.similarity_search.reranking_enabled
        }

    def close(self):
        """Close database connection (for compatibility)."""
        logger.info("✓ Database closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test the enhanced database
    from loguru import logger
    import sys

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    db = NewsDatabase()

    # Show stats
    stats = db.get_stats()
    print("\n" + "="*50)
    print("DATABASE STATISTICS:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("="*50)

    if stats['total_news'] > 0:
        # Test hybrid search
        print("\n" + "="*50)
        print("TESTING HYBRID SEARCH:")
        print("="*50)

        test_query = "Сбербанк объявил о рекордной прибыли"
        print(f"\nQuery: {test_query}\n")

        similar = db.find_similar_news_by_text(query_text=test_query, limit=5)

        for i, news in enumerate(similar, 1):
            print(f"{i}. Score: {news['score']:.4f}")
            print(f"   {news['clean_description'][:150]}...")
            print(f"   Tickers: {news['tickers']}")
            print()
