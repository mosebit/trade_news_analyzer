"""
Модуль для работы с ChromaDB базой новостей
"""

# Fix SQLite version issue - must be before chromadb import
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
import json
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# Фиксированный список тикеров для статистики
TICKERS = ["SBER", "POSI", "ROSN", "YDEX"]


class NewsDatabase:
    def __init__(self, path: str = "./chroma_db"):
        """Инициализация ChromaDB базы данных."""
        self.client = chromadb.PersistentClient(path=path)
        # self.collection = self.client.get_or_create_collection("news")
        self.collection = self.client.get_or_create_collection("news_cosine", metadata={"hnsw:space": "cosine"})
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        print(f"ChromaDB initialized: {path}")

    def create_embedding(self, enriched_data: Dict):
        """Создание эмбеддинга из обогащенных данных."""
        text = (
            f"news description: {enriched_data.get('clean_description', '')} "
            f"sentiment: {enriched_data.get('sentiment', '')} "
            f"impact: {enriched_data.get('level_of_potential_impact_on_price', '')} "
            f"tickers: {', '.join(enriched_data.get('tickers_of_interest', []))}"
        )
        return self.model.encode(text)

    def create_embedding_from_text(self, text: str):
        """Создание эмбеддинга из произвольного текста."""
        return self.model.encode(text)

    # def check_and_save(self, url: str, title: str, original_text: str,
    #               enriched_data: Dict, published_date: str,
    #               published_timestamp: int, other_urls: list = []) -> Optional[str]:
    #     similar_news


    def save_news(self, url: str, title: str, original_text: str,
                  enriched_data: Dict, published_date: str,
                  published_timestamp: int, other_urls: list = [],
                  price_changes: Optional[Dict] = None) -> Optional[str]:
        """Сохранение новости в базу данных."""
        try:
            # Проверка дубликата
            existing = self.collection.get(ids=[url])
            if existing['ids']:
                print(f"Warning: News with URL {url} already exists")
                return url

            # Создание эмбеддинга
            embedding = self.create_embedding(enriched_data)

            # Подготовка данных
            tickers = enriched_data.get('tickers_of_interest', [])
            impact_level = enriched_data.get('level_of_potential_impact_on_price', 'none')

            # Создаем базовые метаданные
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

            # Добавляем изменения цен, если они есть
            if price_changes:
                metadata['price_changes'] = json.dumps(price_changes, ensure_ascii=False)

            # НОВЫЙ ПОДХОД: создаем поля TICKER_impact только для упомянутых тикеров
            for ticker in tickers:
                metadata[f'{ticker}_impact'] = impact_level

            self.collection.add(
                ids=[url],
                embeddings=[embedding.tolist()],
                documents=[enriched_data.get('clean_description', '')],
                metadatas=[metadata]
            )

            print(f"News saved (url={url}, tickers={tickers})")
            return url

        except Exception as e:
            print(f"Error saving news: {e}")
            return None

    def get_news(self, url: str) -> Optional[Dict]:
        """Получить полную информацию о новости."""
        result = self.collection.get(
            ids=[url],
            include=['metadatas', 'documents']
        )

        if not result['ids']:
            return None

        metadata = result['metadatas'][0]

        news_dict = {
            'url': url,
            'title': metadata.get('title', ''),
            'original_text': metadata.get('original_text', ''),
            'clean_description': result['documents'][0],
            'enriched_data': json.loads(metadata.get('enriched_json', '{}')),
            'tickers': metadata.get('tickers', '').split(',') if metadata.get('tickers') else [],
            'sentiment': metadata.get('sentiment', 'neutral'),
            'published_date': metadata.get('published_date', '')
        }

        # Добавляем price_changes если они есть
        if metadata.get('price_changes'):
            news_dict['price_changes'] = json.loads(metadata.get('price_changes'))

        return news_dict

    def find_similar_news_by_text(self, enriched_data: Optional[Dict] = None, query_text: Optional[str] = None,
                                  limit: int = 5, days_back: Optional[int] = None, threshold: Optional[float] = 0.10) -> List[Dict]:
        if enriched_data:
            query_embedding = self.create_embedding(enriched_data)
        elif query_text:
            query_embedding = self.create_embedding_from_text(query_text)
        else:
            print("⚠ Необходимо предоставить enriched_data или query_text.")
            return []

        # Поиск
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], # ChromaDB ожидает list of lists
            n_results=limit,
            # where=where,
            include=['metadatas', 'documents', 'distances']
        )

        similar = []
        # results['ids'][0] - список ID, results['metadatas'][0] - список метаданных
        for i, result_url in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i]
            if results['distances'][0][i] <= threshold:
                similar.append({
                    'url': result_url,
                    'title': metadata.get('title', ''),
                    'clean_description': results['documents'][0][i],
                    'sentiment': metadata.get('sentiment', ''),
                    'published_datetime': metadata.get('published_datetime', ''),
                    'date_timestamp': metadata.get('timestamp', ''),
                    'distance': results['distances'][0][i]
                })

        print(f"Found {len(similar)} similar news.")
        return similar

    def find_similar_news(self, url: str, limit: int = 5,
                         days_back: Optional[int] = None) -> List[Dict]:
        """Поиск похожих новостей."""
        news = self.collection.get(
            ids=[url],
            include=['embeddings', 'metadatas']
        )

        if not news['ids']:
            return []

        embedding = news['embeddings'][0]

        # Фильтр по времени
        where = None
        if days_back:
            min_ts = news['metadatas'][0]['timestamp'] - (days_back * 86400)
            where = {"timestamp": {"$gte": min_ts}}

        # Поиск
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit + 1,
            where=where,
            include=['metadatas', 'documents', 'distances']
        )

        # Убираем саму новость из результатов
        similar = []
        for i, result_url in enumerate(results['ids'][0]):
            if result_url != url:
                similar.append({
                    'url': result_url,
                    'title': results['metadatas'][0][i].get('title', ''),
                    'clean_description': results['documents'][0][i],
                    'sentiment': results['metadatas'][0][i].get('sentiment', ''),
                    'published_date': results['metadatas'][0][i].get('published_date', ''),
                    'distance': results['distances'][0][i]
                })

        return similar[:limit]

    def get_news_by_ticker(self, ticker: str, limit: int = 100,
                          min_impact: Optional[str] = None) -> List[Dict]:
        """
        Получение новостей по тикеру.

        Args:
            ticker: тикер компании (SBER, POSI, ROSN, YDEX)
            limit: максимальное количество результатов
            min_impact: минимальный уровень влияния ('low', 'medium', 'high')
                       если None - возвращает все новости по тикеру

        Returns:
            список словарей с данными новостей
        """
        # НОВЫЙ ПОДХОД: ищем по полю TICKER_impact
        ticker_field = f"{ticker}_impact"

        if min_impact:
            # Фильтр по уровню влияния
            impact_levels = {
                'low': ['low', 'medium', 'high'],
                'medium': ['medium', 'high'],
                'high': ['high']
            }
            where = {ticker_field: {"$in": impact_levels.get(min_impact, ['low', 'medium', 'high'])}}
        else:
            # Любое влияние - просто проверяем что поле существует
            # В ChromaDB поле существует только если оно было добавлено
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

    def delete_news(self, url: str) -> bool:
        """
        Удаление новости из базы данных по URL.

        Args:
            url: URL новости для удаления

        Returns:
            bool: True если новость успешно удалена, False если новость не найдена или произошла ошибка
        """
        try:
            # Проверяем существование новости
            existing = self.collection.get(ids=[url])

            if not existing['ids']:
                print(f"Warning: News with URL {url} not found in database")
                return False

            # Удаляем новость
            self.collection.delete(ids=[url])

            print(f"News deleted (url={url})")
            return True

        except Exception as e:
            print(f"Error deleting news: {e}")
            return False

    def get_stats(self) -> Dict:
        """Получение статистики по базе данных."""
        total = self.collection.count()

        # Подсчет по тикерам используя TICKER_impact поля
        ticker_counts = {}
        for ticker in TICKERS:
            ticker_field = f"{ticker}_impact"
            # Ищем все записи где есть поле TICKER_impact
            results = self.collection.get(
                where={ticker_field: {"$in": ['none', 'low', 'medium', 'high']}},
                limit=100000  # Большое число для получения всех
            )
            ticker_counts[ticker] = len(results['ids'])

        return {
            'total_news': total,
            'by_ticker': ticker_counts,
            'vector_mode': 'ChromaDB (быстрый)'
        }

    def close(self):
        """Закрытие соединения (для совместимости)."""
        print("ChromaDB does not require explicit closing")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    db = NewsDatabase("./chromadb_v1")

    stats = db.get_stats()
    print("\nСтатистика:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if stats['total_news'] > 0:
        # print(db.find_similar_news_by_text(query_text="Заявление Дональда Трампа о полном отказе Индии от российской нефти оказалось полной неожиданностью для индийских НПЗ, включая Indian Oil Corp и Reliance Industries Ltd, которые рассчитывали лишь на незначительное сокращение объемов. В то же время, Mangalore Refinery and Petrochemicals Ltd не намерена менять свои планы поставок. Несмотря на политическое давление, Bloomberg прогнозирует рост поставок нефти из РФ в Индию в октябре на шесть процентов по сравнению с предыдущим месяцем."))

        print("\n--- Пример новостей по SBER ---")
        sber_all = db.get_news_by_ticker('SBER', limit=10)
        print(f"Найдено: {len(sber_all)}")
        for i, sber_event in enumerate(sber_all):
            print(f"{i}.\t{sber_event.get('clean_description')}")

        # print("\n--- Важные новости по SBER (high impact) ---")
        # sber_high = db.get_news_by_ticker('SBER', limit=5, min_impact='high')
        # print(f"Найдено: {len(sber_high)}")

        # for high_news in sber_high:
        #     print(f"{high_news.get('url')} - {high_news.get('clean_description')}")
