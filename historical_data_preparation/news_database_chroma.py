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
from typing import List, Dict, Optional, Literal
# from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import requests
import sys
import os
from dotenv import load_dotenv

# Фиксированный список тикеров для статистики
TICKERS = ["SBER", "POSI", "ROSN", "YDEX"]

@dataclass
class PreparedEvent:
    url: str
    title: str
    clean_description: str
    original_text: str
    tickers: list
    sentiment: str
    impact: Literal["none", "low", "medium", "high"]
    published_date: str
    timestamp: int
    enriched_json: Optional[str] = None
    # embeddings: Optional[list] = None

class NewsDatabase:
    def __init__(self, path: str = "./chroma_db"):
        """Инициализация ChromaDB базы данных."""
        self.client = chromadb.PersistentClient(path=path)
        # self.collection = self.client.get_or_create_collection("news")
        self.collection = self.client.get_or_create_collection("news_cosine", metadata={"hnsw:space": "cosine"})
        # self.model = SentenceTransformer('intfloat/multilingual-e5-small')
        print(f"ChromaDB initialized: {path}")

        # Load .env from project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(project_root, '.env'))
        self.IAM_TOKEN = os.getenv("YANDEX_IAM_TOKEN")
        self.FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

    # def create_embedding(self, enriched_data: Dict):
    #     """Создание эмбеддинга из обогащенных данных."""
    #     text = (
    #         f"news description: {enriched_data.get('clean_description', '')} "
    #         f"sentiment: {enriched_data.get('sentiment', '')} "
    #         f"impact: {enriched_data.get('level_of_potential_impact_on_price', '')} "
    #         f"tickers: {', '.join(enriched_data.get('tickers_of_interest', []))}"
    #     )
    #     return self.model.encode(text)

    def get_embedding(self, text: str, text_type: str = "doc"):
        doc_uri = f"emb://{self.FOLDER_ID}/text-search-doc/latest"
        query_uri = f"emb://{self.FOLDER_ID}/text-search-query/latest"
        embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.IAM_TOKEN}", "x-folder-id": f"{self.FOLDER_ID}"}
        
        query_data = {
            "modelUri": doc_uri if text_type == "doc" else query_uri,
            "text": text,
        }

        request_res = requests.post(embed_url, json=query_data, headers=headers)

        # print(request_res.json())

        return request_res.json()["embedding"]

    def create_embedding_from_text(self, text: str):
        """Создание эмбеддинга из произвольного текста."""
        # return self.model.encode(text)
        return self.get_embedding(text)

    # def check_and_save(self, url: str, title: str, original_text: str,
    #               enriched_data: Dict, published_date: str,
    #               published_timestamp: int, other_urls: list = []) -> Optional[str]:
    #     similar_news
        
    def save_news_new(self, event: PreparedEvent, price_changes: Optional[Dict] = None) -> Optional[str]:
        """Сохранение новости в базу данных."""
        try:
            # Проверка дубликата
            existing = self.collection.get(ids=[event.url])
            if existing['ids']:
                print(f"Warning: News with URL {event.url} already exists")
                return event.url

            # Создание эмбеддинга
            embedding = self.create_embedding_from_text(event.clean_description)
            # embedding = self.create_embedding({
            #     'clean_description': event.clean_description,
            #     'sentiment': event.sentiment,
            #     'level_of_potential_impact_on_price': event.impact,
            #     'tickers_of_interest': event.tickers
            # })

            # Подготовка данных
            tickers = event.tickers
            impact_level = event.impact

            # Создаем базовые метаданные
            metadata = {
                'title': event.title or '',
                'original_text': event.original_text[:3500],
                'tickers': ','.join(tickers),
                'sentiment': event.sentiment,
                'impact': impact_level,
                'published_date': event.published_date or '',
                'timestamp': event.timestamp or 0
            }

            # Добавляем изменения цен, если они есть
            if price_changes:
                metadata['price_changes'] = json.dumps(price_changes, ensure_ascii=False)

            # НОВЫЙ ПОДХОД: создаем поля TICKER_impact только для упомянутых тикеров
            for ticker in tickers:
                if ticker:
                    metadata[f'{ticker}_impact'] = impact_level

            self.collection.add(
                ids=[event.url],
                embeddings=embedding,
                documents=[event.clean_description],
                metadatas=[metadata]
            )

            print(f"News saved (url={event.url}, tickers={tickers})")
            return event.url

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
        
    def find_similar_news_by_event_new(self, event: PreparedEvent, limit: int = 5, days_back: Optional[int] = None, threshold: Optional[float] = 0.10) -> List[Dict]:
        """Finds similar news based on a PreparedEvent object."""
        query_embedding = self.create_embedding_from_text(event.clean_description)
        # query_embedding = self.create_embedding({
        #     'clean_description': event.clean_description,
        #     'sentiment': event.sentiment,
        #     'level_of_potential_impact_on_price': event.impact,
        #     'tickers_of_interest': event.tickers
        # })

        # Поиск
        results = self.collection.query(
            query_embeddings=query_embedding,  # ChromaDB ожидает list of lists
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
        
    def filter_unsaved_urls(self, urls: List[str]) -> List[str]:
        """
        Фильтрует список URL, возвращая только те, которые отсутствуют в базе данных.
        
        Args:
            urls: список URL для проверки
            
        Returns:
            список URL, которые не найдены в базе данных
        """
        if not urls:
            return []
        
        try:
            # Получаем существующие URL одним запросом
            existing = self.collection.get(
                ids=urls,
                include=[]  # не нужны метаданные, только ID
            )
            
            existing_set = set(existing['ids'])
            unsaved = [url for url in urls if url not in existing_set]
            
            print(f"Checked {len(urls)} URLs: {len(unsaved)} new, {len(existing_set)} already in DB")
            return unsaved
            
        except Exception as e:
            print(f"Error filtering URLs: {e}")
            return urls  # в случае ошибки возвращаем все URL


    def close(self):
        """Закрытие соединения (для совместимости)."""
        print("ChromaDB does not require explicit closing")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    db = NewsDatabase("./chroma_db_new")

    stats = db.get_stats()
    print("\nСтатистика:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if stats['total_news'] > 0:

        print("\n--- Пример новостей по SBER ---")
        sber_all = db.get_news_by_ticker('SBER', limit=100)
        print(f"Найдено: {len(sber_all)}")
        for i, sber_event in enumerate(sber_all):
            print(f"{i}.\t{sber_event.get('published_date')}")

        # print("\n--- Важные новости по SBER (high impact) ---")
        # sber_high = db.get_news_by_ticker('SBER', limit=5, min_impact='high')
        # print(f"Найдено: {len(sber_high)}")

        # for high_news in sber_high:
        #     print(f"{high_news.get('url')} - {high_news.get('clean_description')}")
