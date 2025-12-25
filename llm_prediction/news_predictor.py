"""
Основной модуль для предсказания изменения цен на основе новостей
"""

import os
import sys
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fix SQLite version issue - must be before any chromadb imports
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass


from historical_data_preparation import ai_enrichers_and_filters
from historical_data_preparation import news_database_chroma
from historical_data_preparation import create_llm_client
from llm_prediction import prediction_prompts


class TimeframePrediction(BaseModel):
    """Предсказание для одного временного интервала"""
    predicted_change_percent: float = Field(description="Ожидаемое процентное изменение цены")
    confidence: float = Field(ge=0, le=1, description="Уверенность в прогнозе от 0 до 1")
    reasoning: str = Field(description="Краткое объяснение прогноза")


class TickerPrediction(BaseModel):
    """Предсказание для одного тикера"""
    ticker: str
    predictions: Dict[str, TimeframePrediction] = Field(
        description="Прогнозы для разных временных интервалов (1h, 3h, 12h, 24h)"
    )


class NewsPredictor:
    """
    Класс для предсказания изменения цен на основе новостей.

    Использует RAG для поиска похожих исторических новостей
    и LLM для генерации прогнозов.
    """

    def __init__(self, db_path: str = "./chroma_db_new", days_for_duplicate_check: int = 7):
        """
        Args:
            db_path: Путь к ChromaDB базе данных
            days_for_duplicate_check: Количество дней для проверки на дубликаты
        """
        self.db = news_database_chroma.NewsDatabase(db_path)
        self.days_for_duplicate_check = days_for_duplicate_check

        # Инициализация LLM клиента (will auto-load from .env)
        self.client = create_llm_client()
        self.model = self.client.get_model_name()

        # Описания тикеров (можно вынести в отдельный конфиг)
        from historical_data_preparation.parser_smart_lab import tickers_descriptions
        self.tickers_descriptions = tickers_descriptions

    def predict_from_raw_news(self, news_text: str, news_date: Optional[str] = None) -> Dict:
        """
        Полный пайплайн предсказания из сырой новости.

        Args:
            news_text: Текст новости
            news_date: Дата публикации новости (ISO формат), если None - используется текущее время

        Returns:
            Словарь с результатами предсказаний для каждого тикера
        """
        print("\n" + "="*70)
        print("STARTING PREDICTION PIPELINE")
        print("="*70)

        # 1. Обогащение новости с помощью LLM
        print("\nStep 1: Enriching news with LLM...")
        enriched_data = ai_enrichers_and_filters.enrich_news_data(
            news_text,
            self.tickers_descriptions
        )

        if not enriched_data:
            print("Error: Failed to enrich news")
            return {"error": "Failed to enrich news"}

        print(f"News enriched:")
        print(f"  - Тикеры: {enriched_data.get('tickers_of_interest', [])}")
        print(f"  - Тональность: {enriched_data.get('sentiment', 'N/A')}")
        print(f"  - Уровень влияния: {enriched_data.get('level_of_potential_impact_on_price', 'N/A')}")

        # Проверка уровня влияния
        if enriched_data.get('level_of_potential_impact_on_price') == 'none':
            print("Warning: News has no impact on price - skipping prediction")
            return {
                "status": "skipped",
                "reason": "News has no impact on price",
                "enriched_data": enriched_data
            }

        tickers = enriched_data.get('tickers_of_interest', [])
        if not tickers:
            print("Warning: News not related to any ticker - skipping")
            return {
                "status": "skipped",
                "reason": "No relevant tickers found",
                "enriched_data": enriched_data
            }

        # 2. Проверка на дубликаты в последних N днях
        print(f"\nStep 2: Checking for duplicates in last {self.days_for_duplicate_check} days...")
        is_duplicate = self._check_recent_duplicates(enriched_data, news_date)

        if is_duplicate:
            print("Warning: Duplicate found - skipping prediction")
            return {
                "status": "skipped",
                "reason": "Duplicate news found in recent history",
                "enriched_data": enriched_data
            }

        print("No duplicates found")

        # 3. Поиск похожих новостей во всей базе и генерация предсказаний
        print(f"\nStep 3: Finding similar news and generating predictions...")

        predictions = {}
        for ticker in tickers:
            print(f"\n  Анализ для тикера {ticker}:")

            # Ищем похожие новости по тикеру
            similar_news = self._find_similar_news_with_prices(
                enriched_data,
                ticker,
                limit=5
            )

            if not similar_news:
                print(f"    Warning: No similar news with prices found for {ticker}")
                predictions[ticker] = {
                    "status": "insufficient_data",
                    "reason": "No similar historical news with price data found"
                }
                continue

            print(f"    Found {len(similar_news)} similar news")

            # Генерируем предсказание
            prediction = self._generate_prediction(
                enriched_data,
                similar_news,
                ticker
            )

            if prediction:
                predictions[ticker] = prediction
                print(f"    Prediction generated")
            else:
                predictions[ticker] = {
                    "status": "prediction_failed",
                    "reason": "LLM failed to generate prediction"
                }
                print(f"    Error generating prediction")

        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70 + "\n")

        return {
            "status": "success",
            "enriched_data": enriched_data,
            "predictions": predictions,
            "news_date": news_date or datetime.now().isoformat()
        }

    def _check_recent_duplicates(self, enriched_data: dict, news_date: Optional[str]) -> bool:
        """
        Проверяет наличие дубликатов в последних N днях.

        Returns:
            True если найден дубликат, False если нет
        """
        # Поиск похожих новостей
        similar_news = self.db.find_similar_news_by_text(
            query_text=enriched_data.get('clean_description'),
            limit=10,
            threshold=0.10
        )

        if not similar_news:
            return False

        # Фильтруем по времени (последние N дней)
        if news_date:
            news_timestamp = int(datetime.fromisoformat(news_date).timestamp())
        else:
            news_timestamp = int(datetime.now().timestamp())

        cutoff_timestamp = news_timestamp - (self.days_for_duplicate_check * 86400)

        recent_similar = [
            n for n in similar_news
            if n.get('date_timestamp', 0) >= cutoff_timestamp
        ]

        if not recent_similar:
            return False

        # Проверяем только очень похожие новости (distance < 0.05)
        very_similar = [n for n in recent_similar if n.get('distance', 1.0) < 0.05]

        if not very_similar:
            print(f"  Найдено {len(recent_similar)} похожих новостей, но distance > 0.05")
            return False

        print(f"  Найдено {len(very_similar)} очень похожих новостей (distance < 0.05)")
        print("  Запускаем LLM проверку на дубликаты...")

        # Используем LLM для финальной проверки
        # Создаем временный текст для проверки (используем clean_description)
        duplicate_check = ai_enrichers_and_filters.find_duplicates(
            enriched_data.get('clean_description'),
            [n.get('clean_description') for n in very_similar]
        )

        return duplicate_check is not None

    def _find_similar_news_with_prices(
        self,
        enriched_data: dict,
        ticker: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Находит похожие новости по тикеру с информацией о ценах.

        Returns:
            Список похожих новостей с ценовыми данными
        """
        # Получаем новости по тикеру с ценами
        ticker_news = self.db.get_news_by_ticker(ticker, limit=1000)

        if not ticker_news:
            return []

        # Фильтруем только те, у которых есть price_changes
        news_with_prices = []
        for news in ticker_news:
            # Получаем полные данные новости
            full_news = self.db.get_news(news['url'])
            if full_news and full_news.get('price_changes'):
                # Проверяем что есть данные по нужному тикеру
                if ticker in full_news['price_changes']:
                    news_with_prices.append(full_news)

        if not news_with_prices:
            return []

        # Находим наиболее похожие, используя поиск по эмбеддингам
        similar = self.db.find_similar_news_by_text(
            query_text=enriched_data.get('clean_description'),
            limit=limit * 3  # Берем больше, чтобы потом отфильтровать
        )

        # Фильтруем: оставляем только те, которые есть в news_with_prices
        urls_with_prices = {n['url'] for n in news_with_prices}
        filtered_similar = [
            n for n in similar
            if n['url'] in urls_with_prices
        ]

        # Дополняем данными о ценах
        result = []
        for news in filtered_similar[:limit]:
            full_news = self.db.get_news(news['url'])
            if full_news:
                result.append(full_news)

        return result

    def _generate_prediction(
        self,
        enriched_data: dict,
        similar_news: List[Dict],
        ticker: str
    ) -> Optional[Dict]:
        """
        Генерирует предсказание изменения цены с помощью LLM.

        Returns:
            Словарь с предсказаниями для разных временных интервалов
        """
        # Создаем промпт
        prompt = prediction_prompts.create_price_prediction_prompt(
            enriched_data,
            similar_news,
            ticker
        )

        try:
            # Вызываем LLM через unified client
            messages = [
                {
                    "role": "system",
                    "content": "Ты эксперт-аналитик российского фондового рынка. Твоя задача - точно прогнозировать изменение цен акций на основе новостей и исторических данных."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response_content = self.client.chat_completion(
                messages=messages,
                temperature=0.3,
                response_format="json_object"
            )

            if not response_content:
                print(f"    Error: empty response from LLM")
                return None

            # Парсим ответ
            prediction_json = json.loads(response_content)

            # Валидируем структуру
            validated_predictions = {}
            for timeframe in ['1h', '3h', '12h', '24h']:
                if timeframe in prediction_json:
                    tf_pred = prediction_json[timeframe]
                    validated_predictions[timeframe] = {
                        "predicted_change_percent": float(tf_pred.get('predicted_change_percent', 0)),
                        "confidence": float(tf_pred.get('confidence', 0)),
                        "reasoning": str(tf_pred.get('reasoning', ''))
                    }

            return {
                "status": "success",
                "predictions": validated_predictions
            }

        except json.JSONDecodeError as e:
            print(f"    JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"    Error generating prediction: {e}")
            return None

    def close(self):
        """Закрывает соединение с базой данных"""
        self.db.close()


if __name__ == "__main__":
    # Пример использования
    predictor = NewsPredictor("./historical_data_preparation/chroma_db_new")

    example_news = """
    Сбербанк объявил о рекордной прибыли за третий квартал 2024 года.
    Чистая прибыль банка составила 400 млрд рублей, что на 25% выше показателей
    прошлого года. Руководство банка повысило прогноз по ROE до 25%.
    """

    result = predictor.predict_from_raw_news(example_news)
    print("\nRESULT:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    predictor.close()


