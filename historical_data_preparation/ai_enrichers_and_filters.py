from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import json
import requests

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from .llm_client import create_llm_client

from . import llm_client
from .news_database_chroma import PreparedEvent

client = llm_client.create_llm_client()
model = client.get_model_name()

TICKERS = ["SBER", "POSI", "ROSN", "YDEX"]

class EnrichedEventData(BaseModel):
    clean_description: str = Field(
        description="Краткое описание новости без рекламы, воды и лишних деталей"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Общий тон новости относительно упомянутых активов"
    )
    tickers_of_interest: List[Literal["SBER", "POSI", "ROSN", "YDEX"]] = Field(
        default_factory=list,
        description="Список тикеров, к которым относится новость"
    )
    level_of_potential_impact_on_price: Literal["none", "low", "medium", "high"] = Field(
        description="Потенциальное влияние новости на цену актива"
    )

def enrich_news_data(event_description: str, tickers_data: dict):
    """
    Обогащает данные о новости с помощью LLM анализа.

    Args:
        event_description: Текст новости для анализа
        tickers_data: Словарь с данными о тикерах (описание активов, текущие цены и т.д.)
        model: Название модели LLM для использования

    Returns:
        dict: Обогащенные данные о новости или None в случае ошибки
    """

    # Формируем контекст о тикерах для более точного анализа
    tickers_context = "\n".join([
        f"- {ticker}: {data.get('description', 'N/A')}"
        for ticker, data in tickers_data.items()
    ])

    # Улучшенный промпт на русском языке для более точного анализа
    prompt = f"""
        Проанализируй следующую новость и предоставь структурированный ответ.

        НОВОСТЬ:
        {event_description}

        ДОСТУПНЫЕ АКТИВЫ ДЛЯ АНАЛИЗА:
        {tickers_context}

        ЗАДАЧИ:
        1. Создай краткое описание новости (clean_description) - убери рекламу, воду, лишние детали. Оставь только суть, НО ничего полезного НЕ удаляй!
        2. Определи общий тон новости (sentiment): positive (позитивная), negative (негативная) или neutral (нейтральная).
        3. Определи, к каким тикерам из списка [{', '.join(TICKERS)}] относится эта новость (tickers_of_interest). Список может быть пустым, если новость не касается ни одного из активов.
        4. Оцени потенциальное влияние новости на цену активов (level_of_potential_impact_on_price):
        - none: новость не влияет на цену
        - low: минимальное влияние (обычные события)
        - medium: заметное влияние (важные корпоративные события)
        - high: сильное влияние (критические события, смена руководства, крупные сделки)

        Верни ответ строго в JSON формате со следующей структурой:
        {{
        "clean_description": "краткое описание новости",
        "sentiment": "positive/negative/neutral",
        "tickers_of_interest": ["TICKER1", "TICKER2"],
        "level_of_potential_impact_on_price": "none/low/medium/high"
        }}
    """

    try:
        # Вызов LLM с запросом структурированного ответа
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт финансовый аналитик, специализирующийся на российском фондовом рынке. Ты анализируешь новости и определяешь их влияние на котировки акций."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = client.chat_completion(
            messages=messages,
            temperature=0.3,
            response_format="json_object"
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM")
            return None

        # Парсинг ответа с использованием Pydantic модели
        result = EnrichedEventData.model_validate_json(response_content)

        return result.model_dump()

    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON ответа от LLM: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
    except Exception as e:
        print(f"Ошибка при обработке ответа LLM: {e}")
        return None

class DuplicateCheckResult(BaseModel):
    is_duplicate: bool = Field(
        description="Является ли проверяемая новость дубликатом одной из новостей в списке"
    )
    duplicate_index: int | None = Field(
        default=None,
        description="Индекс новости из списка, которая является дубликатом (если is_duplicate=True)"
    )
    reasoning: str = Field(
        description="Краткое объяснение решения"
    )

def find_duplicates(main_news: str, news_list: List[str]):
    """
    Проверяет, является ли основная новость дубликатом одной из новостей в списке.

    Args:
        main_news: Текст основной новости для проверки
        news_list: Список текстовых описаний новостей для сравнения

    Returns:
        dict: {"index": int, "news": str} - найденный дубликат
        или None если дубликат не найден
    """

    # Формируем список новостей с индексами для промпта
    indexed_news = "\n".join([
        f"[{i}] {news}"
        for i, news in enumerate(news_list)
    ])

    prompt = f"""
Определи, является ли ПРОВЕРЯЕМАЯ НОВОСТЬ дубликатом одной из новостей в СПИСКЕ НОВОСТЕЙ.

ПРОВЕРЯЕМАЯ НОВОСТЬ:
{main_news}

СПИСОК НОВОСТЕЙ ДЛЯ СРАВНЕНИЯ:
{indexed_news}

КРИТЕРИИ ДУБЛИКАТА:
- Новости описывают ОДНО И ТО ЖЕ событие
- События происходят в ОДИН И ТОТ ЖЕ момент времени (или очень близко)
- Суть события одинакова, даже если формулировки отличаются

ПРИМЕРЫ ДУБЛИКАТОВ:
- "Сбербанк объявил прибыль 400 млрд за Q3 2024" и "Сбер показал рекордную прибыль в третьем квартале - 400 миллиардов"
- "Роснефть подписала контракт с Китаем на 5 млрд долларов" и "Роснефть заключила сделку с китайской стороной на $5B"

ПРИМЕРЫ НЕ ДУБЛИКАТОВ:
- "Сбербанк объявил прибыль за Q3" и "Сбербанк объявил прибыль за Q2" (разные периоды)
- "Яндекс запустил новый сервис" и "Яндекс обновил существующий сервис" (разные события)
- "Роснефть повысила дивиденды" и "Роснефть сменила CEO" (разные события одной компании)

ЗАДАЧА:
Проверь, есть ли в списке новостей дубликат проверяемой новости.
Если дубликат найден - укажи его индекс (число в квадратных скобках).
Если несколько похожих - выбери НАИБОЛЕЕ похожую.

ФОРМАТ ОТВЕТА:
Верни ответ строго в JSON формате со следующей структурой:
{{
    "is_duplicate": true/false,
    "duplicate_index": число или null (индекс новости из списка, если дубликат найден),
    "reasoning": "краткое объяснение решения"
}}

ВАЖНО:
- duplicate_index должен быть null, если is_duplicate = false
- duplicate_index должен быть числом от 0 до {len(news_list)-1}, если is_duplicate = true
- reasoning должен кратко объяснять, почему новости являются или не являются дубликатами
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу новостного контента. Твоя задача - точно определять, описывают ли две новости одно и то же событие."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = client.chat_completion(
            messages=messages,
            temperature=0.1,
            response_format="json_object"
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM при проверке дубликатов")
            return None

        # Парсинг ответа
        result = DuplicateCheckResult.model_validate_json(response_content)

        # Если найден дубликат, возвращаем его
        if result.is_duplicate and result.duplicate_index is not None:
            if 0 <= result.duplicate_index < len(news_list):
                return {
                    "index": int(result.duplicate_index),
                    "news": news_list[result.duplicate_index]
                }
            else:
                print(f"Предупреждение: LLM вернул некорректный индекс {result.duplicate_index}")
                return None

        return None

    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON ответа от LLM: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
    except Exception as e:
        print(f"Ошибка при проверке дубликатов: {e}")
        return None
    
def find_similar_events_in_history(analyzed_event: PreparedEvent, historical_events: List[PreparedEvent]) -> List[int]:
    """
    Находит похожие события в истории по отношению к анализируемому событию.
    
    Args:
        analyzed_event: Анализируемое событие (PreparedEvent)
        historical_events: Список исторических событий для сравнения
        
    Returns:
        Список индексов событий из historical_events, которые похожи на analyzed_event
    """
    
    # Формируем список событий с индексами для промпта
    indexed_events = "\n".join([
        f"[{i}] {event.clean_description}"
        for i, event in enumerate(historical_events)
    ])
    
    prompt = f"""
Проанализируй следующее событие и определи, какие из исторических событий из СПИСКА СОБЫТИЙ являются похожими.

АНАЛИЗИРУЕМОЕ СОБЫТИЕ:
{analyzed_event.clean_description}

СПИСОК ИСТОРИЧЕСКИХ СОБЫТИЙ ДЛЯ СРАВНЕНИЯ:
{indexed_events}

КРИТЕРИИ ПОХОЖЕСТИ:
- События описывают ОДНО И ТО ЖЕ или очень похожее событие
- События происходят в ОДИН И ТОТ ЖЕ момент времени (или очень близко)
- Суть события одинакова, даже если формулировки отличаются
- События имеют схожее влияние на рынок

ПРИМЕРЫ ПОХОЖИХ СОБЫТИЙ:
- "Сбербанк объявил прибыль 400 млрд за Q3 2024" и "Сбер показал рекордную прибыль в третьем квартале - 400 миллиардов"
- "Роснефть подписала контракт с Китаем на 5 млрд долларов" и "Роснефть заключила сделку с китайской стороной на $5B"

ПРИМЕРЫ НЕ ПОХОЖИХ СОБЫТИЙ:
- "Сбербанк объявил прибыль за Q3" и "Сбербанк объявил прибыль за Q2" (разные периоды)
- "Яндекс запустил новый сервис" и "Яндекс обновил существующий сервис" (разные события)
- "Роснефть повысила дивиденды" и "Роснефть сменила CEO" (разные события одной компании)

ЗАДАЧА:
Определи, какие из исторических событий похожи на анализируемое событие.
Верни ответ строго в JSON формате со следующей структурой:
{{
    "similar_indices": [индексы похожих событий],
    "reasoning": "краткое объяснение выбора"
}}

ВАЖНО:
- Возвращай только индексы из СПИСКА СОБЫТИЙ (от 0 до {len(historical_events)-1})
- Если похожих событий нет - верни пустой массив []
- Индексы должны быть числами, а не строками
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу финансовых событий. Твоя задача - точно определять, описывают ли два события одно и то же или очень похожее событие."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = client.chat_completion(
            messages=messages,
            temperature=0.1,
            response_format="json_object"
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM при поиске похожих событий")
            return []

        # Парсинг ответа
        result = DuplicateCheckResult.model_validate_json(response_content)
        
        # Преобразуем индексы в целые числа
        similar_indices = [int(idx) for idx in result.duplicate_index] if isinstance(result.duplicate_index, list) else []
        
        return similar_indices

    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON ответа от LLM: {e}")
        print(f"Ответ LLM: {response_content}")
        return []
    except Exception as e:
        print(f"Ошибка при поиске похожих событий: {e}")
        return []

class ReportData(BaseModel):
    price_change_prediction: Literal["up", "down", "stable"] = Field(
        description="Прогноз изменения цены актива: up (рост), down (падение), stable (стабильность)"
    )
    event_summary: str = Field(
        description="Краткое описание события и его ключевых моментов"
    )
    similar_events: List[dict] = Field(
        default_factory=list,
        description="Список похожих событий с их описаниями и временем"
    )
    key_factors: List[str] = Field(
        default_factory=list,
        description="Ключевые факторы, повлиявшие на прогноз"
    )
    confidence_level: Literal["low", "medium", "high"] = Field(
        description="Уровень уверенности в прогнозе"
    )
    recommended_action: Literal["buy", "sell", "hold", "monitor"] = Field(
        description="Рекомендуемое торговое действие"
    )

def generate_report(
    analyzed_event: PreparedEvent,
    similar_events_and_prices: Optional[List[dict]] = None,
    fundamental_metrics: Optional[dict] = None
) -> Optional[dict]:
    """
    Генерирует структурированный отчет по событию и сопутствующим данным.
    
    Args:
        analyzed_event: Анализируемое событие
        similar_events_and_prices: Список похожих событий с ценами
        fundamental_metrics: Фундаментальные метрики по тикерам
        
    Returns:
        dict: Структурированный отчет или None в случае ошибки
    """
    
    # Подготовка контекста для промпта
    context_parts = []
    
    # Основное событие
    context_parts.append(f"ОСНОВНОЕ СОБЫТИЕ:\n{analyzed_event.clean_description}")
    
    # Похожие события
    if similar_events_and_prices and len(similar_events_and_prices) > 0:
        similar_context = "\n".join([
            f"Событие {i+1}: {event['event_data']['title']} ({event['event_data']['timestamp']})\n"
            f"Описание: {event['event_data']['clean_description']}\n"
            f"Тикеры: {', '.join(event['event_data']['tickers'])}"
            for i, event in enumerate(similar_events_and_prices[:3])  # Ограничиваем до 3 событий
        ])
        context_parts.append(f"ПОХОЖИЕ СОБЫТИЯ:\n{similar_context}")
    
    # Фундаментальные метрики
    if fundamental_metrics:
        metrics_context = "\n".join([
            f"{ticker}: {metrics.get('company', {}).get('name', 'N/A')} - P/E: {metrics.get('financial_ratios', {}).get('pe_ratio', 'N/A')}"
            for ticker, metrics in fundamental_metrics.items()
        ])
        context_parts.append(f"ФУНДАМЕНТАЛЬНЫЕ МЕТРИКИ:\n{metrics_context}")
    
    full_context = "\n\n".join(context_parts)
    
    prompt = f"""
Проанализируй следующие данные и создай структурированный отчет по событию.

КОНТЕКСТ СОБЫТИЯ:
{full_context}

ЗАДАЧИ:
1. Определи прогноз изменения цены актива (price_change_prediction): up (рост), down (падение), или stable (стабильность)
2. Создай краткое описание события и его ключевых моментов (event_summary)
3. Если были предоставлены похожие события, укажи их в списке similar_events с описаниями
4. Выдели ключевые факторы, повлиявшие на прогноз (key_factors)
5. Определи уровень уверенности в прогнозе (confidence_level): low (низкий), medium (средний), high (высокий)
6. Предложи рекомендуемое торговое действие (recommended_action): buy (покупать), sell (продавать), hold (удерживать), monitor (мониторить)

ВЕРНИ ОТВЕТ СТРОГО В JSON ФОРМАТЕ СО СЛЕДУЮЩЕЙ СТРУКТУРОЙ:
{{
    "price_change_prediction": "up/down/stable",
    "event_summary": "краткое описание события",
    "similar_events": [
        {{
            "timestamp": "время события",
            "description": "описание события",
            "tickers": ["тикеры"]
        }}
    ],
    "key_factors": ["фактор 1", "фактор 2"],
    "confidence_level": "low/medium/high",
    "recommended_action": "buy/sell/hold/monitor"
}}

ВАЖНО:
- Если похожих событий нет, оставь пустой массив similar_events
- Если фундаментальные метрики не предоставлены, не включай их в анализ
- Все поля обязательны, даже если пустые
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу финансовых событий и прогнозированию цен. Ты создаешь структурированные отчеты на основе новостных данных и исторических аналогов."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = client.chat_completion(
            messages=messages,
            temperature=0.3,
            response_format="json_object"
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM при генерации отчета")
            return None

        # Парсинг ответа с использованием Pydantic модели
        result = ReportData.model_validate_json(response_content)

        return result.model_dump()

    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON ответа от LLM: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
    except Exception as e:
        print(f"Ошибка при обработке ответа LLM: {e}")
        return None


# Пример использования
if __name__ == "__main__":
    # Пример данных о тикерах
    example_tickers_data = {
        "SBER": {
            "description": "Сбербанк - крупнейший банк России",
            "current_price": 250.5
        },
        "POSI": {
            "description": "Positive Technologies - компания в области информационной безопасности",
            "current_price": 1500.0
        },
        "ROSN": {
            "description": "Роснефть - нефтегазовая компания",
            "current_price": 450.2
        },
        "YDEX": {
            "description": "Яндекс - технологическая компания",
            "current_price": 3200.0
        }
    }

    # Пример новости
    example_news = """
    Сбербанк объявил о рекордной прибыли за третий квартал 2024 года.
    Чистая прибыль банка составила 400 млрд рублей, что на 25% выше показателей
    прошлого года. Руководство банка повысило прогноз по ROE до 25%.
    """

    result = enrich_news_data(example_news, example_tickers_data)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
