from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal, Optional, Dict
from dotenv import load_dotenv
import json
import requests
import os
import re
from .news_database_chroma import PreparedEvent
import logger

log = logger.get_logger(__name__)


TICKERS = ["SBER", "POSI", "ROSN", "YDEX"]

# Загрузка .env
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, '.env'))

YANDEX_IAM_TOKEN = os.getenv('YANDEX_IAM_TOKEN')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_MODEL_URI = os.getenv('YANDEX_MODEL_URI', f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest")

LLM_CONFIG = {
    'base_url': 'https://llm.api.cloud.yandex.net/foundationModels/v1',
    'completion_url': '/completion',
    'embedding_url': '/textEmbedding',
    'model_uri': YANDEX_MODEL_URI,
    'headers': {
        'Authorization': f'Bearer {YANDEX_IAM_TOKEN}',
        'Content-Type': 'application/json',
        'x-folder-id': YANDEX_FOLDER_ID
    },
    'default_temperature': 0.3,
    'default_max_tokens': 2000
}


def extract_json_from_text(text: str) -> str:
    """
    Извлекает JSON из текста, убирая markdown разметку и лишний текст.

    Args:
        text: Текст, который может содержать JSON в markdown блоках или с текстом

    Returns:
        Очищенная JSON строка
    """
    # Убираем markdown блоки ```json ... ```
    json_block_pattern = r'```(?:json)?\s*({.*?})\s*```'
    match = re.search(json_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1)

    # Ищем JSON объект в тексте
    json_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(0)

    return text


def yandex_chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    response_format: Optional[str] = None,
    retry_on_json_error: bool = True,
    max_retries: int = 3
) -> Optional[str]:
    """
    Прямой вызов YandexGPT API (синхронный) с улучшенной обработкой JSON.

    Args:
        messages: List[Dict] с keys 'role' и 'content'
        temperature: Температура (0.0 - 1.0)
        max_tokens: Максимум токенов в ответе
        response_format: "json_object" для JSON режима
        retry_on_json_error: Повторять запрос при ошибке парсинга JSON
        max_retries: Максимальное количество попыток

    Returns:
        Текст ответа или None при ошибке
    """
    payload = {
        "modelUri": LLM_CONFIG['model_uri'],
        "messages": [
            {"role": msg["role"], "text": msg["content"]}
            for msg in messages
        ],
        "completionOptions": {
            "temperature": temperature,
            "maxTokens": str(max_tokens)
        }
    }

    # ✅ ПРАВИЛЬНОЕ размещение jsonObject согласно документации Yandex Cloud
    # jsonObject должен быть на верхнем уровне payload
    if response_format == "json_object":
        payload["jsonObject"] = True

    for attempt in range(max_retries):
        try:
            log.info(f"Отправка запроса по адресу '{LLM_CONFIG['base_url']}{LLM_CONFIG['completion_url']}', запрос: {str(payload)[:500]}...")
            response = requests.post(
                f"{LLM_CONFIG['base_url']}{LLM_CONFIG['completion_url']}",
                json=payload,
                headers=LLM_CONFIG['headers'],
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            # Извлекаем текст ответа
            result_text = data['result']['alternatives'][0]['message']['text']

            log.info(f"Ответ от LLM: '{result_text[:500]}'")
            # Если требуется JSON, валидируем и очищаем ответ
            if response_format == "json_object":
                try:
                    # Пытаемся извлечь чистый JSON
                    cleaned_json = extract_json_from_text(result_text)
                    # Проверяем валидность JSON
                    json.loads(cleaned_json)
                    return cleaned_json
                except json.JSONDecodeError as je:
                    print(f"Попытка {attempt + 1}/{max_retries}: Ошибка парсинга JSON: {je}")
                    print(f"Исходный ответ: {result_text[:500]}...")

                    if attempt < max_retries - 1 and retry_on_json_error:
                        # БЕЗ дополнительного контекста пытаемся получить ответ от LLM (модель иногда отвечает, что на эту тему говорить не будет, в контекст это класть не стоит)
                        # # Добавляем дополнительную инструкцию в промпт
                        # messages.append({
                        #     "role": "assistant",
                        #     "content": result_text
                        # })
                        # messages.append({
                        #     "role": "user",
                        #     "content": "Ответ не является валидным JSON. Верни ТОЛЬКО чистый JSON объект без markdown разметки, без дополнительного текста и комментариев."
                        # })
                        continue
                    else:

                        return None

            return result_text

        except requests.exceptions.HTTPError as e:
            print(f"YandexGPT HTTP error {e.response.status_code}: {e.response.text}")
            return None
        except KeyError as e:
            print(f"YandexGPT unexpected response format: {e}")
            print(f"Response data: {data if 'data' in locals() else 'N/A'}")
            return None
        except Exception as e:
            print(f"YandexGPT API error: {e}")
            return None

    return None


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

    Returns:
        dict: Обогащенные данные о новости или None в случае ошибки
    """
    tickers_context = "\n".join([
        f"- {ticker}: {data.get('description', 'N/A')}"
        for ticker, data in tickers_data.items()
    ])

    prompt = f"""Ты финансовый аналитик. Твоя задача — проанализировать новость и вернуть структурированные данные в формате JSON.

НОВОСТЬ ДЛЯ АНАЛИЗА:
{event_description}

ДОСТУПНЫЕ АКТИВЫ:
{tickers_context}

ИНСТРУКЦИЯ:
Проанализируй новость и заполни следующие поля:

1. clean_description — краткое содержание новости (2-3 предложения). Убери рекламные блоки, ссылки, теги, метаданные. Сохрани ключевую информацию и цифры.

2. sentiment — оцени тон новости:
   - positive: позитивная для рынка/компаний
   - negative: негативная для рынка/компаний  
   - neutral: нейтральная или смешанная

3. tickers_of_interest — список тикеров из [{', '.join(TICKERS)}], которых касается новость. Если новость не относится к конкретным активам, верни пустой массив [].

4. level_of_potential_impact_on_price — потенциальное влияние на цену:
   - none: не влияет на цену
   - low: слабое влияние (рутинные события)
   - medium: среднее влияние (значимые корпоративные события)
   - high: сильное влияние (критические события, крупные сделки)

ФОРМАТ ОТВЕТА:
Верни ТОЛЬКО JSON без дополнительного текста, без обёрток вида ```json```, без объяснений:

{{
  "clean_description": "текст",
  "sentiment": "positive/negative/neutral",
  "tickers_of_interest": ["TICKER1", "TICKER2"],
  "level_of_potential_impact_on_price": "none/low/medium/high"
}}

Даже если новость содержит рекламу или нерелевантный контент, всегда выполняй анализ и возвращай JSON."""

    try:
        messages = [
            {
                "role": "system",
                "content": "Ты финансовый аналитик российского фондового рынка. Ты анализируешь новости и всегда возвращаешь валидный JSON без дополнительного текста. Ты никогда не отказываешься от выполнения задачи."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = yandex_chat_completion(
            messages, 
            temperature=0.1, 
            response_format="json_object",
            retry_on_json_error=True
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM")
            return None

        # Дополнительная очистка JSON
        cleaned_json = extract_json_from_text(response_content)

        # Парсинг ответа с использованием Pydantic модели
        result = EnrichedEventData.model_validate_json(cleaned_json)
        return result.model_dump()

    except ValidationError as e:
        print(f"Ошибка валидации Pydantic модели: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
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
    indexed_news = "\n".join([
        f"[{i}] {news}"
        for i, news in enumerate(news_list)
    ])

    prompt = f"""Определи, является ли ПРОВЕРЯЕМАЯ НОВОСТЬ дубликатом одной из новостей в СПИСКЕ НОВОСТЕЙ.

ПРОВЕРЯЕМАЯ НОВОСТЬ:
{main_news}

СПИСОК НОВОСТЕЙ ДЛЯ СРАВНЕНИЯ:
{indexed_news}

КРИТЕРИИ ДУБЛИКАТА:
- Новости описывают ОДНО И ТО ЖЕ событие
- События происходят в ОДИН И ТОТ ЖЕ момент времени (или очень близко)
- Суть события одинакова, даже если формулировки отличаются

ЗАДАЧА:
Проверь, есть ли в списке новостей дубликат проверяемой новости.
Если дубликат найден - укажи его индекс (число в квадратных скобках).
Если несколько похожих - выбери НАИБОЛЕЕ похожую.

ВАЖНО: Верни ТОЛЬКО валидный JSON объект без markdown разметки, без пояснений.

Структура JSON:
{{
  "is_duplicate": true/false,
  "duplicate_index": число или null,
  "reasoning": "краткое объяснение решения"
}}

ПРИМЕЧАНИЕ:
- duplicate_index должен быть null, если is_duplicate = false
- duplicate_index должен быть числом от 0 до {len(news_list)-1}, если is_duplicate = true
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу новостного контента. Твоя задача - точно определять, описывают ли две новости одно и то же событие. Ты ВСЕГДА отвечаешь чистым JSON без дополнительного текста."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = yandex_chat_completion(
            messages, 
            temperature=0.1, 
            response_format="json_object",
            retry_on_json_error=True
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM при проверке дубликатов")
            return None

        # Дополнительная очистка JSON
        cleaned_json = extract_json_from_text(response_content)

        # Парсинг ответа
        result = DuplicateCheckResult.model_validate_json(cleaned_json)

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

    except ValidationError as e:
        print(f"Ошибка валидации Pydantic модели: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON ответа от LLM: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
    except Exception as e:
        print(f"Ошибка при проверке дубликатов: {e}")
        return None


class SimilarEventsResult(BaseModel):
    similar_indices: List[int] = Field(
        default_factory=list,
        description="Список индексов похожих событий"
    )
    reasoning: str = Field(
        description="Краткое объяснение выбора"
    )


def find_similar_events_in_history(analyzed_event: PreparedEvent, historical_events: List[PreparedEvent]) -> List[int]:
    """
    Находит похожие события в истории по отношению к анализируемому событию.

    Args:
        analyzed_event: Анализируемое событие (PreparedEvent)
        historical_events: Список исторических событий для сравнения

    Returns:
        Список индексов событий из historical_events, которые похожи на analyzed_event
    """
    indexed_events = "\n".join([
        f"[{i}] {event.clean_description}"
        for i, event in enumerate(historical_events)
    ])

    prompt = f"""Проанализируй следующее событие и определи, какие из исторических событий из СПИСКА СОБЫТИЙ являются похожими.

АНАЛИЗИРУЕМОЕ СОБЫТИЕ:
{analyzed_event.clean_description}

СПИСОК ИСТОРИЧЕСКИХ СОБЫТИЙ ДЛЯ СРАВНЕНИЯ:
{indexed_events}

КРИТЕРИИ ПОХОЖЕСТИ:
- События описывают ОДНО И ТО ЖЕ или очень похожее событие
- События происходят в ОДИН И ТОТ ЖЕ момент времени (или очень близко)
- Суть события одинакова, даже если формулировки отличаются
- События имеют схожее влияние на рынок

ЗАДАЧА:
Определи, какие из исторических событий похожи на анализируемое событие.

ВАЖНО: Верни ТОЛЬКО валидный JSON объект без markdown разметки, без пояснений.

Структура JSON:
{{
  "similar_indices": [индексы похожих событий],
  "reasoning": "краткое объяснение выбора"
}}

ПРИМЕЧАНИЕ:
- Возвращай только индексы от 0 до {len(historical_events)-1}
- Если похожих событий нет - верни пустой массив []
- Индексы должны быть числами, а не строками
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу финансовых событий. Твоя задача - точно определять, описывают ли два события одно и то же или очень похожее событие. Ты ВСЕГДА отвечаешь чистым JSON без дополнительного текста."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = yandex_chat_completion(
            messages, 
            temperature=0.1, 
            response_format="json_object",
            retry_on_json_error=True
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM при поиске похожих событий")
            return []

        # Дополнительная очистка JSON
        cleaned_json = extract_json_from_text(response_content)

        # Парсинг ответа
        result = SimilarEventsResult.model_validate_json(cleaned_json)

        # Валидация индексов
        valid_indices = [
            int(idx) for idx in result.similar_indices 
            if isinstance(idx, int) and 0 <= idx < len(historical_events)
        ]

        return valid_indices

    except ValidationError as e:
        print(f"Ошибка валидации Pydantic модели: {e}")
        print(f"Ответ LLM: {response_content}")
        return []
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
    high_impact_history_events: List[dict] = Field(
        default_factory=list,
        description="Список высоковажных исторических событий, связанных с анализируемым событием"
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
    fundamental_metrics: Optional[dict] = None,
    high_impact_history_events: Optional[List[PreparedEvent]] = None
) -> Optional[dict]:
    """
    Генерирует структурированный отчет по событию и сопутствующим данным.

    Args:
        analyzed_event: Анализируемое событие
        similar_events_and_prices: Список похожих событий с ценами
        fundamental_metrics: Фундаментальные метрики по тикерам
        high_impact_history_events: Список высоковажных исторических событий

    Returns:
        dict: Структурированный отчет или None в случае ошибки
    """
    context_parts = []
    context_parts.append(f"ОСНОВНОЕ СОБЫТИЕ:\n{analyzed_event.clean_description}")

    if similar_events_and_prices and len(similar_events_and_prices) > 0:
        similar_context = "\n".join([
            f"Событие {i+1}: {event['event_data']['title']} ({event['event_data']['timestamp']})\n"
            f"Описание: {event['event_data']['clean_description']}\n"
            f"Тикеры: {', '.join(event['event_data']['tickers'])}"
            for i, event in enumerate(similar_events_and_prices[:3])
        ])
        context_parts.append(f"ПОХОЖИЕ СОБЫТИЯ:\n{similar_context}")

    if high_impact_history_events and len(high_impact_history_events) > 0:
        high_impact_context = "\n".join([
            f"Высоковажное событие {i+1}: {event.title} ({event.timestamp})\n"
            f"Описание: {event.clean_description}\n"
            f"Тикеры: {', '.join(event.tickers)}"
            for i, event in enumerate(high_impact_history_events[:3])
        ])
        context_parts.append(f"ВЫСОКОВАЖНЫЕ ИСТОРИЧЕСКИЕ СОБЫТИЯ:\n{high_impact_context}")

    if fundamental_metrics:
        metrics_context = "\n".join([
            f"{ticker}: {metrics.get('company', {}).get('name', 'N/A')} - P/E: {metrics.get('financial_ratios', {}).get('pe_ratio', 'N/A')} "
            f"EV/EBITDA: {metrics.get('financial_ratios', {}).get('ev_ebitda', 'N/A')}"
            for ticker, metrics in fundamental_metrics.items()
        ])
        context_parts.append(f"ФУНДАМЕНТАЛЬНЫЕ МЕТРИКИ:\n{metrics_context}")

    full_context = "\n\n".join(context_parts)

    prompt = f"""Проанализируй следующие данные и создай структурированный отчет по событию.

КОНТЕКСТ СОБЫТИЯ:
{full_context}

ЗАДАЧИ:
1. Определи прогноз изменения цены актива (price_change_prediction): up (рост), down (падение), или stable (стабильность)
2. Создай краткое описание события и его ключевых моментов (event_summary)
3. Если были предоставлены похожие события, укажи их в списке similar_events с описаниями
4. Если были предоставлены высоковажные исторические события, укажи их в списке high_impact_history_events с описаниями
5. Выдели ключевые факторы, повлиявшие на прогноз (key_factors)
6. Определи уровень уверенности в прогнозе (confidence_level): low (низкий), medium (средний), high (высокий)
7. Предложи рекомендуемое торговое действие (recommended_action): buy (покупать), sell (продавать), hold (удерживать), monitor (мониторить)

ВАЖНО: Верни ТОЛЬКО валидный JSON объект без markdown разметки, без пояснений.

Структура JSON:
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
  "high_impact_history_events": [
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

ПРИМЕЧАНИЕ:
- Если похожих событий нет, оставь пустой массив similar_events
- Если высоковажных исторических событий нет, оставь пустой массив high_impact_history_events
- Все поля обязательны
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по анализу финансовых событий и прогнозированию цен. Ты создаешь структурированные отчеты на основе новостных данных, исторических аналогов и фундаментальной информации. Ты ВСЕГДА отвечаешь чистым JSON без дополнительного текста."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response_content = yandex_chat_completion(
            messages, 
            temperature=0.1, 
            response_format="json_object",
            retry_on_json_error=True
        )

        if not response_content:
            print("Ошибка: пустой ответ от LLM при генерации отчета")
            return None

        # Дополнительная очистка JSON
        cleaned_json = extract_json_from_text(response_content)

        # Парсинг ответа с использованием Pydantic модели
        result = ReportData.model_validate_json(cleaned_json)
        return result.model_dump()

    except ValidationError as e:
        print(f"Ошибка валидации Pydantic модели: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON ответа от LLM: {e}")
        print(f"Ответ LLM: {response_content}")
        return None
    except Exception as e:
        print(f"Ошибка при генерации отчета: {e}")
        return None


# Пример использования
if __name__ == "__main__":
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

    example_news = """
    Сбербанк объявил о рекордной прибыли за третий квартал 2024 года.
    Чистая прибыль банка составила 400 млрд рублей, что на 25% выше показателей
    прошлого года. Руководство банка повысило прогноз по ROE до 25%.
    """

    result = enrich_news_data(example_news, example_tickers_data)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
