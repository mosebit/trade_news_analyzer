from pydantic import BaseModel, Field
from typing import List, Literal
import os
import json
from dotenv import load_dotenv
import sys

from llm_client import create_llm_client

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Initialize LLM client using the factory function
client = create_llm_client(
    use_custom=os.getenv("USE_CUSTOM_CLIENT", "false").lower() == "true",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("LLM_MODEL")
)
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
                    "index": result.duplicate_index,
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


def check_and_handle_duplicates(db, original_text: str, enriched_event: dict, news_timestamp: int, distance_threshold: float = 0.05):
    """
    Проверяет наличие дубликатов новости в базе данных.

    Args:
        db: Экземпляр базы данных с методами find_similar_news_by_text
        original_text: Оригинальный текст новости
        enriched_event: Обогащенные данные новости (с clean_description)
        news_timestamp: Временная метка новости для сравнения
        distance_threshold: Порог distance для LLM проверки (по умолчанию 0.05)

    Returns:
        tuple: (should_skip, similar_event_to_replace)
            - should_skip (bool): True если новость - дубликат и нужно пропустить
            - similar_event_to_replace (dict|None): Данные похожей новости для замены, если текущая новость старше
    """
    # Поиск похожих новостей в базе
    similar_in_db = db.find_similar_news_by_text(query_text=enriched_event.get('clean_description'))

    if not similar_in_db:
        return False, None

    print('RAG - найдены похожие новости в БД')

    # Оптимизация: проверяем distance - если слишком далеко, точно не дубликат
    very_similar = [i for i in similar_in_db if i.get('distance', 1.0) < distance_threshold]

    if not very_similar:
        print(f'RAG distance >= {distance_threshold} для всех похожих новостей - пропускаем дорогую LLM проверку')
        return False, None

    # Только для очень похожих новостей делаем дорогую LLM проверку
    duplicates_verdict = find_duplicates(
        original_text,
        [i.get('clean_description') for i in very_similar]
    )

    if not duplicates_verdict:
        return False, None

    print(f'LLM посчитала новости дубликатами:\n - {original_text}\n - {duplicates_verdict.get("news")}')

    # Получение всех данных о схожей новости
    similar_event_data = None
    for i in very_similar:
        if i.get('clean_description') == duplicates_verdict.get('news'):
            similar_event_data = i
            break

    if not similar_event_data:
        return False, None

    # Сравниваем даты - если текущая новость старше, нужно заменить
    if news_timestamp < similar_event_data['date_timestamp']:
        print(f"Текущая новость старше ({news_timestamp} < {similar_event_data['date_timestamp']}) - нужна замена")
        return True, similar_event_data
    else:
        # Текущая новость новее дубликата - просто пропускаем
        print(f"Текущая новость новее ({news_timestamp} >= {similar_event_data['date_timestamp']}) - пропускаем")
        return True, None


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
