"""
News enrichment using LLM (extracted from ai_enrichers_and_filters.py).
Handles news analysis, duplicate detection, and data enrichment.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional
import json
import time
from loguru import logger

from config import get_config


class EnrichedEventData(BaseModel):
    """Structured output from LLM enrichment."""
    clean_description: str = Field(
        description="Краткое описание новости без рекламы, воды и лишних деталей"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Общий тон новости относительно упомянутых активов"
    )
    tickers_of_interest: List[str] = Field(
        default_factory=list,
        description="Список тикеров, к которым относится новость"
    )
    level_of_potential_impact_on_price: Literal["none", "low", "medium", "high"] = Field(
        description="Потенциальное влияние новости на цену актива"
    )


class DuplicateCheckResult(BaseModel):
    """Structured output from duplicate detection."""
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


class NewsEnricher:
    """Handles LLM-based news enrichment and duplicate detection."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize enricher with configuration."""
        self.config = get_config(config_path)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url
        )

        # Get tickers info
        self.tickers_dict = self.config.get_tickers_dict()
        self.ticker_list = self.config.get_ticker_list()

        logger.info(f"NewsEnricher initialized with model: {self.config.llm.model}")
        logger.info(f"  Tracking tickers: {self.ticker_list}")

    def _call_llm_with_retry(
        self,
        messages: List[Dict],
        response_model: type[BaseModel],
        temperature: float
    ) -> Optional[BaseModel]:
        """
        Call LLM with exponential backoff retry logic.

        Args:
            messages: Chat messages
            response_model: Pydantic model for structured output
            temperature: LLM temperature

        Returns:
            Pydantic model instance or None on failure
        """
        for attempt in range(self.config.llm.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.llm.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=temperature
                )

                # Parse response
                result = response_model.model_validate_json(
                    response.choices[0].message.content
                )

                return result

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error (attempt {attempt + 1}): {e}")
                logger.debug(f"Response: {response.choices[0].message.content}")

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")

            # Wait before retry
            if attempt < self.config.llm.max_retries - 1:
                wait_time = self.config.llm.retry_delay_seconds * (2 ** attempt)
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logger.error(f"Failed after {self.config.llm.max_retries} attempts")
        return None

    def enrich_news(self, news_text: str) -> Optional[Dict]:
        """
        Enrich news with LLM analysis.

        Args:
            news_text: Raw news text

        Returns:
            Enriched data dictionary or None on error
        """
        # Build tickers context
        tickers_context = "\n".join([
            f"- {ticker}: {description}"
            for ticker, description in self.tickers_dict.items()
        ])

        prompt = f"""
Проанализируй следующую новость и предоставь структурированный ответ.

НОВОСТЬ:
{news_text}

ДОСТУПНЫЕ АКТИВЫ ДЛЯ АНАЛИЗА:
{tickers_context}

ЗАДАЧИ:
1. Создай краткое описание новости (clean_description) - убери рекламу, воду, лишние детали. Оставь только суть, НО ничего полезного НЕ удаляй!
2. Определи общий тон новости (sentiment): positive (позитивная), negative (негативная) или neutral (нейтральная).
3. Определи, к каким тикерам из списка [{', '.join(self.ticker_list)}] относится эта новость (tickers_of_interest). Список может быть пустым, если новость не касается ни одного из активов.
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

        result = self._call_llm_with_retry(
            messages=messages,
            response_model=EnrichedEventData,
            temperature=self.config.llm.temperature_enrichment
        )

        if result:
            logger.debug(f"Enriched: {result.clean_description[:80]}...")
            return result.model_dump()

        return None

    def find_duplicate(self, news_text: str, candidate_news_list: List[str]) -> Optional[Dict]:
        """
        Check if news is a duplicate of any news in the list.

        Args:
            news_text: News to check
            candidate_news_list: List of candidate duplicate news texts

        Returns:
            {"index": int, "news": str} if duplicate found, None otherwise
        """
        if not candidate_news_list:
            return None

        # Build indexed list
        indexed_news = "\n".join([
            f"[{i}] {news}"
            for i, news in enumerate(candidate_news_list)
        ])

        prompt = f"""
Определи, является ли ПРОВЕРЯЕМАЯ НОВОСТЬ дубликатом одной из новостей в СПИСКЕ НОВОСТЕЙ.

ПРОВЕРЯЕМАЯ НОВОСТЬ:
{news_text}

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
- duplicate_index должен быть числом от 0 до {len(candidate_news_list)-1}, если is_duplicate = true
- reasoning должен кратко объяснять, почему новости являются или не являются дубликатами
"""

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

        result = self._call_llm_with_retry(
            messages=messages,
            response_model=DuplicateCheckResult,
            temperature=self.config.llm.temperature_duplicate_check
        )

        if result and result.is_duplicate and result.duplicate_index is not None:
            if 0 <= result.duplicate_index < len(candidate_news_list):
                logger.info(f"Duplicate found: index={result.duplicate_index}, reason={result.reasoning}")
                return {
                    "index": result.duplicate_index,
                    "news": candidate_news_list[result.duplicate_index]
                }
            else:
                logger.warning(f"Invalid duplicate index: {result.duplicate_index}")

        return None


if __name__ == "__main__":
    # Test enricher
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    enricher = NewsEnricher()

    test_news = """
    Сбербанк объявил о рекордной прибыли за третий квартал 2024 года.
    Чистая прибыль банка составила 400 млрд рублей, что на 25% выше показателей
    прошлого года. Руководство банка повысило прогноз по ROE до 25%.
    """

    print("\n" + "="*50)
    print("TESTING NEWS ENRICHMENT:")
    print("="*50)

    result = enricher.enrich_news(test_news)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
