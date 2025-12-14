"""
Промпты для LLM предсказаний изменения цен
"""

def create_price_prediction_prompt(
    current_news: dict,
    similar_news_with_prices: list,
    ticker: str
) -> str:
    """
    Создает промпт для предсказания изменения цены.

    Args:
        current_news: Текущая новость (с enriched данными)
        similar_news_with_prices: Список похожих новостей с их ценовыми изменениями
        ticker: Тикер для которого делаем предсказание

    Returns:
        Сформированный промпт
    """

    # Формируем данные о текущей новости
    current_news_text = f"""
ТЕКУЩАЯ НОВОСТЬ (для которой нужно сделать прогноз):
Тикер: {ticker}
Описание: {current_news.get('clean_description', '')}
Тональность: {current_news.get('sentiment', '')}
Уровень влияния: {current_news.get('level_of_potential_impact_on_price', '')}
Дата публикации: {current_news.get('published_date', '')}
"""

    # Формируем данные о похожих новостях из истории
    similar_news_text = "ПОХОЖИЕ НОВОСТИ ИЗ ИСТОРИИ (с фактическими изменениями цен):\n\n"

    for idx, news in enumerate(similar_news_with_prices, 1):
        similar_news_text += f"--- Похожая новость #{idx} ---\n"
        similar_news_text += f"Описание: {news.get('clean_description', 'N/A')}\n"
        similar_news_text += f"Тональность: {news.get('sentiment', 'N/A')}\n"
        similar_news_text += f"Уровень влияния: {news.get('enriched_data', {}).get('level_of_potential_impact_on_price', 'N/A')}\n"
        similar_news_text += f"Дата публикации: {news.get('published_date', 'N/A')}\n"

        # Добавляем изменения цен (дельты)
        price_changes = news.get('price_changes', {}).get(ticker, {})
        if price_changes:
            similar_news_text += "Изменения цены после новости (абсолютные значения):\n"
            for timeframe, change in price_changes.items():
                if change is not None:
                    percent_sign = "+" if change >= 0 else ""
                    similar_news_text += f"  {timeframe}: {percent_sign}{change:.2f} руб.\n"
                else:
                    similar_news_text += f"  {timeframe}: нет данных\n"
        else:
            similar_news_text += "Изменения цены: нет данных\n"

        similar_news_text += "\n"

    # Основной промпт
    prompt = f"""
Ты эксперт-аналитик российского фондового рынка с глубокими знаниями влияния новостей на котировки акций.

Твоя задача: спрогнозировать изменение цены акции {ticker} после публикации новости.

{current_news_text}

{similar_news_text}

ЗАДАНИЕ:
На основе анализа похожих исторических новостей и их влияния на цены, спрогнозируй изменение цены акции {ticker}
после текущей новости на следующие временные интервалы: 1 час, 3 часа, 12 часов, 24 часа.

ОБРАТИ ВНИМАНИЕ: В исторических данных указаны АБСОЛЮТНЫЕ изменения цены в рублях (например, +15.5 руб означает рост на 15.5 рублей).
Используй эти данные для оценки ПРОЦЕНТНОГО изменения, учитывая типичный диапазон цен для данного актива.

Для каждого временного интервала укажи:
1. predicted_change_percent - ожидаемое процентное изменение цены (положительное для роста, отрицательное для падения)
2. confidence - уверенность в прогнозе от 0 до 1 (где 1 = абсолютная уверенность)
3. reasoning - краткое объяснение прогноза (1-2 предложения)

ВАЖНО:
- Учитывай тональность, уровень влияния и контекст новости
- Сравнивай с историческими аналогами и их абсолютными изменениями цен
- Будь реалистичен: большинство новостей влияют на цену в пределах -5% до +5%
- Если похожих новостей мало или данных недостаточно, снижай confidence
- Учитывай временную динамику: эффект новости может усиливаться или ослабевать со временем
- Помни о масштабе: изменение на 10 рублей для акции стоимостью 200 руб это ~5%, а для акции в 2000 руб - это ~0.5%

Верни ответ СТРОГО в JSON формате:
{{
    "1h": {{
        "predicted_change_percent": число (например, 1.5 для роста на 1.5% или -2.3 для падения на 2.3%),
        "confidence": число от 0 до 1,
        "reasoning": "краткое объяснение"
    }},
    "3h": {{
        "predicted_change_percent": число,
        "confidence": число от 0 до 1,
        "reasoning": "краткое объяснение"
    }},
    "12h": {{
        "predicted_change_percent": число,
        "confidence": число от 0 до 1,
        "reasoning": "краткое объяснение"
    }},
    "24h": {{
        "predicted_change_percent": число,
        "confidence": число от 0 до 1,
        "reasoning": "краткое объяснение"
    }}
}}
"""

    return prompt
