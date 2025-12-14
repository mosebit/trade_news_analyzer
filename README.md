# Trade News Analyzer

Система для предсказания изменения цен российских акций на основе анализа новостей с использованием RAG и LLM.

## Описание

Проект собирает новости о компаниях из разных источников, обогащает их с помощью LLM, сохраняет фактические изменения цен после публикации новости, и использует эти данные для предсказания будущих изменений цен на новые новости.

### Pipeline предсказаний

```
1. Новая новость → LLM обогащение (тикеры, sentiment, impact)
                          ↓
2. Проверка на дубликаты за последние 7 дней (RAG + LLM)
                          ↓
3. Если дубликат → пропуск
   Если не дубликат → продолжить
                          ↓
4. Поиск 5 самых похожих новостей во всей базе (RAG)
   - Фильтр по тикеру
   - Только новости с историческими данными о ценах
                          ↓
5. LLM анализ:
   - Вход: текущая новость + 5 похожих с их price changes
   - Выход: предсказание % изменения на 1h, 3h, 12h, 24h
           + confidence (0-1) + обоснование
```

### Сбор исторических данных

```
Новость (Smart-Lab/E-Disclosure/РБК)
    ↓
LLM обогащение (тикеры, sentiment, impact)
    ↓
Дедупликация (RAG distance < 0.05 → LLM проверка)
    ↓
Получение price changes с MOEX (дельта в рублях через 1h/3h/12h/24h)
    ↓
Сохранение в ChromaDB (с эмбеддингами)
```

## Установка

```bash
# 1. Создать и активировать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 2. Установить зависимости
pip install -r requirements.txt
python -m playwright install chromium

# 3. Создать .env файл в корне проекта
cat > .env << 'EOF'
LLM_MODEL=gpt-4o-mini
BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key_here
USE_CUSTOM_CLIENT=false
EOF
```

## Архитектура

```
trade_news_analyzer/
├── .env                              # Конфигурация LLM
├── requirements.txt                  # Зависимости
│
├── historical_data_preparation/      # Сбор исторических данных
│   ├── llm_client.py                 # Унифицированный LLM клиент
│   ├── ai_enrichers_and_filters.py   # Обогащение новостей LLM
│   ├── news_database_chroma.py       # ChromaDB интерфейс
│   ├── future_price_moex.py          # Получение цен с MOEX
│   ├── parser_smart_lab.py           # Парсер Smart-Lab
│   ├── parser_edisclosure.py         # Парсер E-Disclosure
│   └── parser_edisclosure_playwright.py
│
├── llm_prediction/                   # Модуль предсказаний
│   ├── news_predictor.py             # Основной pipeline
│   └── prediction_prompts.py         # Промпты для LLM
│
├── predict_news.py                   # CLI для предсказаний
└── collect_test_data.py              # CLI для сбора данных
```

## Выбранные тикеры

- **SBER** - Сбербанк
- **POSI** - Positive Technologies
- **ROSN** - Роснефть
- **YDEX** - Яндекс
