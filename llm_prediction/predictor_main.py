"""В этом модуле реализована логика обработки новостей, появляющихся в реальном времени"""
import sys
# from pathlib import Path
# project_root = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(project_root))

from historical_data_preparation import news_database_chroma
from historical_data_preparation import saving_pipeline
from historical_data_preparation import ai_enrichers_and_filters
from historical_data_preparation import future_price_moex
from llm_prediction import searcher

import os
import json
from datetime import datetime, timedelta

def get_fundamental_metrics(ticker: str, info_file_path=None):
    if not info_file_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        info_file_path = os.path.join(script_dir, 'fundamental_metrics.json')

    # Проверка существования файла
    if not os.path.exists(info_file_path):
        raise FileNotFoundError(f"Файл {info_file_path} не найден")

    # Чтение JSON файла
    try:
        with open(info_file_path, 'r', encoding='utf-8') as f:
            companies_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON: {e}")
    except Exception as e:
        raise IOError(f"Ошибка чтения файла: {e}")

    # Поиск компании по тикеру (case-insensitive)
    ticker_upper = ticker.upper()
    for company_data in companies_data:
        if company_data.get('company', {}).get('ticker', '').upper() == ticker_upper:
            return company_data

    # Если не найдено
    available_tickers = [c.get('company', {}).get('ticker') for c in companies_data]
    print(f"Тикер '{ticker}' не найден. Доступные тикеры: {available_tickers}")
    return None

def get_prices_intervals(event: news_database_chroma.PreparedEvent):
    event_time_moex = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    event_time_minus_day = (datetime.fromtimestamp(event.timestamp) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    event_time_minus_week = (datetime.fromtimestamp(event.timestamp) - timedelta(weeks=1)).strftime("%Y-%m-%d %H:%M:%S")
    event_time_minus_month = (datetime.fromtimestamp(event.timestamp) - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

    tickers_and_prices = {}
    for ticker in event.tickers:
        prices_lists = {
            "last_1_day_and_30m_intervals": future_price_moex.load_moex_candles(ticker, event_time_minus_day, event_time_moex, interval_minutes=30),
            "last_7_days_and_8h_intervals": future_price_moex.load_moex_candles(ticker, event_time_minus_week, event_time_moex, interval_minutes=480),
            "last_1_month_and_1day_intervals": future_price_moex.load_moex_candles(ticker, event_time_minus_month, event_time_moex, interval_days=1)
        }
        tickers_and_prices[ticker] = prices_lists

    return tickers_and_prices

def event_process_chain(
        new_event: news_database_chroma.PreparedEvent, 
        chroma_db: news_database_chroma.NewsDatabase
        ):
    similar_events_and_prices = None
    fundamental_metrics=None

    # проверка импакта - если ниже чем medium, то данную новость просто сохраняем в БД, к публикации не готовим
    if new_event.impact not in ["medium", "high"]:
        saving_pipeline.saving_pipeline(new_event)
        return None
    
    # получение исторических данных
    similar_from_rag = chroma_db.find_similar_news_by_event_new(new_event, limit=10)
    if similar_from_rag:
        similar_validated_indexes = ai_enrichers_and_filters.find_similar_events_in_history(new_event, similar_from_rag)
        
        if similar_validated_indexes:
            similar_events = []
            for i in similar_validated_indexes:
                similar_events.append(similar_from_rag[i])

            # получение списков цен по разным интервалам для похожих новостей
            similar_events_and_prices = []
            for history_event in similar_events:
                similar_events_and_prices.append({
                    "event_data": {
                        "timestamp": history_event.timestamp,
                        "tickers": history_event.tickers,
                        "title": history_event.title,
                        "clean_description": history_event.clean_description,
                        "url": history_event.url
                    },
                    "prices": get_prices_intervals(history_event)
                })

    # получение фундаментальных параметров для оценки актива
    fundamental_metrics = {}
    for ticker in new_event.tickers:
        ticker_metrics = get_fundamental_metrics(ticker) 
        if ticker_metrics:
            fundamental_metrics[ticker] = ticker_metrics

    # отправка данных в LLM для подготовки отчета
    llm_report = ai_enrichers_and_filters.generate_report(
        new_event,
        similar_events_and_prices,
        fundamental_metrics
    )

    saving_pipeline.saving_pipeline(new_event)

    if llm_report:
        print("\n" + "="*60)
        print("ФИНАНСОВЫЙ ОТЧЕТ ПО СОБЫТИЮ")
        print("="*60)
        print(f"Событие: {new_event.title}")
        print(f"Время: {datetime.fromtimestamp(new_event.timestamp)}")
        print(f"Тикеры: {', '.join(new_event.tickers)}")
        print(f"Прогноз изменения цены: {llm_report['price_change_prediction']}")
        print(f"Уровень уверенности: {llm_report['confidence_level']}")
        print(f"Рекомендация: {llm_report['recommended_action']}")
        print("\nКраткое описание события:")
        print(llm_report['event_summary'])
        print("\nКлючевые факторы:")
        for factor in llm_report['key_factors']:
            print(f"  • {factor}")
        if llm_report['similar_events']:
            print("\nПохожие события:")
            for i, event in enumerate(llm_report['similar_events'], 1):
                print(f"  {i}. {event['description']} ({event['timestamp']})")
                print(f"     Тикеры: {', '.join(event['tickers'])}")
        print("="*60)

    return llm_report

def evaluate_new_events(tickers=['SBER', 'POSI'], time_gap_seconds=320000):
    db = news_database_chroma.NewsDatabase(path='./chroma_db_new')
    new_events = searcher.find_new_news(tickers, time_gap_seconds)
    for event in new_events:
        event_process_chain(event, db)
    
if __name__ == "__main__":
    evaluate_new_events(['SBER', 'POSI'], 160000)


    # print(json.dumps(get_fundamental_metrics("SBER"), indent=2, ensure_ascii=False))
    # print(get_fundamental_metrics("LOL"))