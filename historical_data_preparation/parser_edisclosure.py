import requests
import time
from typing import Optional
from lxml import html
from datetime import datetime
import os
from dotenv import load_dotenv

from . import parser_edisclosure_playwright as edisclosure_api
from . import news_database_chroma
from . import ai_enrichers_and_filters
from . import future_price_moex
from . import saving_pipeline

# # TODO - tickers_descriptions перенести в JSON
# from .parser_smart_lab import tickers_descriptions

import logger

log = logger.get_logger(__name__)

mapping_tickers_ids = {
    "POSI": [38196, 38538],
    "SBER": [3043],
    "ROSN": [6505],
    "YDEX": [39059]
}

def get_years_in_past(year):
    now = datetime.now()
    current_year = now.year
    years = list(range(year, current_year + 1))
    return years

def get_pretty_data_from_one_post(html_content):
    tree = html.fromstring(html_content)

    # Парсим дату
    date_str = tree.xpath('//div[@class="time left"]/span[@class="date"]/text()')[0].strip()
    # '12.12.2025 16:17'

    parts = date_str.replace(',', '').split()
    day, month, year = map(int, parts[0].split('.'))
    hour, minute = map(int, parts[1].split(':'))
    date_obj = datetime(
        year,
        month,
        day,
        hour,
        minute
    )

    # Текст
    title = tree.xpath('//h4/text()')[0]
    content = tree.xpath('//div[@style="word-break: break-word; word-wrap: break-word; white-space: pre-wrap;"]/text()')[0]
    # tags = tree.xpath('//ul[@class="tags"]//a/text()')

    return {
        'date': date_obj.isoformat(),
        'date_timestamp': int(date_obj.timestamp()),
        'text': f"TITLE:\n{title}. CONTENT:\n{content}.",
        'title': title
    }

def analyze_company_news_until_timestamp(ticker, date_obj_until: datetime):
    until_year = date_obj_until.year
    years_list = get_years_in_past(until_year)

    edisclosure_client = edisclosure_api.EDisclosureClient()
    db = news_database_chroma.NewsDatabase("./chroma_db_new")

    for company_id in mapping_tickers_ids.get(ticker):
        end_of_analysis = False
        for year in years_list:
            log.info(f"Получение набора идентификаторов новостей по компании '{ticker}' за год '{year}'")
            events_small_data = edisclosure_client.get_events_data_by_year(company_id, year)

            if events_small_data:
                # вытаскиваются ids новостей из полученного набора данных
                events_ids = [event.get('pseudoGUID') for event in events_small_data]

                # анализ каждой новости по отдельности
                for event_id in events_ids:
                    log.info(f"Получение черновых данных для новости с event_id: '{event_id}' и company_id: '{company_id}'")
                    event_raw_data, event_url = edisclosure_client.get_one_event_raw_data(event_id, company_id)
                    # парсинг черновых данных - получение из них даты, описания и отдельно заголовка новости
                    event_parsed_data = get_pretty_data_from_one_post(event_raw_data)

                    # отправка в LLM для обогащения
                    log.info(f"Обогащение новости при помощи LLM, title обрабатываемой: '{event_parsed_data.get('title')}'")
                    enriched_event = ai_enrichers_and_filters.enrich_news_data(event_parsed_data['text'])

                    prepared_for_saving = news_database_chroma.PreparedEvent(
                        url=event_url,
                        title=event_parsed_data.get('title'),
                        clean_description=enriched_event.get('clean_description'),
                        original_text=event_parsed_data.get('text'),
                        tickers=enriched_event.get('tickers_of_interest', []),
                        sentiment=enriched_event.get('sentiment'),
                        impact=enriched_event.get('level_of_potential_impact_on_price'),
                        published_date=event_parsed_data.get('date'),
                        timestamp=event_parsed_data.get('date_timestamp')
                    )

                    log.info(f"Отправка новости в пайплайн сохранения в БД")
                    saving_pipeline.saving_pipeline(prepared_for_saving, './chroma_db_new')

                    # если время собранной новости уже больше граничного для анализа, то сворачиваемся
                    if event_parsed_data.get('date_timestamp') < date_obj_until.timestamp():
                        end_of_analysis = True
                        break

                    # # Сохраняем новость если она прошла все проверки
                    # if enriched_event and enriched_event.get('level_of_potential_impact_on_price') in ["low", "medium", "high"]:
                    #     # Получаем изменения цен
                    #     try:
                    #         price_changes = future_price_moex.get_future_price_changes(
                    #             news_time=event_parsed_data['date'],
                    #             tickers=enriched_event.get('tickers_of_interest', [])
                    #         )
                    #         print(f"Got price changes for news")
                    #     except Exception as e:
                    #         print(f"Warning: Error fetching prices: {e}")
                    #         price_changes = None

                    #     # Генерируем URL из event_id
                    #     event_url = f"https://www.e-disclosure.ru/portal/event.aspx?EventId={event_id}&CompanyId={company_id}"

                    #     db.save_news(
                    #         url=event_url,
                    #         title=event_parsed_data['title'],
                    #         original_text=event_parsed_data['text'],
                    #         enriched_data=enriched_event,
                    #         published_date=event_parsed_data['date'],
                    #         published_timestamp=event_parsed_data['date_timestamp'],
                    #         price_changes=price_changes
                    #     )

                if end_of_analysis:
                    break


            elif events_small_data == []: # когда возвращается пустой список это значит новостей за переданный год и последующие уже нет
                break

    # Закрываем БД
    db.close()


def prepare_news_until_date(date_iso: str, tickers: list):
    date_obj = datetime.fromisoformat(date_iso)
    timestamp_of_end = int(date_obj.timestamp())

    for ticker in tickers:
        current_timestamp = analyze_company_news_until_timestamp(ticker, date_obj)

    return timestamp_of_end


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

    prepare_news_until_date("2025-10-01T10:30:00", ["SBER", "POSI", "ROSN", "YDEX"])
