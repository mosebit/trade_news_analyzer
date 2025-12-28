"""
Данный модуль отвечает за поиск свежих непроанализированных новостей.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from historical_data_preparation import parser_smart_lab
from historical_data_preparation import news_database_chroma
import logger

import time
from typing import List, Dict, Optional

log = logger.get_logger(__name__)

DB_PATH = './chroma_db_new'
TICKERS = ["SBER", "POSI", "ROSN", "YDEX"]

# 86400 - секунд в сутках
def process_new_urls_smartlab(ticker, gap_length_sec=86400):
    db = news_database_chroma.NewsDatabase(path='./chroma_db_new')
    current_timestamp = time.time()

    timestamp_end = current_timestamp - gap_length_sec

    log.info(f"Получение списка опубликованных новостей за последние '{gap_length_sec}' секунд")
    smart_lab_urls = parser_smart_lab.prepare_urls_until_timestamp(timestamp_end, ticker)

    unsaved_urls = db.filter_unsaved_urls(smart_lab_urls)
    log.info(f"Последние несохраненные посты: '{unsaved_urls}'")

    prepared_news = []
    for url in unsaved_urls:
        prepared_for_saving = parser_smart_lab.analyze_page_with_url(url)
        prepared_news.append(prepared_for_saving)
        print(f"Обработана новость для последующего анализа: '{url}'")

    return prepared_news

def find_new_news(tickers=None, gap_length_sec=86400) -> List[news_database_chroma.PreparedEvent]:
    tickers_of_interest = tickers if tickers else TICKERS
    
    final_events_list = []
    for ticker in tickers_of_interest:
        news_list = process_new_urls_smartlab(ticker, gap_length_sec)
        if isinstance(news_list, list) and news_list:
            final_events_list.extend(news_list)
    
    return final_events_list



if __name__ == "__main__":
    # process_new_urls_smartlab('SBER', 160000)
    find_new_news(['SBER', 'POSI'], 160000)