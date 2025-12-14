import requests
from typing import Optional
from lxml import html
from datetime import datetime
import os
from dotenv import load_dotenv

from . import ai_enrichers_and_filters
from . import news_database_chroma
from . import future_price_moex

headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "priority": "u=0, i",
    # "referer": f"https://smart-lab.ru/forum/news/{ticker}/page{doc_index-1}/",
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
}

def fetch_raw_smartlab_post_links(ticker: str, doc_index: int) -> Optional[requests.Response]:
    url = f"https://smart-lab.ru/forum/news/{ticker}/page{doc_index}/"

    try:
        response = requests.get(url, headers=headers)
        return response
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None

def fetch_raw_data_by_url(url: str):
    try:
        response = requests.get(url, headers=headers)
        return response
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None

def get_pretty_post_links(html_content):
    tree = html.fromstring(html_content)

    links = tree.xpath('//ul[@class="temp_headers temp_headers--have-numbers"]//a/@href')

    # /blog/1118401.php -> https://smart-lab.ru/blog/news/1118401.php
    full_links = [f"https://smart-lab.ru/blog/news/{link.split('/')[2]}" for link in links]

    return full_links

# def get_pretty_data_from_one_post(html_content):
#     tree = html.fromstring(html_content)

#     # Дата
#     date = tree.xpath('//li[@class="date"]/text()')[0].strip()

#     # Весь текст для эмбеддинга
#     title = tree.xpath('//h1[@class="title "]//span/text()')[0]
#     content = ' '.join(tree.xpath('//div[@class="content"]//text()[normalize-space()]'))
#     tags = tree.xpath('//ul[@class="tags"]//a/text()')

#     # Объединяем
#     full_text = f"{title}. {content}. {' '.join(tags)}"

#     return {
#         'date': date,
#         'text': full_text,
#         'title': title  # для отладки
#     }

MONTHS = {'января':1,'февраля':2,'марта':3,'апреля':4,'мая':5,'июня':6,
          'июля':7,'августа':8,'сентября':9,'октября':10,'ноября':11,'декабря':12}

def get_pretty_data_from_one_post(html_content):
    tree = html.fromstring(html_content)

    # Парсим дату
    date_str = tree.xpath('//li[@class="date"]/text()')[0].strip()
    parts = date_str.replace(',', '').split()
    date_obj = datetime(
        int(parts[2]),  # year
        MONTHS[parts[1]],  # month
        int(parts[0]),  # day
        *map(int, parts[3].split(':'))  # hour, minute
    )

    # Текст
    title = tree.xpath('//h1[@class="title "]//span/text()')[0]
    content = ' '.join(tree.xpath('//div[@class="content"]//text()[normalize-space()]'))
    tags = tree.xpath('//ul[@class="tags"]//a/text()')

    return {
        'date': date_obj.isoformat(),
        'date_timestamp': int(date_obj.timestamp()),
        'text': f"{title}. {content}. {' '.join(tags)}",
        'title': title
    }

tickers_descriptions = {
    "POSI": {
        "description": "Positive Technologies (тикер POSI на Московской бирже MOEX) — российский разработчик решений в сфере кибербезопасности для корпоративных и государственных клиентов, работающий в секторе программного обеспечения и ИБ‑сервисов, с выручкой порядка 24–25 млрд руб. в 2024 году и включением акций в первый котировальный список и ключевые ИТ‑индексы Мосбиржи. Для инвестора это история роста в нише кибербезопасности с высокой долей выручки от лицензий, заметной волатильностью прибыли (сокращение чистой прибыли по МСФО в 2024 году примерно до 3,7 млрд руб.) и активной работой с розничными акционерами, в том числе через SPO и программы обратного выкупа."
    },
    "SBER": {
        "description": "Сбербанк  (тикер SBER на Московской бирже) — крупнейший универсальный банк России и Восточной Европы, работающий в финансовом секторе (банковские услуги, финтех, экосистемные цифровые сервисы) с прибылью по МСФО около 1,58 трлн руб. за 2024 год и рентабельностью капитала около 24–25%. Для инвестора это высокомаржинальный лидер рынка с масштабным дивидендным потенциалом и ликвидностью «голубой фишки», чувствительный к ставке ЦБ, состоянию российской экономики и регуляторным/геополитическим рискам."
    },
    "ROSN": {
        "description": "Роснефть  (тикер ROSN на Московской бирже) — крупнейшая российская публичная нефтегазовая компания с вертикально интегрированным бизнесом по добыче нефти и газа, переработке и розничным продажам нефтепродуктов и существенной долей в экспорте углеводородов. Для инвестора это системообразующий эмитент сырьевого сектора с триллионной выручкой в год, высокой капиталоемкостью, чувствительностью к ценам на нефть, курсу рубля, ставке ЦБ и санкционным/регуляторным рискам, но традиционно значимым дивидендным потоком и ролью в индексах российского рынка."
    },
    "YDEX": {
        "description": "Яндекс (тикер YDEX на Московской бирже) — крупнейшая российская ИТ‑компания, работающая в секторе интернет‑сервисов и цифровых экосистем (поиск и реклама, такси, e‑commerce, медиа и облака) с быстро растущей выручкой и высокой маржинальностью по EBITDA на фоне умеренной долговой нагрузки. Для инвестора это классическая история роста с регулярными полугодовыми дивидендами (ориентир на существенную долю от прибыли) и чувствительностью к динамике российского потребительского спроса, регуляторной политике в сфере ИТ и конкуренции на рынке онлайн‑рекламы и сервисов"
    }
}

def analyze_page_of_news_NEW(ticker: str, page_index: int):
    raw_posts = fetch_raw_smartlab_post_links(ticker, page_index)
    links_list = get_pretty_post_links(raw_posts.text)

    # Инициализация БД
    # db = news_database.NewsDatabase("news_data.db")
    db = news_database_chroma.NewsDatabase("./chroma_db_new")

    smallest_date_int = 32536799999 # Maximum value of timestamp

    for link in links_list:
        # получение данных по конкретному посту
        raw_page = fetch_raw_data_by_url(link)
        data_from_post = get_pretty_data_from_one_post(raw_page.text)

        if data_from_post['date_timestamp'] < smallest_date_int:
            smallest_date_int = data_from_post['date_timestamp']

        # TODO: дополнительный шаг (НА БУДУЩЕЕ) - проверка наличия дублкатов новости в базе

        # отправка в LLM для обогащения
        enriched_event = ai_enrichers_and_filters.enrich_news_data(
            data_from_post['text'],
            tickers_descriptions
            )

        # проверка на дубликат
        # similar_in_db = db.find_similar_news_by_text(query_text=data_from_post.get('text'))
        similar_in_db = db.find_similar_news_by_text(query_text=enriched_event.get('clean_description'))
        if similar_in_db:
            print('RAG - найдены похожие новости в БД')
            duplicates_verdict = ai_enrichers_and_filters.find_duplicates(data_from_post['text'], [i.get('clean_description') for i in similar_in_db])
            if duplicates_verdict:
                print(f"LLM посчитала новости дубликатами:\n - {data_from_post['text']}\n - {duplicates_verdict.get('news')}")

                # получение всех данных о схожей новости
                for i in similar_in_db:
                    if i.get('clean_description') == duplicates_verdict.get('news'):
                        similar_event_data = i

                if data_from_post['date_timestamp'] < similar_event_data['date_timestamp']:
                    # удаление похожей новости и добавление вместо нее той, которая появилась раньше
                    db.delete_news(similar_event_data.get('url'))

                    # Получаем изменения цен
                    try:
                        price_changes = future_price_moex.get_future_price_changes(
                            news_time=data_from_post['date'],
                            tickers=enriched_event.get('tickers_of_interest', [])
                        )
                        print(f"Got price changes for news")
                    except Exception as e:
                        print(f"Warning: Error fetching prices: {e}")
                        price_changes = None

                    db.save_news(
                        url=link,
                        title=data_from_post['title'],
                        original_text=data_from_post['text'],
                        enriched_data=enriched_event,
                        published_date=data_from_post['date'],
                        published_timestamp=data_from_post['date_timestamp'],
                        other_urls=[similar_event_data['url']],
                        price_changes=price_changes
                    )

                continue



        # print(enriched_event['clean_description'])
        if enriched_event and enriched_event.get('level_of_potential_impact_on_price') in ["low", "medium", "high"]:
            # Получаем изменения цен
            try:
                price_changes = future_price_moex.get_future_price_changes(
                    news_time=data_from_post['date'],
                    tickers=enriched_event.get('tickers_of_interest', [])
                )
                print(f"Got price changes for news")
            except Exception as e:
                print(f"Warning: Error fetching prices: {e}")
                price_changes = None

            db.save_news(
                url=link,
                title=data_from_post['title'],
                original_text=data_from_post['text'],
                enriched_data=enriched_event,
                published_date=data_from_post['date'],
                published_timestamp=data_from_post['date_timestamp'],
                price_changes=price_changes
            )

    # Вывод статистики
    print("\n" + "="*50)
    print("СТАТИСТИКА:")
    stats = db.get_stats()
    print(f"Всего новостей в базе: {stats['total_news']}")
    print(f"По тикерам: {stats['by_ticker']}")
    print("="*50)

    db.close()

    return smallest_date_int

# def analyze_page_of_news(ticker: str, page_index: int):
#     raw_posts = fetch_raw_smartlab_post_links(ticker, page_index)
#     links_list = get_pretty_post_links(raw_posts.text)

#     # Инициализация БД
#     db = news_database.NewsDatabase("news_data.db")

#     smallest_date_int = 32536799999 # Maximum value of timestamp

#     for link in links_list:
#         # получение данных по конкретному посту
#         raw_page = fetch_raw_data_by_url(link)
#         data_from_post = get_pretty_data_from_one_post(raw_page.text)

#         if data_from_post['date_timestamp'] < smallest_date_int:
#             smallest_date_int = data_from_post['date_timestamp']

#         # TODO: дополнительный шаг (НА БУДУЩЕЕ) - проверка наличия дублкатов новости в базе

#         # отправка в LLM для обогащения
#         enriched_event = ai_enrichers_and_filters.enrich_news_data(
#             data_from_post['text'],
#             tickers_descriptions
#             )

#         # print(enriched_event['clean_description'])
#         if enriched_event and enriched_event.get('level_of_potential_impact_on_price') in ["low", "medium", "high"]:
#             db.save_news(
#                 url=link,
#                 title=data_from_post['title'],
#                 original_text=data_from_post['text'],
#                 enriched_data=enriched_event,
#                 published_date=data_from_post['date'],
#                 published_timestamp=data_from_post['date_timestamp']
#             )

#     # Вывод статистики
#     print("\n" + "="*50)
#     print("СТАТИСТИКА:")
#     stats = db.get_stats()
#     print(f"Всего новостей в базе: {stats['total_news']}")
#     print(f"По тикерам: {stats['by_ticker']}")
#     print("="*50)

#     db.close()

#     return smallest_date_int

# 2024-01-16T10:30:00
def prepare_news_until_date(date_iso: str, tickers: list):
    date_obj = datetime.fromisoformat(date_iso)
    timestamp_of_end = int(date_obj.timestamp())

    for ticker in tickers:
        page_index = 1
        while True:
            current_timestamp = analyze_page_of_news_NEW(ticker, page_index)
            if current_timestamp < timestamp_of_end:
                break
            page_index += 1

            print(f"\ncurrent_timestamp: {current_timestamp}, timestamp_of_end: {timestamp_of_end}\n")

    return timestamp_of_end


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

    prepare_news_until_date("2025-10-01T10:30:00", ["SBER", "POSI", "ROSN", "YDEX"])
    # analyze_page_of_news("POSI", 0)


# # Пример использования функции
# if __name__ == "__main__":
#     # Пример вызова функции с тикером POSI и индексом документа 3
#     result = fetch_raw_smartlab_post_links("POSI", 3)
#     if result:
#         print(f"Статус код: {result.status_code}")
#         # print(f"Заголовки: {result.headers}")
#         # # Выводим первые 500 символов содержимого
#         # print(f"Содержимое (первые 500 символов): {result.text[:500]}")

#         # получение готового к использованию списка ссылок на посты
#         links_list = get_pretty_post_links(result.text)
#         print(links_list)
