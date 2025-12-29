"""
В этом файле реализован пайплайн умного сохранения в БД информации о новом событии (новости) связанной
с каким-то из анализируемых активов.

Перед сохранением события в БД мы должны убедиться в отсутствии дубликатов данной новости в базе, 
если есть похожие события нужно проверить при помощи LLM действительно ли они являются дубликатами.

Если в базе информация об этой новости уже сохранена, то нужно проверить какая из новостей (сохраненная или 
анализируемая в данный момент) имеет более раннюю дату публикации, в базе должна остаться самая ранняя новость.
"""

from . import ai_enrichers_and_filters
from . import news_database_chroma
from . import future_price_moex


def find_duplicates(
        new_event: news_database_chroma.PreparedEvent, 
        db_name: str = './chroma_db_new') -> news_database_chroma.PreparedEvent:
    """
    Проверяет, есть ли в базе дубликаты новостей, похожих на new_event.

    Args:
        new_event: Обогащенная информация о новой новости
        db: Экземпляр базы данных для поиска похожих новостей
    
    Returns:
        Словарь с информацией о дубликате, если он найден, иначе None
    """

    # Ищем похожие новости в БД - RAG
    db = news_database_chroma.NewsDatabase(db_name)
    similar_in_db = db.find_similar_news_by_event_new(new_event, threshold=0.1)

    if not similar_in_db:
        return None
    
    # проверка похожих новостей на наличие дубликата
    duplicates_verdict = ai_enrichers_and_filters.find_duplicates(
        new_event.clean_description, 
        [i.clean_description for i in similar_in_db]
        )

    if isinstance(duplicates_verdict, dict) and duplicates_verdict.get('index', -1) >= 0:
        return similar_in_db[duplicates_verdict.get('index')]

    return None

def saving_pipeline(
        new_event: news_database_chroma.PreparedEvent, 
        db_name: str = './chroma_db_new'
        ):
    db = news_database_chroma.NewsDatabase(db_name)
    similar_in_db = find_duplicates(new_event, db_name)

    # сохранение новости не происходит только в одном случае - когда она уже
    # сохранена в базе (была получена из другого источника) и дата сохраненной новости
    # является более ранней чем у анализируемой на данный момент
    if similar_in_db:
        if similar_in_db.timestamp > new_event.timestamp:
            # новость с более поздней датой из БД удаляем, далее будет сохранена более свежая 
            db.delete_news(similar_in_db.url)
        else:
            # новость с более ранней датой уже в БД
            return None
    
    # price_changes = future_price_moex.get_future_price_changes(
    #     news_time=new_event.published_date,
    #     tickers=new_event.tickers
    # )
    db.save_news_new(new_event)


if __name__ == "__main__":
    new_event_example = news_database_chroma.PreparedEvent(
        title="Герман Греф о рынке ипотеки",
        clean_description="Герман Греф заявил об устойчивом тренде снижения ставок на рынке, включая ипотеку. Ожидается рост объема ипотеки при одновременном сворачивании льготных программ и дифференциации ставок в зависимости от количества детей в семье. Прогнозируется восстановление выдач ипотеки в следующем году и нормализация ситуации на рынке.",
        original_text="Some original text",
        tickers=['SBER'],
        sentiment='neutral',
        impact='none',
        published_date="2024-01-01",
        timestamp=1704067200
    )
    saving_pipeline(new_event_example)

# if __name__ == "__main__":
#     new_event_example = {
#         # 'clean_description': 'Сбер, Греф',
#         'clean_description': 'Герман Греф заявил об устойчивом тренде снижения ставок на рынке, включая ипотеку. Ожидается рост объема ипотеки при одновременном сворачивании льготных программ и дифференциации ставок в зависимости от количества детей в семье. Прогнозируется восстановление выдач ипотеки в следующем году и нормализация ситуации на рынке.',
#         'sentiment': 'neutral',
#         'tickers_of_interest': ['SBER'],
#         'level_of_potential_impact_on_price': 'none'
#     }
#     find_duplicates(new_event_example)
