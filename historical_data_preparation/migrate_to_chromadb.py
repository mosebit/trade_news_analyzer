"""
Скрипт миграции данных из SQLite + sqlite-vec в ChromaDB

Использование:
    python migrate_to_chroma.py path/to/news_data.db
    python migrate_to_chroma.py path/to/news_data.db --chroma-path ./new_chroma_db
"""

import sqlite3
import chromadb
import json
import sys
import argparse
from typing import Dict, List
from tqdm import tqdm


def load_news_from_sqlite(db_path: str) -> List[Dict]:
    """
    Загружает все новости из SQLite базы данных.
    
    Args:
        db_path: путь к SQLite базе данных
        
    Returns:
        список словарей с данными новостей
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Получаем все новости с тикерами
    query = """
        SELECT 
            n.id,
            n.url,
            n.title,
            n.original_text,
            n.clean_description,
            n.sentiment,
            n.impact_level,
            n.published_date,
            n.published_timestamp,
            GROUP_CONCAT(nt.ticker) as tickers
        FROM news n
        LEFT JOIN news_tickers nt ON n.id = nt.news_id
        GROUP BY n.id
        ORDER BY n.id
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    news_list = []
    for row in rows:
        news_list.append({
            'id': row[0],
            'url': row[1],
            'title': row[2],
            'original_text': row[3] or '',
            'clean_description': row[4] or '',
            'sentiment': row[5] or 'neutral',
            'impact_level': row[6] or 'none',
            'published_date': row[7],
            'published_timestamp': row[8],
            'tickers': row[9].split(',') if row[9] else []
        })
    
    conn.close()
    return news_list


def migrate_to_chroma(sqlite_path: str, chroma_path: str = "./chroma_db"):
    """
    Мигрирует данные из SQLite в ChromaDB.
    
    Args:
        sqlite_path: путь к исходной SQLite базе
        chroma_path: путь для новой ChromaDB базы
    """
    print(f"🔄 Начало миграции из {sqlite_path} в {chroma_path}")
    
    # Загружаем данные из SQLite
    print("\n📖 Загрузка данных из SQLite...")
    news_list = load_news_from_sqlite(sqlite_path)
    print(f"✓ Загружено {len(news_list)} новостей")
    
    if len(news_list) == 0:
        print("⚠ База данных пуста, нечего мигрировать")
        return
    
    # Инициализация ChromaDB
    print(f"\n📦 Создание ChromaDB в {chroma_path}...")
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Удаляем коллекцию если существует (для чистой миграции)
    try:
        client.delete_collection("news")
        print("✓ Старая коллекция удалена")
    except:
        pass
    
    # collection = client.create_collection("news")
    collection = client.get_or_create_collection("news_cosine", metadata={"hnsw:space": "cosine"})

    print("✓ Новая коллекция создана")
    
    # Миграция данных
    print(f"\n🚀 Миграция {len(news_list)} новостей...")
    
    # ChromaDB поддерживает батчинг - мигрируем по 100 записей
    batch_size = 100
    migrated = 0
    errors = 0
    
    for i in tqdm(range(0, len(news_list), batch_size), desc="Миграция"):
        batch = news_list[i:i+batch_size]
        
        ids = []
        documents = []
        metadatas = []
        
        for news in batch:
            try:
                # ID - используем URL (уникальный идентификатор)
                ids.append(news['url'])
                
                # Document - очищенное описание
                documents.append(news['clean_description'])
                
                # Metadata - все остальные поля
                original_text = news['original_text'][:3500] if news['original_text'] else ''
                
                # Создаем enriched_data для совместимости
                enriched_data = {
                    'clean_description': news['clean_description'],
                    'sentiment': news['sentiment'],
                    'tickers_of_interest': news['tickers'],
                    'level_of_potential_impact_on_price': news['impact_level']
                }
                
                # Базовые метаданные
                metadata = {
                    'title': news['title'] or '',
                    'original_text': original_text,
                    'tickers': ','.join(news['tickers']),
                    'sentiment': news['sentiment'],
                    'impact': news['impact_level'],
                    'published_date': news['published_date'] or '',
                    'timestamp': news['published_timestamp'] or 0,
                    'enriched_json': json.dumps(enriched_data, ensure_ascii=False),
                    'sqlite_id': news['id']
                }
                
                # НОВЫЙ ПОДХОД: добавляем TICKER_impact только для упомянутых тикеров
                for ticker in news['tickers']:
                    if ticker.strip():  # Проверка что тикер не пустой
                        metadata[f'{ticker.strip()}_impact'] = news['impact_level']
                
                metadatas.append(metadata)
                migrated += 1
                
            except Exception as e:
                print(f"\n✗ Ошибка при обработке новости {news.get('url', 'unknown')}: {e}")
                errors += 1
        
        # Добавляем батч в ChromaDB
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"\n✗ Ошибка при добавлении батча: {e}")
            errors += len(batch)
    
    # Финальная статистика
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ МИГРАЦИИ:")
    print(f"  ✓ Успешно мигрировано: {migrated} новостей")
    if errors > 0:
        print(f"  ✗ Ошибок: {errors}")
    print(f"  📍 ChromaDB база: {chroma_path}")
    
    # Проверка
    final_count = collection.count()
    print(f"  🔍 Записей в ChromaDB: {final_count}")
    
    if final_count == migrated:
        print("\n✅ Миграция завершена успешно!")
    else:
        print(f"\n⚠ Предупреждение: ожидалось {migrated}, найдено {final_count}")
    
    # Проверка структуры данных
    print("\n🔍 Проверка структуры метаданных...")
    sample = collection.get(limit=1, include=['metadatas'])
    if sample['ids']:
        print("Пример метаданных первой записи:")
        metadata_keys = list(sample['metadatas'][0].keys())
        print(f"  Поля: {', '.join(metadata_keys)}")
        
        # Показываем TICKER_impact поля
        ticker_fields = [k for k in metadata_keys if k.endswith('_impact')]
        if ticker_fields:
            print(f"  TICKER_impact поля: {', '.join(ticker_fields)}")
    
    print("="*50)
    
    # Примеры запросов
    print("\n💡 Примеры использования новой базы:")
    print(f"""
from news_database_chroma import NewsDatabase

db = NewsDatabase(path="{chroma_path}")

# Получить все новости по тикеру
sber_news = db.get_news_by_ticker("SBER", limit=10)

# Получить только важные новости
sber_important = db.get_news_by_ticker("SBER", limit=10, min_impact='high')

# Статистика
stats = db.get_stats()
print(stats)
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Миграция данных из SQLite + sqlite-vec в ChromaDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python migrate_to_chroma.py news_data.db
  python migrate_to_chroma.py news_data.db --chroma-path ./my_chroma_db
  python migrate_to_chroma.py /path/to/news_data.db --chroma-path /path/to/chroma
        """
    )
    
    parser.add_argument(
        'sqlite_db',
        help='Путь к исходной SQLite базе данных'
    )
    
    parser.add_argument(
        '--chroma-path',
        default='./chroma_db',
        help='Путь для новой ChromaDB базы (по умолчанию: ./chroma_db)'
    )
    
    args = parser.parse_args()
    
    # Проверка существования исходной БД
    import os
    if not os.path.exists(args.sqlite_db):
        print(f"❌ Ошибка: файл {args.sqlite_db} не найден")
        sys.exit(1)
    
    # Запуск миграции
    try:
        migrate_to_chroma(args.sqlite_db, args.chroma_path)
    except Exception as e:
        print(f"\n❌ Критическая ошибка миграции: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()