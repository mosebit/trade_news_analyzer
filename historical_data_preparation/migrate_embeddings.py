"""
Миграция эмбеддингов: копирование данных из старой БД в новую с новыми эмбеддингами.
"""
import sys
from pathlib import Path
from typing import List
import json

# Добавляем родительскую директорию в путь для импорта
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Теперь можем импортировать из пакета
from historical_data_preparation.news_database_chroma import NewsDatabase
from historical_data_preparation import ai_enrichers_and_filters

def migrate_embeddings(
    old_db_path: str = "./chroma_db",
    new_db_path: str = "./chroma_db_new",
    batch_size: int = 100
):
    """
    Миграция эмбеддингов из старой БД в новую.
    
    Args:
        old_db_path: путь к старой базе данных ChromaDB
        new_db_path: путь к новой базе данных ChromaDB
        batch_size: размер батча для обработки
    """
    print(f"Миграция эмбеддингов:")
    print(f"  Из: {old_db_path}")
    print(f"  В:  {new_db_path}")
    
    # Открываем старую БД для чтения
    old_db = NewsDatabase(old_db_path)
    
    # Открываем новую БД для записи
    new_db = NewsDatabase(new_db_path)
    
    # Получаем все записи из старой БД
    try:
        result = old_db.collection.get(
            limit=1000000,
            include=['metadatas', 'documents']
        )
        
        ids = result.get('ids', [])
        metadatas = result.get('metadatas', [])
        documents = result.get('documents', [])
        
        total_count = len(ids)
        print(f"\nНайдено записей для миграции: {total_count}")
        
        if total_count == 0:
            print("Нет записей для миграции")
            return
        
        # Обрабатываем батчами
        success_count = 0
        error_count = 0
        
        for i in range(0, total_count, batch_size):
            batch_end = min(i + batch_size, total_count)
            batch_ids = ids[i:batch_end]
            batch_docs = documents[i:batch_end]
            batch_metas = metadatas[i:batch_end]
            
            print(f"\nОбработка батча {i//batch_size + 1}: записи {i+1}-{batch_end} из {total_count}")
            
            # Обрабатываем каждую запись в батче
            for j, url in enumerate(batch_ids):
                try:
                    # Получаем новый эмбеддинг для документа
                    new_embedding = ai_enrichers_and_filters.get_embedding(batch_docs[j])
                    
                    # Сохраняем в новую БД
                    new_db.collection.add(
                        ids=[url],
                        embeddings=[new_embedding],
                        documents=[batch_docs[j]],
                        metadatas=[batch_metas[j]]
                    )
                    
                    success_count += 1
                    
                    if (j + 1) % 10 == 0:
                        print(f"  Обработано {j + 1}/{len(batch_ids)} в текущем батче")
                        
                except Exception as e:
                    error_count += 1
                    print(f"  Ошибка обработки записи {url}: {e}")
                    continue
            
            print(f"Батч {i//batch_size + 1} завершён (успешно: {success_count}, ошибок: {error_count})")
        
        print(f"\n{'='*60}")
        print(f"✓ Миграция завершена!")
        print(f"  Успешно обработано: {success_count}")
        print(f"  Ошибок: {error_count}")
        print(f"  Всего: {total_count}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Критическая ошибка при чтении базы данных: {e}")
        return
    finally:
        old_db.close()
        new_db.close()


if __name__ == "__main__":
    # Укажите пути к старой и новой базам данных
    OLD_DB_PATH = "./chroma_db_new_old_embeddings"
    NEW_DB_PATH = "./chroma_db_new"
    
    # Запускаем миграцию
    migrate_embeddings(
        old_db_path=OLD_DB_PATH,
        new_db_path=NEW_DB_PATH,
        batch_size=100
    )
