from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
from dotenv import load_dotenv
import json
import requests
import os
from .news_database_chroma import PreparedEvent

TICKERS = ["SBER", "POSI", "ROSN", "YDEX"]

# Загрузка .env
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

YANDEX_IAM_TOKEN = os.getenv('YANDEX_IAM_TOKEN')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_MODEL_URI = os.getenv('YANDEX_MODEL_URI', f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest")

LLM_CONFIG = {
    'base_url': 'https://llm.api.cloud.yandex.net/foundationModels/v1',
    'completion_url': '/completion',
    'embedding_url': '/textEmbedding',
    'model_uri': YANDEX_MODEL_URI,
    'headers': {
        'Authorization': f'Bearer {YANDEX_IAM_TOKEN}',
        'Content-Type': 'application/json',
        'x-folder-id': YANDEX_FOLDER_ID
    },
    'default_temperature': 0.3,
    'default_max_tokens': 2000
}


def yandex_chat_completion(
    messages: List[Dict[str, str]], 
    temperature: float = 0.3, 
    max_tokens: int = 2000,
    response_format: Optional[str] = None
) -> Optional[str]:
    """
    Прямой вызов YandexGPT API (синхронный)
    
    Args:
        messages: List[Dict] с keys 'role' и 'content'
        temperature: Температура (0.0 - 1.0)
        max_tokens: Максимум токенов в ответе
        response_format: "json_object" для JSON режима
    
    Returns:
        Текст ответа или None при ошибке
    """
    
    # Преобразование сообщений: content → text
    payload = {
        "modelUri": LLM_CONFIG['model_uri'],
        "messages": [
            {"role": msg["role"], "text": msg["content"]} 
            for msg in messages
        ],
        "completionOptions": {
            "temperature": temperature,
            "maxTokens": str(max_tokens),
            # Добавлена поддержка reasoning (рекомендуется по документации)
            "reasoningOptions": {
                "mode": "DISABLED"
            }
        }
    }
    
    # ✅ ИСПРАВЛЕНО: jsonObject на верхний уровень payload, НЕ в completionOptions
    if response_format == "json_object":
        payload["jsonObject"] = True
    
    try:
        response = requests.post(
            f"{LLM_CONFIG['base_url']}{LLM_CONFIG['completion_url']}",
            json=payload,
            headers=LLM_CONFIG['headers'],
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        
        # ✓ Эта структура правильна согласно документации
        return data['result']['alternatives'][0]['message']['text']
        
    except requests.exceptions.HTTPError as e:
        print(f"YandexGPT HTTP error {e.response.status_code}: {e.response.text}")
        return None
    except KeyError as e:
        print(f"YandexGPT unexpected response format: {e}")
        print(f"Response data: {data if 'data' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"YandexGPT API error: {e}")
        return None

# import requests
# import numpy as np
# from scipy.spatial.distance import cdist

# FOLDER_ID = "b1gb4tgg41s74ql3b06b"
# IAM_TOKEN = "AQVN2LwQNMuyrIEzP1awzoyF5uxkRqH3Vpbbwz3k"
# doc_uri = f"emb://{FOLDER_ID}/text-search-doc/latest"
# query_uri = f"emb://{FOLDER_ID}/text-search-query/latest"
# embed_url = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
# headers = {"Content-Type": "application/json", "Authorization": f"Bearer {IAM_TOKEN}", "x-folder-id": f"{FOLDER_ID}"}
# doc_texts = [
#   """Александр Сергеевич Пушкин (26 мая [6 июня] 1799, Москва — 29 января [10 февраля] 1837, Санкт-Петербург) — русский поэт, драматург и прозаик, заложивший основы русского реалистического направления, литературный критик и теоретик литературы, историк, публицист, журналист.""",
#   """Ромашка — род однолетних цветковых растений семейства астровые, или сложноцветные, по современной классификации объединяет около 70 видов невысоких пахучих трав, цветущих с первого года жизни."""
# ]
# query_text = "когда день рождения Пушкина?"

# # Создаем эмбеддинг запроса
# def get_embedding(text: str, text_type: str = "doc") -> np.array:
#     query_data = {
#         "modelUri": doc_uri if text_type == "doc" else query_uri,
#         "text": text,
#     }

#     request_res = requests.post(embed_url, json=query_data, headers=headers)

#     print(request_res.json())

#     return np.array(
#         request_res.json()["embedding"]
#     )

# # Создаем эмбеддинг текстов
# query_embedding = get_embedding(query_text, text_type="doc")
# docs_embedding = [get_embedding(doc_text) for doc_text in doc_texts]

# # Вычисляем косинусное расстояние
# dist = cdist(query_embedding[None, :], docs_embedding, metric="cosine")

# # Вычисляем косинусное сходство
# sim = 1 - dist

# # most similar doc text
# print(doc_texts[np.argmax(sim)])

