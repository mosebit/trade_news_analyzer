"""
ПОЛНОЕ РЕШЕНИЕ С PLAYWRIGHT - ВСЕ ЗАПРОСЫ ЧЕРЕЗ БРАУЗЕР

Установка:
1. pip install playwright beautifulsoup4
2. playwright install chromium
"""

from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
import time
import json
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup


class EDisclosureClient:
    """Клиент для работы с e-disclosure.ru через браузер"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.base_url = "https://www.e-disclosure.ru"
    
    def __enter__(self):
        self._start_browser()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _start_browser(self):
        """Запуск браузера"""
        self.playwright = sync_playwright().start()
        
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        self.context = self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36',
            locale='ru-RU'
        )
        
        self.page = self.context.new_page()
        
        # Скрываем автоматизацию
        self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        print("✓ Браузер запущен")
    
    def get_events_data_by_year(self, company_id: int, year: int) -> List[Dict[str, Any]]:
        """
        Получение списка событий через браузер (перехват API запроса)
        
        Args:
            company_id: ID компании
            year: Год
        
        Returns:
            Список событий
        """
        if not self.page:
            self._start_browser()
        
        events_data = []
        
        # Перехватываем API ответ
        def handle_response(response):
            if '/api/events/page' in response.url and response.status == 200:
                try:
                    data = response.json()
                    events_data.extend(data)
                except:
                    pass
        
        self.page.on('response', handle_response)
        
        # Посещаем страницу компании
        company_url = f"{self.base_url}/portal/company.aspx?id={company_id}"
        print(f"Загружаем страницу компании: {company_url}")
        self.page.goto(company_url, wait_until='domcontentloaded')
        
        # Ждем загрузки anti-bot
        time.sleep(3)
        
        # Ищем и кликаем на год в фильтре (если есть)
        try:
            # Пытаемся найти селектор года или просто ждем загрузки событий
            self.page.wait_for_load_state('networkidle', timeout=10000)
        except:
            pass
        
        time.sleep(2)
        
        # Если события не перехвачены, пробуем получить через JS
        if not events_data:
            print("Пробуем получить события через JavaScript...")
            try:
                # Выполняем JS для получения данных
                api_url = f"{self.base_url}/api/events/page?companyId={company_id}&year={year}"
                events_json = self.page.evaluate(f"""
                    fetch('{api_url}', {{
                        headers: {{
                            'X-Requested-With': 'XMLHttpRequest'
                        }}
                    }})
                    .then(r => r.json())
                    .then(data => JSON.stringify(data))
                """)
                
                if events_json:
                    events_data = json.loads(events_json)
            except Exception as e:
                print(f"Ошибка при JS запросе: {e}")
        
        self.page.remove_listener('response', handle_response)
        
        if events_data:
            print(f"✓ Получено {len(events_data)} событий за {year} год")
            return events_data
        else:
            raise Exception(f"Не удалось получить события за {year} год")
    
    def get_one_event_raw_data(self, event_id: str, company_id: Optional[int] = None) -> str:
        """
        Получение HTML страницы события
        
        Args:
            event_id: ID события (pseudoGUID)
            company_id: ID компании (опционально)
        
        Returns:
            HTML контент страницы события
        """
        if not self.page:
            self._start_browser()
        
        # Если company_id указан, сначала посещаем страницу компании
        if company_id:
            company_url = f"{self.base_url}/portal/company.aspx?id={company_id}"
            print(f"Посещаем страницу компании: {company_url}")
            self.page.goto(company_url, wait_until='domcontentloaded')
            time.sleep(2)
        
        # Переходим на страницу события
        event_url = f"{self.base_url}/portal/event.aspx?EventId={event_id}"
        print(f"Загружаем страницу события: {event_url}")
        self.page.goto(event_url, wait_until='domcontentloaded')
        
        # Ждем загрузки (обход anti-bot)
        print("Ожидание загрузки страницы...")
        max_wait = 15
        start = time.time()
        
        while time.time() - start < max_wait:
            html = self.page.content()
            
            # Проверяем, загрузилась ли реальная страница
            if 'servicepipe.ru' not in html or len(html) > 10000:
                if 'id_spinner' not in html or len(html) > 10000:
                    print(f"✓ Страница загружена за {time.time() - start:.1f}с")
                    break
            
            time.sleep(1)
        
        # Дополнительная пауза
        time.sleep(2)
        html = self.page.content()
        
        # Проверяем результат
        if 'servicepipe.ru' in html and len(html) < 5000:
            print("⚠ Внимание: возможно, anti-bot защита не пройдена")
        else:
            print(f"✓ Успешно получен HTML ({len(html)} символов)")
        
        return html, event_url
    
    def close(self):
        """Закрыть браузер"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print("✓ Браузер закрыт")


# ========== ПРОСТЫЕ ФУНКЦИИ ДЛЯ БЫСТРОГО ИСПОЛЬЗОВАНИЯ ==========

def get_events_data_by_year(company_id: int, year: int, headless: bool = True) -> List[Dict[str, Any]]:
    """
    Получение списка событий компании за год
    
    Args:
        company_id: ID компании
        year: Год
        headless: Запускать браузер в фоне
    
    Returns:
        Список событий
    """
    with EDisclosureClient(headless=headless) as client:
        return client.get_events_data_by_year(company_id, year)


def get_one_event_raw_data(event_id: str, company_id: int, headless: bool = True) -> str:
    """
    Получение HTML страницы события
    
    Args:
        event_id: ID события (pseudoGUID)
        company_id: ID компании
        headless: Запускать браузер в фоне
    
    Returns:
        HTML контент страницы события
    """
    with EDisclosureClient(headless=headless) as client:
        return client.get_one_event_raw_data(event_id, company_id)


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    company_id = 39059
    year = 2025
    
    print("=" * 80)
    print("ВАРИАНТ 1: ИСПОЛЬЗОВАНИЕ КЛАССА (ДЛЯ МНОЖЕСТВЕННЫХ ЗАПРОСОВ)")
    print("=" * 80)
    
    # Открываем браузер один раз для всех операций
    with EDisclosureClient(headless=True) as client:
        print("\n1. Получение списка событий:")
        print("-" * 80)
        
        try:
            events = client.get_events_data_by_year(company_id, year)
            
            if events:
                print(f"\nНайдено событий: {len(events)}")
                print(f"\nПервые 3 события:")
                for i, event in enumerate(events[:3], 1):
                    print(f"\n{i}. {event['eventName']}")
                    print(f"   Дата: {event['eventDate']}")
                    print(f"   ID: {event['pseudoGUID']}")
                
                print("\n2. Получение деталей первого события:")
                print("-" * 80)
                
                event_id = events[0]['pseudoGUID']
                html = client.get_one_event_raw_data(event_id, company_id)
                
                # Анализ результата
                if 'servicepipe.ru' not in html and len(html) > 10000:
                    print(f"\n✓✓✓ УСПЕХ! Страница загружена корректно")
                    print(f"Размер HTML: {len(html)} символов")
                    
                    # Парсим с BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    title = soup.find('title')
                    if title:
                        print(f"Title: {title.text.strip()}")
                    
                    # Сохраняем в файл
                    with open('event_page.html', 'w', encoding='utf-8') as f:
                        f.write(html)
                    print("✓ HTML сохранен в event_page.html")
                
        except Exception as e:
            print(f"\n✗ Ошибка: {e}")
    
    print("\n" + "=" * 80)
    print("ВАРИАНТ 2: ПРОСТЫЕ ФУНКЦИИ (ДЛЯ РАЗОВЫХ ЗАПРОСОВ)")
    print("=" * 80)
    
    try:
        # Каждая функция открывает свой браузер
        print("\n1. Получение событий:")
        events = get_events_data_by_year(company_id, year, headless=True)
        print(f"Получено: {len(events)} событий")
        
        if events:
            print("\n2. Получение деталей:")
            html = get_one_event_raw_data(events[0]['pseudoGUID'], company_id, headless=True)
            print(f"HTML размер: {len(html)} символов")
    
    except Exception as e:
        print(f"✗ Ошибка: {e}")
    
    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print("=" * 80)
    
    print("\n💡 СОВЕТ: Используйте ВАРИАНТ 1 (класс) если делаете много запросов")
    print("   Это быстрее, т.к. браузер открывается только один раз")