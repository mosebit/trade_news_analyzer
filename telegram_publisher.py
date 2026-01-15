"""
Telegram Publisher для публикации отчетов
Универсальный форматтер словарей для публикации в Telegram канал/чат
Версия для python-telegram-bot 20+ (async/await)
"""
import os
import asyncio
from typing import Any, Dict
from dotenv import load_dotenv
from telegram import Bot
from telegram.constants import ParseMode

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_TARGET_CHAT_ID = os.getenv("TELEGRAM_TARGET_CHAT_ID")


def _format_value(value: Any, indent: int = 0) -> str:
    """Рекурсивное форматирование значений любого типа"""
    pad = "  " * indent

    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}*{k}:*")
                lines.append(_format_value(v, indent + 1))
            else:
                lines.append(f"{pad}*{k}:* {v}")
        return "\n".join(lines)

    elif isinstance(value, list):
        lines = []
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}▪️ Элемент {i + 1}:")
                lines.append(_format_value(item, indent + 1))
            else:
                lines.append(f"{pad}▪️ {item}")
        return "\n".join(lines)

    else:
        return f"{pad}{value}"


def format_report(report: Dict[str, Any]) -> str:
    """
    Форматирует словарь в красивый текст для Telegram
    Использует Markdown форматирование

    Специальная обработка для поля 'similar_events' - показывает похожие новости
    """
    lines = ["📊 *ОТЧЕТ О ПРОГНОЗЕ*", "━━━━━━━━━━━━━━━━━━━━"]

    # Handle similar_events specially if present
    similar_events_data = report.pop('similar_events', None)

    for key, value in report.items():
        if isinstance(value, (dict, list)):
            lines.append(f"\n*{key.upper()}:*")
            lines.append(_format_value(value, indent=1))
        else:
            lines.append(f"*{key}:* {value}")

    # Add similar events section with better formatting
    if similar_events_data:
        lines.append(f"\n*📰 ПОХОЖИЕ СОБЫТИЯ ИЗ ИСТОРИИ:*")

        if isinstance(similar_events_data, list):
            for i, event in enumerate(similar_events_data[:5], 1):  # Show max 5
                if isinstance(event, dict):
                    lines.append(f"\n  *{i}.* {event.get('description', event.get('title', 'N/A'))}")

                    # Add URL if available (make it clickable)
                    if 'url' in event:
                        lines.append(f"     🔗 [{event['url']}]({event['url']})")
                else:
                    lines.append(f"\n  ▪️ {event}")

    lines.append("\n━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


async def _publish_report_async(report: Dict[str, Any]) -> None:
    """
    Асинхронная публикация отчета в Telegram канал/чат

    Args:
        report: Словарь с данными отчета (структура произвольная)

    Raises:
        ValueError: Если не заданы токен или chat_id в .env
        Exception: При ошибке отправки сообщения
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_TARGET_CHAT_ID:
        raise ValueError(
            "Telegram bot token или target chat id не заданы в .env\n"
            "Добавьте: TELEGRAM_BOT_TOKEN и TELEGRAM_TARGET_CHAT_ID"
        )

    async with Bot(token=TELEGRAM_BOT_TOKEN) as bot:
        try:
            text = format_report(report)
        except Exception as e:
            text = f"⚠️ Ошибка форматирования отчета: {e}"

        # Telegram лимит: 4096 символов на сообщение
        MAX_LENGTH = 4000

        if len(text) <= MAX_LENGTH:
            await bot.send_message(
                chat_id=TELEGRAM_TARGET_CHAT_ID,
                text=text,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            # Разбиваем на части по границам строк
            chunks = []
            while text:
                if len(text) <= MAX_LENGTH:
                    chunks.append(text)
                    break

                split_pos = text.rfind("\n", 0, MAX_LENGTH)
                if split_pos == -1:
                    split_pos = MAX_LENGTH

                chunks.append(text[:split_pos])
                text = text[split_pos:].lstrip()

            # Отправляем части
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await bot.send_message(
                        chat_id=TELEGRAM_TARGET_CHAT_ID,
                        text=chunk,
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await bot.send_message(
                        chat_id=TELEGRAM_TARGET_CHAT_ID,
                        text=f"_Продолжение ({i + 1}/{len(chunks)})_\n{chunk}",
                        parse_mode=ParseMode.MARKDOWN
                    )


def publish_report(report: Dict[str, Any]) -> None:
    """
    Синхронная обертка для публикации отчета
    Использует asyncio для запуска асинхронной функции

    Args:
        report: Словарь с данными отчета (структура произвольная)
    """
    try:
        # Пытаемся получить текущий event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если loop уже запущен, создаем новый в отдельном потоке
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(_publish_report_async(report))
        else:
            loop.run_until_complete(_publish_report_async(report))
    except RuntimeError:
        # Если нет event loop, создаем новый
        asyncio.run(_publish_report_async(report))
