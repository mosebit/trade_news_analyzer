import requests
import pandas as pd
from typing import List, Dict, Union

MOEX_CANDLES_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"


def load_moex_candles(
    ticker: str,
    start_date: str,
    end_date: str,
    interval_minutes: int = None,
    interval_days: int = None,
) -> pd.DataFrame:
    """
    Загружает свечи с MOEX за указанный период.
    
    Args:
        ticker: Тикер акции (например, "SBER")
        start_date: Дата начала в формате "YYYY-MM-DD"
        end_date: Дата окончания в формате "YYYY-MM-DD"
        interval_minutes: Интервал в минутах (1, 10, 60 и т.д.)
        interval_days: Интервал в днях (7, 31 и т.д.)
        
    Returns:
        DataFrame со свечами
    """
    # Проверка, что задан ровно один тип интервала
    if interval_minutes is None and interval_days is None:
        raise ValueError("Необходимо указать interval_minutes или interval_days")
    
    if interval_minutes is not None and interval_days is not None:
        raise ValueError("Нужно указать только один параметр: interval_minutes или interval_days")
    
    # Определяем interval для API MOEX
    if interval_minutes is not None:
        interval = interval_minutes
    else:
        # MOEX API использует специальные значения для дневных интервалов:
        # 24 = 1 день, 7 = неделя, 31 = месяц, 4 = квартал
        if interval_days == 1:
            interval = 24
        elif interval_days == 7:
            interval = 7
        elif interval_days in [30, 31]:
            interval = 31
        elif interval_days in [90, 91]:
            interval = 4
        else:
            # Для других значений используем дневной интервал
            interval = 24
    
    all_parts = []
    start = 0

    while True:
        params = {
            "from": start_date,
            "till": end_date,
            "interval": interval,
            "start": start,   # пагинация
        }
        url = MOEX_CANDLES_URL.format(ticker=ticker)
        r = requests.get(url, params=params)
        r.raise_for_status()
        j = r.json()

        candles = j.get("candles", {})
        cols = candles.get("columns", [])
        data = candles.get("data", [])

        if not data:
            break

        df_part = pd.DataFrame(data, columns=cols)
        all_parts.append(df_part)
        start += len(df_part)

    if not all_parts:
        return pd.DataFrame()

    df = pd.concat(all_parts, ignore_index=True)

    df["datetime"] = pd.to_datetime(df["end"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df

if __name__ == "__main__":
    # prices = get_future_price_changes("2025-12-03 12:00:00", ["SBER"])
    # print(prices)

    # Интервал 10 минут
    df = load_moex_candles("SBER", "2025-01-01", "2025-01-31", interval_minutes=10)
    close_prices = df['close'].tolist()
    # Интервал 1 час
    df = load_moex_candles("SBER", "2025-01-01", "2025-01-31", interval_minutes=60)

    # Дневные свечи
    df = load_moex_candles("SBER", "2024-01-01", "2025-01-31", interval_days=1)

    # Недельные свечи
    df = load_moex_candles("SBER", "2024-01-01", "2025-01-31", interval_days=7)

    # Месячные свечи
    df = load_moex_candles("SBER", "2020-01-01", "2025-01-31", interval_days=31)