import requests
import pandas as pd
from typing import List, Dict, Union

MOEX_CANDLES_URL = "https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"


def load_moex_candles(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: int
) -> pd.DataFrame:
    all_parts = []
    start = 0

    while True:
        params = {
            "from": start_date,
            "till": end_date,
            "interval": interval,
            "start": start,
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


def get_future_prices(
    news_time: Union[str, pd.Timestamp],
    tickers: List[str],
    shifts_hours: List[int] = [1, 3, 12, 24],
    interval_minutes: int = 10,
    price_field: str = "close",
) -> Dict[str, Dict[str, Union[float, None]]]:
    if isinstance(news_time, str):
        news_time = pd.to_datetime(news_time)

    shifts = {f"{h}h": pd.Timedelta(hours=h) for h in shifts_hours}
    max_shift = max(shifts_hours)

    start_date = news_time.date().strftime("%Y-%m-%d")
    end_date = (news_time + pd.Timedelta(hours=max_shift) + pd.Timedelta(days=1)).date().strftime("%Y-%m-%d")

    result: Dict[str, Dict[str, Union[float, None]]] = {}

    for ticker in tickers:
        df = load_moex_candles(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval_minutes,
        )

        if df.empty or price_field not in df.columns:
            result[ticker] = {shift_label: None for shift_label in shifts.keys()}
            continue

        df = df.sort_values("datetime").reset_index(drop=True)

        ticker_res: Dict[str, Union[float, None]] = {}

        for shift_label, delta in shifts.items():
            target_time = news_time + delta
            row = df[df["datetime"] >= target_time].head(1)

            if row.empty:
                price = None
            else:
                price = float(row.iloc[0][price_field])

            ticker_res[shift_label] = price

        result[ticker] = ticker_res

    return result

# example
prices = get_future_prices("2025-12-03 12:00:00", ["SBER", "YDEX", "POSI", "ROSN"])
print(prices)