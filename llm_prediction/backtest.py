import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, List, Dict, Optional, Tuple
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from historical_data_preparation.news_database_chroma import NewsDatabase, PreparedEvent
from historical_data_preparation.ai_enrichers_and_filters import chat_completion
from historical_data_preparation.future_price_moex import load_moex_candles


def get_price_change_3h(ticker: str, event_timestamp: int) -> Optional[float]:
    event_time = datetime.fromtimestamp(event_timestamp)
    event_time_str = event_time.strftime("%Y-%m-%d %H:%M:%S")

    end_time = event_time + timedelta(hours=4)
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    df = load_moex_candles(
        ticker,
        event_time_str,
        end_time_str,
        interval_minutes=10
    )

    if df.empty:
        return None

    initial_price = df.iloc[0]['close']

    target_time = event_time + timedelta(hours=3)
    df['time_diff'] = abs((df['datetime'] - target_time).dt.total_seconds())
    closest_idx = df['time_diff'].idxmin()
    final_price = df.loc[closest_idx, 'close']

    price_change_pct = ((final_price - initial_price) / initial_price) * 100
    return price_change_pct


def classify_price_change(price_change_pct: float, threshold: float = 0.5) -> str:
    if price_change_pct > threshold:
        return 'up'
    elif price_change_pct < -threshold:
        return 'down'
    else:
        return 'stable'


def find_similar_events_before(
    db: NewsDatabase,
    current_event: PreparedEvent,
    limit: int = 5,
) -> List[PreparedEvent]:
    similar_events = db.find_similar_news_by_event_new(
        event=current_event,
        limit=limit * 5,
        threshold=0.30
    )

    current_tickers = set[Any](current_event.tickers)
    filtered = []

    for event in similar_events:
        if event.timestamp >= current_event.timestamp:
            continue

        event_tickers = set(event.tickers)
        if not current_tickers.intersection(event_tickers):
            continue

        filtered.append(event)

        if len(filtered) >= limit:
            break

    return filtered


def generate_prediction(
    current_event: PreparedEvent,
    similar_events: List[PreparedEvent]
) -> Optional[Dict]:
    similar_context = []
    for event in similar_events[:5]:
        similar_context.append({
            'title': event.title,
            'description': event.clean_description[:200],
            'timestamp': event.published_date,
            'tickers': event.tickers,
            'sentiment': event.sentiment,
            'impact': event.impact
        })

    prompt = f"""Проанализируй новость и предскажи изменение цены акции на основе похожих событий из прошлого.

ТЕКУЩАЯ НОВОСТЬ:
Заголовок: {current_event.title}
Описание: {current_event.clean_description}
Тикеры: {', '.join(current_event.tickers)}
Sentiment: {current_event.sentiment}
Impact: {current_event.impact}

ПОХОЖИЕ СОБЫТИЯ ИЗ ПРОШЛОГО:
{json.dumps(similar_context, ensure_ascii=False, indent=2)}

На основе анализа текущей новости и похожих событий из прошлого, предскажи:
1. price_change_prediction: "up" (рост > 0.5%), "down" (падение < -0.5%), или "stable" (от -0.5% до +0.5%)
2. confidence_level: "low", "medium", или "high"
3. reasoning: краткое обоснование прогноза (1-2 предложения)

ВАЖНО: Верни ТОЛЬКО валидный JSON без markdown разметки.

Формат:
{{
  "price_change_prediction": "up/down/stable",
  "confidence_level": "low/medium/high",
  "reasoning": "обоснование прогноза"
}}
"""

    messages = [
        {
            "role": "system",
            "content": "Ты эксперт по анализу финансовых новостей и прогнозированию цен акций. Ты ВСЕГДА отвечаешь чистым JSON без дополнительного текста."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = chat_completion(
            messages,
            temperature=0.2,
            response_format="json_object",
            retry_on_json_error=True
        )
        if not response:
            return None

        prediction = json.loads(response)
        return prediction
    except Exception:
        return None


def run_backtest(
    tickers: List[str],
    sample_sizes: Dict[str, int],
    db_path: str = './historical_data_preparation/chroma_db_new'
) -> pd.DataFrame:
    db = NewsDatabase(path=db_path)

    all_results = []
    ticker_summaries = {}

    for ticker in tickers:
        sample_size = sample_sizes.get(ticker, 0)
        if sample_size == 0:
            continue

        all_news = db.get_news_by_ticker(ticker, limit=1000)

        news_events = []
        for news_dict in all_news:
            try:
                metadata_result = db.collection.get(ids=[news_dict['url']], include=['metadatas'])
                if not metadata_result['metadatas']:
                    continue

                metadata = metadata_result['metadatas'][0]
                timestamp = metadata.get('timestamp', 0)

                if timestamp == 0:
                    continue

                event = PreparedEvent(
                    url=news_dict['url'],
                    title=news_dict.get('title', ''),
                    clean_description=news_dict.get('clean_description', ''),
                    original_text='',
                    tickers=news_dict.get('tickers', []),
                    sentiment=news_dict.get('sentiment', 'neutral'),
                    impact=news_dict.get('impact_level', 'none'),
                    published_date=metadata.get('published_date', ''),
                    timestamp=timestamp
                )
                news_events.append(event)
            except Exception as e:
                continue

        news_events_sorted = sorted(news_events, key=lambda x: x.timestamp)
        test_events = news_events_sorted[-sample_size:]

        ticker_results = []

        for event in test_events:
            similar_events = find_similar_events_before(db, event, limit=10)

            prediction = generate_prediction(event, similar_events)

            if not prediction:
                continue

            actual_change_pct = get_price_change_3h(ticker, event.timestamp)

            if actual_change_pct is None:
                continue

            actual_class = classify_price_change(actual_change_pct)
            predicted_class = prediction['price_change_prediction']
            is_correct = (predicted_class == actual_class)

            ticker_results.append({
                'ticker': ticker,
                'url': event.url,
                'predicted': predicted_class,
                'actual': actual_class,
                'actual_change_pct': actual_change_pct,
                'confidence': prediction['confidence_level'],
                'correct': is_correct
            })

        all_results.extend(ticker_results)

        df_ticker = pd.DataFrame(ticker_results)

        actual_up = (df_ticker['actual'] == 'up').sum()
        actual_down = (df_ticker['actual'] == 'down').sum()
        actual_stable = (df_ticker['actual'] == 'stable').sum()
        total = len(df_ticker)

        pred_up = (df_ticker['predicted'] == 'up').sum()
        pred_down = (df_ticker['predicted'] == 'down').sum()
        pred_stable = (df_ticker['predicted'] == 'stable').sum()

        correct_up = ((df_ticker['predicted'] == 'up') & (df_ticker['actual'] == 'up')).sum()
        correct_down = ((df_ticker['predicted'] == 'down') & (df_ticker['actual'] == 'down')).sum()
        correct_stable = ((df_ticker['predicted'] == 'stable') & (df_ticker['actual'] == 'stable')).sum()

        ticker_summaries[ticker] = {
            'total': total,
            'actual': {
                'up': actual_up / total * 100,
                'down': actual_down / total * 100,
                'stable': actual_stable / total * 100
            },
            'predicted': {
                'up': pred_up / total * 100,
                'down': pred_down / total * 100,
                'stable': pred_stable / total * 100
            },
            'correct': {
                'up': (correct_up / pred_up * 100) if pred_up > 0 else 0,
                'down': (correct_down / pred_down * 100) if pred_down > 0 else 0,
                'stable': (correct_stable / pred_stable * 100) if pred_stable > 0 else 0
            },
            'overall_accuracy': (df_ticker['correct'].sum() / total * 100)
        }

    results_df = pd.DataFrame(all_results)

    print("\nRESULT METRICS\n")

    total_tested = len(results_df)
    total_correct = results_df['correct'].sum()
    overall_acc = (total_correct / total_tested) * 100

    print(f"Total news tested: {total_tested}")
    print(f"Overall accuracy: {overall_acc:.1f}% ({total_correct}/{total_tested})")

    for ticker in tickers:
        if ticker not in ticker_summaries:
            continue

        summary = ticker_summaries[ticker]

        print(f"\n{ticker} - {summary['total']} news (Overall Accuracy: {summary['overall_accuracy']:.1f}%)")
        print("─" * 60)
        print(f"{'':12s} │ {'UP':>8s} │ {'DOWN':>8s} │ {'STABLE':>8s}")
        print("─" * 60)
        print(f"{'Actual':12s} │ {summary['actual']['up']:7.1f}% │ {summary['actual']['down']:7.1f}% │ {summary['actual']['stable']:7.1f}%")
        print(f"{'Predicted':12s} │ {summary['predicted']['up']:7.1f}% │ {summary['predicted']['down']:7.1f}% │ {summary['predicted']['stable']:7.1f}%")
        print(f"{'Correct':12s} │ {summary['correct']['up']:7.1f}% │ {summary['correct']['down']:7.1f}% │ {summary['correct']['stable']:7.1f}%")


if __name__ == "__main__":
    sample_sizes = {'SBER': 41, 'POSI': 7, 'ROSN': 35, 'YDEX': 17}
    tickers = ['SBER', 'POSI', 'ROSN', 'YDEX']
    run_backtest(tickers, sample_sizes)
