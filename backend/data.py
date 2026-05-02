import requests
import pandas as pd
from typing import Optional

def fetch_binance_klines(symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 1000, end_time: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch klines (candlesticks) from Binance API.
    Returns a DataFrame with columns: timestamp, open, high, low, close, volume, etc.
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df
