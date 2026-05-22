import os
import requests
import pandas as pd
import time

def fetch_binance_klines(symbol="BTCUSDT", interval="5m", limit=1000, end_time=None):
    """
    Fetch a single batch of klines from the Binance API.
    Uses endTime to allow backward pagination.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if end_time:
        params["endTime"] = end_time
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_historical_dataset(symbol="BTCUSDT", interval="5m", total_candles=5000):
    """
    Fetch a large historical dataset of klines by looping in batches of 1000 backwards in time.
    """
    print(f"Fetching {total_candles} candles for {symbol} ({interval}) from Binance...")
    all_klines = []
    current_end = None
    
    batch_size = 1000
    while len(all_klines) < total_candles:
        remaining = total_candles - len(all_klines)
        limit = min(batch_size, remaining)
        
        try:
            klines = fetch_binance_klines(symbol, interval, limit, current_end)
            if not klines:
                break
            
            # Prepend the older klines to keep chronological order
            all_klines = klines + all_klines
            
            # The next batch must end 1ms before the oldest candle in the current batch
            first_open_time = klines[0][0]
            current_end = first_open_time - 1
            
            print(f"Fetched {len(all_klines)}/{total_candles} candles...")
            # Small delay to respect API limits
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching data from Binance: {e}")
            break
            
    # Format raw API response into a pandas DataFrame
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    # In case we fetched slightly more due to batch alignment, slice to match requested amount
    df = pd.DataFrame(all_klines[-total_candles:], columns=columns)
    
    # Cast numerical fields to float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
        
    # Convert timestamps to human-readable dates
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    return df

if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "5m"
    total_candles = 5000
    
    df = fetch_historical_dataset(symbol, interval, total_candles)
    
    # Ensure dataset directory exists
    os.makedirs("dataset", exist_ok=True)
    csv_path = os.path.join("dataset", "btc_5m_historical.csv")
    
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved {len(df)} rows of data to {csv_path}!")
