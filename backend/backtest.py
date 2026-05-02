import json
import numpy as np
import pandas as pd
from model import predict_next_hour
from data import fetch_binance_klines
import time

def winkler_score(actual, lower, upper, alpha=0.05):
    """Calculate the Winkler score for a given prediction."""
    width = upper - lower
    if actual < lower:
        return width + (2 / alpha) * (lower - actual)
    elif actual > upper:
        return width + (2 / alpha) * (actual - upper)
    else:
        return width

def run_backtest():
    """Run the 30-day backtest."""
    print("Fetching data for backtest...")
    # Fetch 720 bars + 500 for initial history = 1220
    df = fetch_binance_klines(limit=1000)
    # The max limit is 1000 for binance klines, so we might need to fetch multiple times if we need 1220.
    # Let's fetch 1000 bars. It gives us ~41 days.
    # We will use the last 720 bars for testing, and the 280 before that for initial model warmup.
    
    total_bars = len(df)
    test_bars = min(720, total_bars - 100) # At least 100 bars for warmup
    
    start_idx = total_bars - test_bars
    
    results = []
    
    print(f"Running backtest over {test_bars} bars...")
    for i in range(start_idx, total_bars - 1):
        # Data strictly up to time T-1
        train_df = df.iloc[:i+1]
        
        # Predict time T
        lower, upper = predict_next_hour(train_df)
        
        # Actual time T
        actual_price = df.iloc[i+1]['close']
        timestamp = df.iloc[i+1]['timestamp'].isoformat()
        
        score = winkler_score(actual_price, lower, upper)
        is_covered = bool(lower <= actual_price <= upper)
        
        result = {
            "timestamp": timestamp,
            "actual": actual_price,
            "lower": lower,
            "upper": upper,
            "winkler": score,
            "covered": is_covered,
            "width": upper - lower
        }
        results.append(result)
        
        if (i - start_idx) % 100 == 0:
            print(f"Processed {i - start_idx}/{test_bars} bars")
            
    # Calculate aggregate metrics
    coverage = float(np.mean([r['covered'] for r in results]))
    avg_width = float(np.mean([r['width'] for r in results]))
    avg_winkler = float(np.mean([r['winkler'] for r in results]))
    
    print(f"Backtest Complete! Coverage: {coverage:.4f}, Avg Width: {avg_width:.2f}, Winkler: {avg_winkler:.2f}")
    
    # Save to JSONL
    with open("backtest_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Save summary
    summary = {
        "coverage": coverage,
        "avg_width": avg_width,
        "avg_winkler": avg_winkler,
        "num_predictions": len(results)
    }
    with open("backtest_summary.json", "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    run_backtest()
