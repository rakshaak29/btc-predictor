import numpy as np
import pandas as pd
from scipy.stats import t

def predict_next_hour_full(df: pd.DataFrame, num_simulations: int = 10000, alpha: float = 0.05) -> dict:
    """
    Predicts the next hour 95% confidence interval for BTC closing price.
    Returns a rich dict with lower, upper, simulated_prices, volatility data, and confidence level.

    `df` must strictly contain data UP TO time T (no peeking into the future).
    """
    closes = df['close'].values
    log_returns = np.diff(np.log(closes))

    # --- Volatility Clustering via EWMA ---
    returns_series = pd.Series(log_returns)
    ewma_vol = returns_series.ewm(span=24, adjust=False).std().values[-1]
    if np.isnan(ewma_vol) or ewma_vol == 0:
        ewma_vol = np.std(log_returns)

    # --- Get last 48 hours of hourly volatility for chart ---
    rolling_vol = returns_series.rolling(window=10).std().fillna(method='bfill').values
    vol_threshold = float(np.percentile(rolling_vol, 70))
    last_48_vol = rolling_vol[-48:].tolist()

    # --- Determine confidence level ---
    current_vol = float(rolling_vol[-1])
    if current_vol < vol_threshold * 0.7:
        confidence = "High"
        confidence_msg = "Recent volatility is low — model predictions are more reliable"
        confidence_color = "green"
    elif current_vol < vol_threshold:
        confidence = "Medium"
        confidence_msg = "Moderate volatility — predictions are reasonably reliable"
        confidence_color = "yellow"
    else:
        confidence = "Low"
        confidence_msg = "High volatility detected — wider ranges, less certainty"
        confidence_color = "red"

    # --- Fit Student-t for fat tails ---
    recent_returns = log_returns[-100:]
    std_dev = np.std(recent_returns)
    if std_dev > 0:
        standardized = recent_returns / std_dev
        try:
            df_t, loc, scale = t.fit(standardized)
            df_t = float(np.clip(df_t, 2.1, 10.0))
        except Exception:
            df_t = 3.0
    else:
        df_t = 3.0

    # --- Monte Carlo Simulation ---
    current_price = float(closes[-1])
    drift = float(np.mean(log_returns))
    simulated_std_returns = t.rvs(df_t, size=num_simulations)
    simulated_log_returns = drift + (simulated_std_returns * ewma_vol)
    simulated_prices = current_price * np.exp(simulated_log_returns)

    lower_bound = float(np.percentile(simulated_prices, (alpha / 2) * 100))
    upper_bound = float(np.percentile(simulated_prices, (1 - alpha / 2) * 100))

    # Sample a subset of sim prices for the histogram (300 bins worth of data)
    histogram, bin_edges = np.histogram(simulated_prices, bins=60)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

    return {
        "lower": lower_bound,
        "upper": upper_bound,
        "current_price": current_price,
        "width": upper_bound - lower_bound,
        "confidence": confidence,
        "confidence_msg": confidence_msg,
        "confidence_color": confidence_color,
        "histogram": {
            "counts": histogram.tolist(),
            "bin_centers": bin_centers,
        },
        "volatility": {
            "last_48": last_48_vol,
            "threshold": vol_threshold,
            "current": current_vol,
        },
        "ewma_vol": float(ewma_vol),
        "df_t": df_t,
    }


def predict_next_hour(df: pd.DataFrame, num_simulations: int = 10000, alpha: float = 0.05):
    """Slim version for backtest (returns just lower, upper)."""
    result = predict_next_hour_full(df, num_simulations, alpha)
    return result["lower"], result["upper"]
