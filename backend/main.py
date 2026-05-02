from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sqlite3
import json
import csv
import io
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import t as student_t, skew, kurtosis
from data import fetch_binance_klines
from model import predict_next_hour_full

app = FastAPI()

DB_FILE = "predictions.db"

# ─── Database Setup ──────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            current_price REAL,
            lower_bound   REAL,
            upper_bound   REAL,
            actual_price  REAL,
            hit           INTEGER,
            created_at    TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ─── API: Full current prediction ────────────────────────────────────────────
@app.get("/api/prediction/current")
def get_current_prediction(currency: str = Query("USD")):
    try:
        df = fetch_binance_klines(limit=500)
        result = predict_next_hour_full(df)

        timestamp = datetime.now().isoformat()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (timestamp, current_price, lower_bound, upper_bound, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, result["current_price"], result["lower"], result["upper"], timestamp))
        conn.commit()
        conn.close()

        chart_slice = df.tail(51)[['timestamp', 'open', 'high', 'low', 'close']].copy()
        chart_slice['timestamp'] = chart_slice['timestamp'].dt.strftime('%m/%d %H:%M')
        chart_data = chart_slice.to_dict(orient='records')

        return {
            "current_price": result["current_price"],
            "prediction": {
                "lower": result["lower"],
                "upper": result["upper"],
                "width": result["width"],
            },
            "confidence": result["confidence"],
            "confidence_msg": result["confidence_msg"],
            "confidence_color": result["confidence_color"],
            "histogram": result["histogram"],
            "volatility": result["volatility"],
            "ewma_vol": result["ewma_vol"],
            "df_t": result["df_t"],
            "chart_data": chart_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── API: Backtest Metrics ────────────────────────────────────────────────────
@app.get("/api/metrics")
def get_metrics():
    if os.path.exists("backtest_summary.json"):
        with open("backtest_summary.json", "r") as f:
            return json.load(f)
    return {"coverage": None, "avg_width": None, "avg_winkler": None, "message": "Backtest not run yet"}


# ─── API: Backtest Detailed Results ──────────────────────────────────────────
@app.get("/api/backtest/results")
def get_backtest_results(limit: int = Query(50, ge=1, le=720)):
    if not os.path.exists("backtest_results.jsonl"):
        return []
    rows = []
    with open("backtest_results.jsonl", "r") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows[-limit:]


# ─── API: Live Prediction History ────────────────────────────────────────────
@app.get("/api/prediction/history")
def get_prediction_history(limit: int = Query(50, ge=1, le=200)):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT id, timestamp, current_price, lower_bound, upper_bound, actual_price, hit '
        'FROM predictions ORDER BY id DESC LIMIT ?', (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "timestamp": r[1], "price": r[2], "lower": r[3], "upper": r[4], "actual": r[5], "hit": r[6]} for r in rows]


# ─── API: Export CSV ──────────────────────────────────────────────────────────
@app.get("/api/prediction/history/csv")
def download_history_csv():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, current_price, lower_bound, upper_bound, actual_price, hit FROM predictions ORDER BY id DESC LIMIT 200')
    rows = cursor.fetchall()
    conn.close()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Price at Prediction", "Lower Bound", "Upper Bound", "Actual Price", "Hit"])
    for r in rows:
        writer.writerow(r)
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=prediction_history.csv"})


# ═══════════════════════════════════════════════════════════════════════
# ── UNIQUE FEATURE 1: Market Regime Classifier ────────────────────────
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/regime")
def get_market_regime():
    """
    Classify the current BTC market regime using multiple signals:
    - Momentum (ADX proxy via directional moves)
    - Mean-reversion tendency (z-score of price vs rolling mean)
    - Volatility acceleration (EWMA vol change rate)
    - Price structure (consecutive green/red bars)
    """
    try:
        df = fetch_binance_klines(limit=200)
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        log_returns = np.diff(np.log(closes))

        # ── Momentum Score ─────────────────────────────────────────────
        # Rolling 10-bar return
        rolling_10 = (closes[-1] - closes[-11]) / closes[-11]
        rolling_24 = (closes[-1] - closes[-25]) / closes[-25]

        # Consecutive same-direction candles (bullish/bearish run)
        directions = np.sign(np.diff(closes[-15:]))
        run = 0
        for d in reversed(directions):
            if d == directions[-1]:
                run += 1
            else:
                break
        momentum_direction = "Bullish" if directions[-1] > 0 else "Bearish"
        momentum_strength = min(run / 8.0, 1.0)  # 0-1 scale

        # ── Mean-Reversion Score ─────────────────────────────────────────
        rolling_mean = np.mean(closes[-48:])
        rolling_std = np.std(closes[-48:])
        z_score = (closes[-1] - rolling_mean) / rolling_std if rolling_std > 0 else 0.0
        # High |z_score| = stretched from mean = mean-reversion likely

        # ── Volatility Acceleration ─────────────────────────────────────
        returns_series = pd.Series(log_returns)
        ewma_recent = float(returns_series.ewm(span=6).std().values[-1])
        ewma_baseline = float(returns_series.ewm(span=24).std().values[-1])
        vol_ratio = ewma_recent / ewma_baseline if ewma_baseline > 0 else 1.0

        # ── Classify Regime ─────────────────────────────────────────────
        # Scoring heuristic (fully deterministic, no ML library needed)
        regimes = {}

        # Trending: high momentum, low mean-reversion tendency
        trending_score = momentum_strength * 0.6 + (1 - min(abs(z_score) / 2.5, 1.0)) * 0.4
        # Mean-reverting: price far from mean + low volatility acceleration
        ranging_score = min(abs(z_score) / 2.0, 1.0) * 0.5 + (1 - min(vol_ratio, 2.0) / 2.0) * 0.5
        # Breakout: volatility acceleration + consolidation (low recent momentum)
        breakout_score = min(vol_ratio - 1.0, 1.0) * 0.6 + (1 - momentum_strength) * 0.4
        # Fear/Panic: extreme negative momentum + high vol
        fear_score = max(-rolling_24 * 10, 0) * 0.5 + min(vol_ratio / 2.0, 1.0) * 0.5

        total = trending_score + ranging_score + breakout_score + fear_score
        if total > 0:
            regimes = {
                "Trending": round(trending_score / total * 100, 1),
                "Range-Bound": round(ranging_score / total * 100, 1),
                "Pre-Breakout": round(breakout_score / total * 100, 1),
                "High-Fear": round(fear_score / total * 100, 1),
            }
        else:
            regimes = {"Trending": 25, "Range-Bound": 25, "Pre-Breakout": 25, "High-Fear": 25}

        dominant_regime = max(regimes, key=regimes.get)

        # ── Price Range Context (last 24h) ─────────────────────────────
        high_24 = float(np.max(highs[-24:]))
        low_24 = float(np.min(lows[-24:]))
        range_pct_24h = (high_24 - low_24) / low_24 * 100

        # ── Support/Resistance levels (simple local extrema) ───────────
        from scipy.signal import argrelextrema
        close_series = closes[-100:]
        local_max_idx = argrelextrema(close_series, np.greater, order=5)[0]
        local_min_idx = argrelextrema(close_series, np.less, order=5)[0]
        resistances = sorted([float(close_series[i]) for i in local_max_idx[-3:]], reverse=True) if len(local_max_idx) else []
        supports = sorted([float(close_series[i]) for i in local_min_idx[-3:]], reverse=True) if len(local_min_idx) else []

        return {
            "dominant_regime": dominant_regime,
            "regime_probabilities": regimes,
            "momentum": {
                "direction": momentum_direction,
                "strength": round(momentum_strength * 100, 1),
                "rolling_10h_pct": round(rolling_10 * 100, 3),
                "rolling_24h_pct": round(rolling_24 * 100, 3),
                "consecutive_bars": int(run),
            },
            "mean_reversion": {
                "z_score": round(float(z_score), 3),
                "price_vs_48h_mean": round(float(closes[-1] - rolling_mean), 2),
                "48h_mean": round(float(rolling_mean), 2),
            },
            "volatility_accel": {
                "ratio": round(float(vol_ratio), 3),
                "interpretation": "Accelerating" if vol_ratio > 1.15 else "Stable" if vol_ratio > 0.85 else "Decelerating",
            },
            "range_24h": {
                "high": high_24,
                "low": low_24,
                "range_pct": round(range_pct_24h, 3),
            },
            "levels": {
                "resistances": resistances,
                "supports": supports,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# ── UNIQUE FEATURE 2: What-If Scenario Studio ─────────────────────────
# ═══════════════════════════════════════════════════════════════════════
class ScenarioParams(BaseModel):
    vol_multiplier: float = 1.0   # 0.1 – 5.0
    drift_bias_pct: float = 0.0   # -2% to +2%
    confidence_level: float = 0.95  # 0.80-0.99
    num_simulations: int = 10000

@app.post("/api/scenario")
def run_scenario(params: ScenarioParams):
    """
    Re-run GBM simulation with user-modified parameters.
    Returns the new prediction interval + histogram for live preview.
    """
    try:
        vol_multiplier = float(np.clip(params.vol_multiplier, 0.1, 10.0))
        drift_bias = float(np.clip(params.drift_bias_pct / 100.0, -0.05, 0.05))
        alpha = float(np.clip(1.0 - params.confidence_level, 0.01, 0.20))
        n_sims = int(np.clip(params.num_simulations, 100, 50000))

        df = fetch_binance_klines(limit=200)
        closes = df['close'].values
        log_returns = np.diff(np.log(closes))

        returns_series = pd.Series(log_returns)
        ewma_vol = float(returns_series.ewm(span=24, adjust=False).std().values[-1])
        if np.isnan(ewma_vol) or ewma_vol == 0:
            ewma_vol = float(np.std(log_returns))

        # Apply multiplier
        ewma_vol *= vol_multiplier

        # Fit Student-t
        recent = log_returns[-100:]
        std_dev = float(np.std(recent))
        if std_dev > 0:
            try:
                df_t, _, _ = student_t.fit(recent / std_dev)
                df_t = float(np.clip(df_t, 2.1, 10.0))
            except Exception:
                df_t = 3.0
        else:
            df_t = 3.0

        current_price = float(closes[-1])
        drift = float(np.mean(log_returns)) + drift_bias
        sim_returns = student_t.rvs(df_t, size=n_sims)
        sim_log_r = drift + (sim_returns * ewma_vol)
        sim_prices = current_price * np.exp(sim_log_r)

        lower = float(np.percentile(sim_prices, alpha / 2 * 100))
        upper = float(np.percentile(sim_prices, (1 - alpha / 2) * 100))

        hist_counts, bin_edges = np.histogram(sim_prices, bins=60)
        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()

        return {
            "current_price": current_price,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "applied_vol": ewma_vol,
            "applied_drift": drift,
            "histogram": {"counts": hist_counts.tolist(), "bin_centers": bin_centers},
            "params_used": {
                "vol_multiplier": vol_multiplier,
                "drift_bias_pct": params.drift_bias_pct,
                "confidence_level": params.confidence_level,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# ── UNIQUE FEATURE 3: Tail Risk Dashboard ─────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/tail-risk")
def get_tail_risk():
    """
    Compute tail risk metrics from simulated prices and historical returns:
    - VaR 95%, 99% (Value at Risk)
    - CVaR 95%, 99% (Conditional Value at Risk / Expected Shortfall)
    - Skewness of recent log-returns
    - Excess Kurtosis (fat-tailedness)
    - Max Drawdown over last 24h
    - Probability of >1%, >2%, >3% move in next hour
    """
    try:
        df = fetch_binance_klines(limit=500)
        closes = df['close'].values
        log_returns = np.diff(np.log(closes))
        current_price = float(closes[-1])

        # Simulate 50k paths for precision
        returns_series = pd.Series(log_returns)
        ewma_vol = float(returns_series.ewm(span=24, adjust=False).std().values[-1])
        if np.isnan(ewma_vol) or ewma_vol == 0:
            ewma_vol = float(np.std(log_returns))

        recent = log_returns[-100:]
        std_dev = float(np.std(recent))
        df_t = 3.0
        if std_dev > 0:
            try:
                df_t, _, _ = student_t.fit(recent / std_dev)
                df_t = float(np.clip(df_t, 2.1, 10.0))
            except Exception:
                pass

        drift = float(np.mean(log_returns))
        n_sims = 50000
        sim_returns = drift + student_t.rvs(df_t, size=n_sims) * ewma_vol
        sim_prices = current_price * np.exp(sim_returns)
        sim_pnl = sim_prices - current_price  # Dollar P&L

        # VaR and CVaR
        var_95 = float(np.percentile(sim_pnl, 5))
        var_99 = float(np.percentile(sim_pnl, 1))
        cvar_95 = float(np.mean(sim_pnl[sim_pnl <= var_95]))
        cvar_99 = float(np.mean(sim_pnl[sim_pnl <= var_99]))

        # Move probabilities
        prob_up_1pct = float(np.mean(sim_prices >= current_price * 1.01) * 100)
        prob_dn_1pct = float(np.mean(sim_prices <= current_price * 0.99) * 100)
        prob_up_2pct = float(np.mean(sim_prices >= current_price * 1.02) * 100)
        prob_dn_2pct = float(np.mean(sim_prices <= current_price * 0.98) * 100)

        # Skewness & Kurtosis of last 200 log returns
        recent_200 = log_returns[-200:]
        ret_skew = float(skew(recent_200))
        ret_kurt = float(kurtosis(recent_200))  # excess kurtosis (normal=0)

        # Max drawdown over last 24h (24 hourly bars)
        last_24 = closes[-25:]
        rolling_max = np.maximum.accumulate(last_24)
        drawdowns = (last_24 - rolling_max) / rolling_max * 100
        max_dd_24h = float(np.min(drawdowns))

        # Annualised volatility proxy
        annualised_vol = float(ewma_vol * np.sqrt(8760) * 100)

        # Best/Worst expected outcomes at different confidence levels
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        price_dist = {str(p): float(np.percentile(sim_prices, p)) for p in percentiles}

        return {
            "current_price": current_price,
            "var": {
                "var_95_dollar": var_95,
                "var_99_dollar": var_99,
                "var_95_pct": round(var_95 / current_price * 100, 4),
                "var_99_pct": round(var_99 / current_price * 100, 4),
            },
            "cvar": {
                "cvar_95_dollar": cvar_95,
                "cvar_99_dollar": cvar_99,
                "cvar_95_pct": round(cvar_95 / current_price * 100, 4),
                "cvar_99_pct": round(cvar_99 / current_price * 100, 4),
            },
            "move_probabilities": {
                "up_1pct": prob_up_1pct,
                "down_1pct": prob_dn_1pct,
                "up_2pct": prob_up_2pct,
                "down_2pct": prob_dn_2pct,
            },
            "distribution_shape": {
                "skewness": round(ret_skew, 4),
                "excess_kurtosis": round(ret_kurt, 4),
                "interpretation_skew": "Left-skewed (bearish lean)" if ret_skew < -0.1 else "Right-skewed (bullish lean)" if ret_skew > 0.1 else "Symmetric",
                "interpretation_kurt": "Fat-tailed (extreme moves likely)" if ret_kurt > 1 else "Thin-tailed" if ret_kurt < -0.5 else "Near-normal",
                "student_t_df": round(df_t, 2),
            },
            "max_drawdown_24h_pct": round(max_dd_24h, 4),
            "annualised_vol_pct": round(annualised_vol, 2),
            "price_percentile_distribution": price_dist,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Serve static frontend ───────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")
