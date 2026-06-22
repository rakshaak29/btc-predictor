from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
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

# ─── Root & Health ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def read_root():
    return HTMLResponse(content=INDEX_HTML)

@app.get("/health")
def health_check():
    return {"status": "ok"}

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

from fastapi.responses import HTMLResponse, Response

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Range Predictor — AlphaI × Polaris</title>
    <meta name="description" content="Live Bitcoin 1-hour range predictor using GBM with Student-t tails and volatility clustering. Backtest metrics, Monte Carlo histogram, and prediction history.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="stylesheet" href="style.css">
</head>
<body data-theme="dark">

    <!-- ── SIDEBAR ───────────────────────────── -->
    <aside id="sidebar">
        <div class="sidebar-header">
            <span class="sidebar-logo">₿</span>
            <h2>BTC Predictor</h2>
        </div>

        <nav class="sidebar-nav">
            <button class="nav-btn active" data-section="dashboard" id="nav-dashboard">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
                Dashboard
            </button>
            <button class="nav-btn" data-section="backtest" id="nav-backtest">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                Backtest
            </button>
            <button class="nav-btn" data-section="history" id="nav-history">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                History
            </button>
            <button class="nav-btn" data-section="montecarlo" id="nav-montecarlo">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                Monte Carlo
            </button>
            <button class="nav-btn" data-section="volatility" id="nav-volatility">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
                Volatility
            </button>
            <div class="nav-separator">✦ Unique Features</div>
            <button class="nav-btn" data-section="regime" id="nav-regime">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
                Regime AI
            </button>
            <button class="nav-btn" data-section="scenario" id="nav-scenario">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
                Scenario Studio
            </button>
            <button class="nav-btn" data-section="tailrisk" id="nav-tailrisk">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                Tail Risk
            </button>
        </nav>

        <div class="sidebar-controls">
            <h4>Controls</h4>
            <div class="control-group">
                <label>Simulations</label>
                <select id="sim-count">
                    <option value="1000">1,000</option>
                    <option value="5000">5,000</option>
                    <option value="10000" selected>10,000</option>
                </select>
            </div>
            <div class="control-group">
                <label>💱 Display Currency</label>
                <select id="currency-select">
                    <option value="USD" selected>🇺🇸 USD — US Dollar</option>
                    <option value="EUR">🇪🇺 EUR — Euro</option>
                    <option value="GBP">🇬🇧 GBP — British Pound</option>
                    <option value="INR">🇮🇳 INR — Indian Rupee</option>
                    <option value="JPY">🇯🇵 JPY — Japanese Yen</option>
                    <option value="CNY">🇨🇳 CNY — Chinese Yuan</option>
                    <option value="AUD">🇦🇺 AUD — Australian Dollar</option>
                    <option value="CAD">🇨🇦 CAD — Canadian Dollar</option>
                    <option value="CHF">🇨🇭 CHF — Swiss Franc</option>
                    <option value="SGD">🇸🇬 SGD — Singapore Dollar</option>
                    <option value="AED">🇦🇪 AED — UAE Dirham</option>
                    <option value="SAR">🇸🇦 SAR — Saudi Riyal</option>
                    <option value="BRL">🇧🇷 BRL — Brazilian Real</option>
                    <option value="MXN">🇲🇽 MXN — Mexican Peso</option>
                    <option value="KRW">🇰🇷 KRW — South Korean Won</option>
                    <option value="HKD">🇭🇰 HKD — Hong Kong Dollar</option>
                    <option value="SEK">🇸🇪 SEK — Swedish Krona</option>
                    <option value="NOK">🇳🇴 NOK — Norwegian Krone</option>
                    <option value="DKK">🇩🇰 DKK — Danish Krone</option>
                    <option value="NZD">🇳🇿 NZD — New Zealand Dollar</option>
                    <option value="ZAR">🇿🇦 ZAR — South African Rand</option>
                    <option value="TRY">🇹🇷 TRY — Turkish Lira</option>
                    <option value="IDR">🇮🇩 IDR — Indonesian Rupiah</option>
                    <option value="MYR">🇲🇾 MYR — Malaysian Ringgit</option>
                    <option value="THB">🇹🇭 THB — Thai Baht</option>
                    <option value="PHP">🇵🇭 PHP — Philippine Peso</option>
                    <option value="PLN">🇵🇱 PLN — Polish Złoty</option>
                    <option value="CZK">🇨🇿 CZK — Czech Koruna</option>
                    <option value="HUF">🇭🇺 HUF — Hungarian Forint</option>
                    <option value="ILS">🇮🇱 ILS — Israeli Shekel</option>
                </select>
                <div id="exchange-rate-badge" style="font-size:0.65rem;color:var(--muted);margin-top:0.3rem;text-align:right">Rate: 1 USD = 1.00 USD</div>
            </div>

            <div class="control-group">
                <label>Theme</label>
                <div class="theme-switch">
                    <span>Light</span>
                    <label class="toggle">
                        <input type="checkbox" id="theme-toggle" checked>
                        <span class="track"></span>
                    </label>
                    <span>Dark</span>
                </div>
            </div>
            <button class="refresh-btn" id="refresh-btn">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M23 4v6h-6"/><path d="M1 20v-6h6"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                Refresh Data
            </button>
        </div>

        <div class="last-update">
            Last updated: <span id="last-updated">—</span>
        </div>
        <div class="built-by">
            Built by Raksha AK
        </div>
    </aside>

    <!-- ── MAIN CONTENT ───────────────────────── -->
    <div class="main-wrapper">

        <!-- TOP BAR -->
        <header class="topbar">
            <div class="topbar-left">
                <span class="badge-challenge">AlphaI × Polaris Challenge</span>
                <h1 id="page-title">Dashboard</h1>
            </div>
            <div class="topbar-right">
                <div class="countdown-widget">
                    <span class="countdown-label">Prediction expires in</span>
                    <div class="countdown-ring">
                        <svg viewBox="0 0 42 42" width="52" height="52">
                            <circle cx="21" cy="21" r="18" fill="none" stroke="var(--border)" stroke-width="3"/>
                            <circle id="countdown-arc" cx="21" cy="21" r="18" fill="none" stroke="var(--accent)" stroke-width="3"
                                stroke-dasharray="113 113" stroke-dashoffset="0"
                                transform="rotate(-90 21 21)" stroke-linecap="round"/>
                        </svg>
                        <span id="countdown-text" class="countdown-time">--:--</span>
                    </div>
                </div>
                <div class="live-indicator">
                    <span class="pulse-dot"></span>
                    LIVE
                </div>
                <div class="current-price-mini">
                    <span>BTC/USDT</span>
                    <strong id="topbar-price">—</strong>
                </div>
            </div>
        </header>

        <!-- CONFIDENCE BANNER -->
        <div id="confidence-banner" class="confidence-banner hidden">
            <span id="confidence-icon">⚡</span>
            <div>
                <strong id="confidence-label">Model Confidence: —</strong>
                <p id="confidence-msg">Loading prediction data…</p>
            </div>
            <div class="banner-model-info">
                <span>GBM</span>
                <span>+</span>
                <span>Student-t</span>
                <span>+</span>
                <span>EWMA Vol</span>
            </div>
        </div>

        <main id="main-content">

            <!-- ════ SECTION: DASHBOARD ════ -->
            <section id="section-dashboard" class="section active">
                <!-- Hero Prediction Card -->
                <div class="prediction-hero glass-card">
                    <div class="hero-left">
                        <p class="label-sm">CURRENT BTC PRICE</p>
                        <div class="big-price" id="hero-price">Loading…</div>
                        <p class="label-sm muted" id="hero-timestamp"></p>
                    </div>
                    <div class="hero-divider"></div>
                    <div class="hero-right">
                        <p class="label-sm">95% PREDICTED RANGE — NEXT HOUR</p>
                        <div class="range-display">
                            <span class="range-low" id="hero-lower">—</span>
                            <div class="range-arrow">
                                <div class="range-bar">
                                    <div class="range-fill" id="range-fill"></div>
                                </div>
                                <span class="range-width-label" id="range-width">width: —</span>
                            </div>
                            <span class="range-high" id="hero-upper">—</span>
                        </div>
                    </div>
                </div>

                <!-- Metrics Row -->
                <div class="metrics-row">
                    <div class="metric-card glass-card" id="mc-coverage">
                        <div class="metric-icon">🎯</div>
                        <div class="metric-body">
                            <h3>Coverage</h3>
                            <div class="metric-val" id="metric-coverage">—</div>
                            <p>Target: 95%</p>
                        </div>
                        <div class="metric-badge" id="coverage-badge"></div>
                    </div>
                    <div class="metric-card glass-card" id="mc-width">
                        <div class="metric-icon">📏</div>
                        <div class="metric-body">
                            <h3>Avg Range Width</h3>
                            <div class="metric-val" id="metric-width">—</div>
                            <p>30-day backtest</p>
                        </div>
                    </div>
                    <div class="metric-card glass-card" id="mc-winkler">
                        <div class="metric-icon">📊</div>
                        <div class="metric-body">
                            <h3>Winkler Score</h3>
                            <div class="metric-val" id="metric-winkler">—</div>
                            <p>Lower is better</p>
                        </div>
                    </div>
                    <div class="metric-card glass-card" id="mc-preds">
                        <div class="metric-icon">🔢</div>
                        <div class="metric-body">
                            <h3>Total Predictions</h3>
                            <div class="metric-val" id="metric-preds">—</div>
                            <p>Backtest sample</p>
                        </div>
                    </div>
                </div>

                <!-- Main Price Chart (last 50 bars + ribbon) -->
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>Price Chart — Last 50 Hours</h2>
                        <span class="card-tag">Candlestick + Prediction Ribbon</span>
                    </div>
                    <div class="chart-wrap" style="height:380px">
                        <canvas id="priceChart"></canvas>
                    </div>
                </div>
            </section>

            <!-- ════ SECTION: BACKTEST ════ -->
            <section id="section-backtest" class="section">
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>30-Day Backtest Results</h2>
                        <span class="card-tag">720 hourly predictions</span>
                    </div>
                    <!-- Coverage gauge -->
                    <div class="gauge-row">
                        <div class="gauge-box">
                            <svg class="gauge-svg" viewBox="0 0 200 120">
                                <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="var(--border)" stroke-width="16"/>
                                <path id="gauge-arc" d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="var(--accent)" stroke-width="16" stroke-dasharray="0 251" stroke-linecap="round"/>
                                <text x="100" y="95" text-anchor="middle" font-size="26" font-weight="800" fill="var(--text)" id="gauge-text">—</text>
                                <text x="100" y="115" text-anchor="middle" font-size="12" fill="var(--muted)">Coverage</text>
                            </svg>
                        </div>
                        <div class="backtest-stats">
                            <div class="bstat">
                                <span class="bstat-label">Coverage (95% target)</span>
                                <span class="bstat-val" id="bt-coverage">—</span>
                            </div>
                            <div class="bstat">
                                <span class="bstat-label">Average Width</span>
                                <span class="bstat-val" id="bt-width">—</span>
                            </div>
                            <div class="bstat">
                                <span class="bstat-label">Mean Winkler Score</span>
                                <span class="bstat-val" id="bt-winkler">—</span>
                            </div>
                            <div class="bstat">
                                <span class="bstat-label">Predictions Made</span>
                                <span class="bstat-val" id="bt-preds">—</span>
                            </div>
                        </div>
                    </div>
                    <!-- Winkler chart -->
                    <div class="chart-wrap" style="height:320px; margin-top:2rem">
                        <canvas id="winklerChart"></canvas>
                    </div>
                </div>

                <!-- Backtest table -->
                <div class="card glass-card" style="margin-top:1.5rem">
                    <div class="card-header">
                        <h2>Prediction-by-Prediction Results</h2>
                        <span class="card-tag last-n">Showing last 50</span>
                    </div>
                    <div class="table-wrap">
                        <table id="backtest-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Timestamp</th>
                                    <th>Actual Price</th>
                                    <th>Lower Bound</th>
                                    <th>Upper Bound</th>
                                    <th>Width</th>
                                    <th>Winkler</th>
                                    <th>Result</th>
                                </tr>
                            </thead>
                            <tbody id="backtest-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </section>

            <!-- ════ SECTION: HISTORY ════ -->
            <section id="section-history" class="section">
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>Live Prediction History</h2>
                        <div class="header-actions">
                            <div class="accuracy-pill" id="live-accuracy">Live Accuracy: —</div>
                            <a href="/api/prediction/history/csv" class="btn-download" id="csv-download">
                                ⬇ Download CSV
                            </a>
                        </div>
                    </div>
                    <div class="table-wrap">
                        <table id="history-table">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Timestamp</th>
                                    <th>Price</th>
                                    <th>Lower</th>
                                    <th>Upper</th>
                                    <th>Width</th>
                                    <th>Actual</th>
                                    <th>Result</th>
                                </tr>
                            </thead>
                            <tbody id="history-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </section>

            <!-- ════ SECTION: MONTE CARLO ════ -->
            <section id="section-montecarlo" class="section">
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>Monte Carlo Simulation</h2>
                        <span class="card-tag">10,000 Simulated Price Paths</span>
                    </div>
                    <div class="mc-stats-row">
                        <div class="mc-stat">
                            <span class="mcs-label">2.5th Percentile</span>
                            <span class="mcs-val red" id="mc-p025">—</span>
                        </div>
                        <div class="mc-stat">
                            <span class="mcs-label">Median (50th)</span>
                            <span class="mcs-val" id="mc-p50">—</span>
                        </div>
                        <div class="mc-stat">
                            <span class="mcs-label">97.5th Percentile</span>
                            <span class="mcs-val green" id="mc-p975">—</span>
                        </div>
                        <div class="mc-stat">
                            <span class="mcs-label">Current Price</span>
                            <span class="mcs-val accent" id="mc-current">—</span>
                        </div>
                    </div>
                    <div class="chart-wrap" style="height:400px">
                        <canvas id="histogramChart"></canvas>
                    </div>
                    <p class="chart-note">Histogram of 10,000 simulated next-hour BTC prices using GBM + Student-t. Dashed lines mark the 2.5th, 50th, and 97.5th percentiles. The shaded green band is the 95% confidence interval.</p>
                </div>
            </section>

            <!-- ════ SECTION: VOLATILITY ════ -->
            <section id="section-volatility" class="section">
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>Volatility Regime</h2>
                        <span class="card-tag">Last 48 Hours of EWMA Vol</span>
                    </div>
                    <div class="vol-summary" id="vol-summary">
                        <div class="vol-stat">
                            <span class="vs-label">Current Volatility</span>
                            <span class="vs-val" id="vol-current">—</span>
                        </div>
                        <div class="vol-stat">
                            <span class="vs-label">Threshold (70th pct)</span>
                            <span class="vs-val" id="vol-threshold">—</span>
                        </div>
                        <div class="vol-stat">
                            <span class="vs-label">Regime</span>
                            <span class="vs-val" id="vol-regime">—</span>
                        </div>
                        <div class="vol-stat">
                            <span class="vs-label">EWMA Span</span>
                            <span class="vs-val">24h window</span>
                        </div>
                    </div>
                    <div class="chart-wrap" style="height:350px">
                        <canvas id="volChart"></canvas>
                    </div>
                    <p class="chart-note">The red dashed line is the volatility threshold (70th percentile of recent 10-bar rolling stddev). Bars above the threshold indicate a volatile regime — the model widens its prediction intervals accordingly.</p>
                </div>
            </section>

            <!-- ════ SECTION: REGIME AI ════ -->
            <section id="section-regime" class="section">
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>Market Regime Classifier</h2>
                        <span class="card-tag unique-badge">✦ Unique Feature</span>
                    </div>
                    <div class="regime-grid">
                        <div class="regime-dominant glass-card">
                            <p class="label-sm">DOMINANT REGIME</p>
                            <div class="regime-name" id="regime-dominant">—</div>
                            <p class="regime-desc" id="regime-desc">Analysing market structure…</p>
                        </div>
                        <div class="regime-bars">
                            <p class="label-sm">REGIME PROBABILITIES</p>
                            <div id="regime-prob-bars"></div>
                        </div>
                    </div>
                    <div class="regime-signals">
                        <div class="signal-card">
                            <h4>📈 Momentum</h4>
                            <div class="sig-row"><span>Direction</span><span id="sig-momentum-dir">—</span></div>
                            <div class="sig-row"><span>Strength</span><span id="sig-momentum-str">—</span></div>
                            <div class="sig-row"><span>10h Return</span><span id="sig-momentum-10">—</span></div>
                            <div class="sig-row"><span>Consecutive Bars</span><span id="sig-momentum-run">—</span></div>
                        </div>
                        <div class="signal-card">
                            <h4>↔ Mean Reversion</h4>
                            <div class="sig-row"><span>Z-Score</span><span id="sig-zscor">—</span></div>
                            <div class="sig-row"><span>48h Mean</span><span id="sig-mean">—</span></div>
                            <div class="sig-row"><span>Price vs Mean</span><span id="sig-vs-mean">—</span></div>
                        </div>
                        <div class="signal-card">
                            <h4>⚡ Volatility Accel.</h4>
                            <div class="sig-row"><span>Vol Ratio</span><span id="sig-vol-ratio">—</span></div>
                            <div class="sig-row"><span>Status</span><span id="sig-vol-status">—</span></div>
                            <div class="sig-row"><span>24h Range %</span><span id="sig-range-pct">—</span></div>
                        </div>
                        <div class="signal-card">
                            <h4>🎯 Support / Resistance</h4>
                            <div class="sig-row"><span>R1</span><span id="sig-r1" class="hit-yes">—</span></div>
                            <div class="sig-row"><span>R2</span><span id="sig-r2" class="hit-yes">—</span></div>
                            <div class="sig-row"><span>S1</span><span id="sig-s1" class="hit-no">—</span></div>
                            <div class="sig-row"><span>S2</span><span id="sig-s2" class="hit-no">—</span></div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- ════ SECTION: SCENARIO STUDIO ════ -->
            <section id="section-scenario" class="section">
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>What-If Scenario Studio</h2>
                        <span class="card-tag unique-badge">✦ Unique Feature — Live Interactive</span>
                    </div>
                    <p class="chart-note" style="margin-bottom:1.5rem">Drag the sliders to re-run the GBM simulation instantly with custom parameters. See how your prediction range changes in real time.</p>
                    <div class="scenario-layout">
                        <div class="scenario-controls">
                            <div class="slider-group">
                                <label>Volatility Multiplier: <strong id="sl-vol-val">1.0×</strong></label>
                                <input type="range" id="sl-vol" min="0.1" max="5" step="0.1" value="1.0" class="styled-slider">
                                <div class="slider-hints"><span>0.1× (calm)</span><span>5× (extreme)</span></div>
                            </div>
                            <div class="slider-group">
                                <label>Drift Bias: <strong id="sl-drift-val">0.00%</strong></label>
                                <input type="range" id="sl-drift" min="-2" max="2" step="0.05" value="0" class="styled-slider">
                                <div class="slider-hints"><span>−2% (bearish)</span><span>+2% (bullish)</span></div>
                            </div>
                            <div class="slider-group">
                                <label>Confidence Level: <strong id="sl-conf-val">95%</strong></label>
                                <input type="range" id="sl-conf" min="80" max="99" step="1" value="95" class="styled-slider">
                                <div class="slider-hints"><span>80%</span><span>99%</span></div>
                            </div>
                            <div class="scenario-result glass-card">
                                <p class="label-sm">SCENARIO PREDICTION</p>
                                <div class="scenario-range">
                                    <span class="range-low" id="sc-lower">—</span>
                                    <span class="sc-sep">to</span>
                                    <span class="range-high" id="sc-upper">—</span>
                                </div>
                                <div class="scenario-width" id="sc-width">Width: —</div>
                                <div class="scenario-vs">
                                    <span>vs. Model: </span>
                                    <span id="sc-vs">—</span>
                                </div>
                            </div>
                        </div>
                        <div class="scenario-chart-wrap">
                            <div class="chart-wrap" style="height:380px">
                                <canvas id="scenarioChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- ════ SECTION: TAIL RISK ════ -->
            <section id="section-tailrisk" class="section">
                <div class="card glass-card">
                    <div class="card-header">
                        <h2>Tail Risk Dashboard</h2>
                        <span class="card-tag unique-badge">✦ Unique Feature — VaR / CVaR / Skewness</span>
                    </div>
                    <div class="tail-grid">
                        <div class="tail-block">
                            <h3>📉 Value at Risk (VaR)</h3>
                            <div class="tail-stat-row"><span>VaR 95%</span><span class="hit-no" id="tr-var95">—</span></div>
                            <div class="tail-stat-row"><span>VaR 99%</span><span class="hit-no" id="tr-var99">—</span></div>
                            <p class="chart-note">Max expected loss in 95% / 99% of scenarios over 1 hour.</p>
                        </div>
                        <div class="tail-block">
                            <h3>🔥 Expected Shortfall (CVaR)</h3>
                            <div class="tail-stat-row"><span>CVaR 95%</span><span class="hit-no" id="tr-cvar95">—</span></div>
                            <div class="tail-stat-row"><span>CVaR 99%</span><span class="hit-no" id="tr-cvar99">—</span></div>
                            <p class="chart-note">Average loss in the worst 5% / 1% of outcomes.</p>
                        </div>
                        <div class="tail-block">
                            <h3>🎲 Move Probabilities</h3>
                            <div class="tail-stat-row"><span>P(+1%)</span><span class="hit-yes" id="tr-up1">—</span></div>
                            <div class="tail-stat-row"><span>P(−1%)</span><span class="hit-no" id="tr-dn1">—</span></div>
                            <div class="tail-stat-row"><span>P(+2%)</span><span class="hit-yes" id="tr-up2">—</span></div>
                            <div class="tail-stat-row"><span>P(−2%)</span><span class="hit-no" id="tr-dn2">—</span></div>
                        </div>
                        <div class="tail-block">
                            <h3>📐 Distribution Shape</h3>
                            <div class="tail-stat-row"><span>Skewness</span><span id="tr-skew">—</span></div>
                            <div class="tail-stat-row"><span>Excess Kurtosis</span><span id="tr-kurt">—</span></div>
                            <div class="tail-stat-row"><span>Student-t df</span><span id="tr-df">—</span></div>
                            <div class="tail-stat-row"><span>Ann. Volatility</span><span id="tr-annvol">—</span></div>
                        </div>
                    </div>
                    <div class="tail-interp">
                        <p id="tr-skew-interp" class="interp-tag"></p>
                        <p id="tr-kurt-interp" class="interp-tag"></p>
                        <p id="tr-maxdd" class="interp-tag"></p>
                    </div>
                    <div class="card-header" style="margin-top:2rem; margin-bottom:1rem">
                        <h2>Price Percentile Distribution</h2>
                        <span class="card-tag">Next-Hour Outcomes</span>
                    </div>
                    <div class="chart-wrap" style="height:320px">
                        <canvas id="tailChart"></canvas>
                    </div>
                </div>
            </section>

        </main>
    </div>

    <!-- Spinner overlay -->
    <div id="spinner" class="spinner-overlay">
        <div class="spinner-box">
            <div class="spinner-ring"></div>
            <p>Fetching live BTC data…</p>
        </div>
    </div>

    <script src="app.js"></script>
</body>
</html>

"""

STYLE_CSS = """
/* ══════════════════════════════════════════════════════════════════
   BTC RANGE PREDICTOR — Full Design System
   Dark theme: charcoal + emerald (STRICTLY NO BLUE)
   Light theme: warm whites + orange accent
══════════════════════════════════════════════════════════════════ */

/* ── Tokens ──────────────────────────────────────────────────── */
:root {
    --sidebar-w: 240px;
    --radius: 14px;
    --radius-sm: 8px;
    --transition: 0.35s ease;
}

[data-theme="dark"] {
    --bg:         #0f0f0f;
    --surface:    #1a1a1a;
    --surface2:   #222222;
    --border:     rgba(255,255,255,0.07);
    --text:       #f0f0f0;
    --muted:      #7a7a7a;
    --accent:     #10b981;        /* Emerald Green */
    --accent2:    #f97316;        /* Sunset Orange */
    --red:        #ef4444;
    --green:      #22c55e;
    --yellow:     #eab308;
    --glass-bg:   rgba(26,26,26,0.75);
    --glass-border: rgba(255,255,255,0.06);
    --shadow:     0 8px 40px rgba(0,0,0,0.45);
    --chart-grid: rgba(255,255,255,0.06);
    --chart-text: #a0a0a0;
    --sidebar-bg: #141414;
}

[data-theme="light"] {
    --bg:         #f5f4f2;
    --surface:    #ffffff;
    --surface2:   #f0ede8;
    --border:     rgba(0,0,0,0.08);
    --text:       #1a1a1a;
    --muted:      #888888;
    --accent:     #059669;
    --accent2:    #ea580c;
    --red:        #dc2626;
    --green:      #16a34a;
    --yellow:     #ca8a04;
    --glass-bg:   rgba(255,255,255,0.85);
    --glass-border: rgba(0,0,0,0.07);
    --shadow:     0 4px 24px rgba(0,0,0,0.10);
    --chart-grid: rgba(0,0,0,0.07);
    --chart-text: #555555;
    --sidebar-bg: #ffffff;
}

/* ── Reset ───────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { height: 100%; }
body {
    font-family: 'Outfit', sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100%;
    display: flex;
    transition: background var(--transition), color var(--transition);
    overflow-x: hidden;
}

/* ── Layout ─────────────────────────────────────────────────── */
#sidebar {
    position: fixed; top: 0; left: 0; bottom: 0;
    width: var(--sidebar-w);
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 1.5rem 1rem;
    z-index: 100;
    overflow-y: auto;
    transition: background var(--transition), border-color var(--transition);
}

.main-wrapper {
    margin-left: var(--sidebar-w);
    flex: 1;
    display: flex; flex-direction: column;
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
.sidebar-header {
    display: flex; align-items: center; gap: 0.75rem;
    margin-bottom: 2rem;
}
.sidebar-logo {
    font-size: 1.8rem; font-weight: 900; color: var(--accent);
    width: 40px; height: 40px; background: rgba(16,185,129,0.12);
    border-radius: 10px; display: grid; place-items: center;
}
.sidebar-header h2 { font-size: 1.05rem; font-weight: 700; }

.sidebar-nav { display: flex; flex-direction: column; gap: 0.25rem; margin-bottom: 2rem; }

.nav-btn {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.65rem 0.85rem;
    background: transparent; border: none;
    border-radius: var(--radius-sm);
    color: var(--muted); font-family: 'Outfit', sans-serif;
    font-size: 0.9rem; font-weight: 500;
    cursor: pointer; text-align: left;
    transition: all 0.2s;
}
.nav-btn svg { width: 17px; height: 17px; flex-shrink: 0; }
.nav-btn:hover { background: var(--surface2); color: var(--text); }
.nav-btn.active {
    background: rgba(16,185,129,0.12);
    color: var(--accent); font-weight: 600;
}

.sidebar-controls { margin-top: auto; padding-top: 1rem; border-top: 1px solid var(--border); }
.sidebar-controls h4 { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 1rem; }

.control-group { margin-bottom: 1rem; }
.control-group label { display: block; font-size: 0.78rem; color: var(--muted); margin-bottom: 0.4rem; }
.control-group select {
    width: 100%; padding: 0.4rem 0.6rem;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: var(--radius-sm); color: var(--text);
    font-family: 'Outfit', sans-serif; font-size: 0.85rem;
    cursor: pointer;
}

.theme-switch { display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; }
.toggle { position: relative; width: 44px; height: 24px; cursor: pointer; }
.toggle input { opacity: 0; width: 0; height: 0; }
.track {
    position: absolute; inset: 0;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; transition: background 0.3s;
}
.track::after {
    content: '';
    position: absolute; top: 3px; left: 3px;
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--muted); transition: all 0.3s;
}
.toggle input:checked + .track { background: var(--accent); }
.toggle input:checked + .track::after { left: 23px; background: #fff; }

.refresh-btn {
    display: flex; align-items: center; justify-content: center; gap: 0.5rem;
    width: 100%; padding: 0.6rem; margin-top: 0.75rem;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
    border-radius: var(--radius-sm); color: var(--accent);
    font-family: 'Outfit', sans-serif; font-size: 0.85rem; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
}
.refresh-btn:hover { background: rgba(16,185,129,0.2); }
.refresh-btn.spinning svg { animation: spin 1s linear infinite; }

.last-update {
    font-size: 0.7rem; color: var(--muted);
    margin-top: 0.75rem; text-align: center;
}

.built-by {
    font-size: 0.75rem; color: var(--accent2);
    margin-top: 0.5rem; text-align: center;
    font-weight: 700; letter-spacing: 0.5px;
}

/* ── Topbar ──────────────────────────────────────────────────── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 2rem;
    background: var(--sidebar-bg);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 50;
}
.topbar-left { display: flex; flex-direction: column; }
.badge-challenge {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: var(--accent2);
}
#page-title { font-size: 1.4rem; font-weight: 800; margin-top: 2px; }
.topbar-right { display: flex; align-items: center; gap: 1.5rem; }
.live-indicator {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px;
    color: var(--green);
}
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 0 rgba(34,197,94,0.5);
    animation: pulse 1.8s infinite;
}
.current-price-mini { text-align: right; }
.current-price-mini span { display: block; font-size: 0.7rem; color: var(--muted); }
.current-price-mini strong { font-size: 1.1rem; font-weight: 800; }

/* ── Confidence Banner ───────────────────────────────────────── */
.confidence-banner {
    display: flex; align-items: center; gap: 1rem;
    padding: 0.85rem 2rem;
    font-size: 0.88rem;
    border-bottom: 1px solid var(--border);
    transition: background 0.3s;
}
.confidence-banner.hidden { display: none; }
.confidence-banner.green { background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.2); }
.confidence-banner.yellow { background: rgba(234,179,8,0.08); border-color: rgba(234,179,8,0.2); }
.confidence-banner.red { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.2); }
#confidence-icon { font-size: 1.5rem; }
.confidence-banner strong { font-size: 0.92rem; }
.confidence-banner p { color: var(--muted); font-size: 0.8rem; }
.banner-model-info {
    margin-left: auto; display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.72rem; color: var(--muted);
}
.banner-model-info span:not(:last-child):after { content: '·'; margin-left: 0.5rem; color: var(--border); }

/* ── Main ────────────────────────────────────────────────────── */
#main-content { padding: 2rem; flex: 1; }

/* ── Sections ────────────────────────────────────────────────── */
.section { display: none; }
.section.active { display: block; }

/* ── Glass Cards ─────────────────────────────────────────────── */
.glass-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    transition: background var(--transition), border-color var(--transition);
}

.card { padding: 1.5rem 2rem; }
.card + .card { margin-top: 1.5rem; }

.card-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.5rem;
}
.card-header h2 { font-size: 1.15rem; font-weight: 700; }
.card-tag {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1px;
    color: var(--accent); background: rgba(16,185,129,0.1);
    padding: 0.2rem 0.65rem; border-radius: 20px;
}

/* ── Hero Prediction Card ─────────────────────────────────────── */
.prediction-hero {
    display: flex; align-items: center;
    padding: 2rem 2.5rem; gap: 3rem;
    margin-bottom: 1.5rem;
}
.hero-left { flex: 1; }
.hero-divider { width: 1px; height: 80px; background: var(--border); }
.hero-right { flex: 2; }
.label-sm {
    font-size: 0.68rem; letter-spacing: 2px; font-weight: 700;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 0.4rem; display: block;
}
.big-price {
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(135deg, var(--text) 40%, var(--muted) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}

.range-display { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
.range-low  { font-size: clamp(1rem, 2.2vw, 1.8rem); font-weight: 800; color: var(--red);   white-space: nowrap; }
.range-high { font-size: clamp(1rem, 2.2vw, 1.8rem); font-weight: 800; color: var(--green); white-space: nowrap; }
.range-arrow { flex: 1; display: flex; flex-direction: column; gap: 0.3rem; }
.range-bar {
    height: 8px; background: var(--surface2);
    border-radius: 4px; overflow: hidden;
}
.range-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, var(--red), var(--accent), var(--green));
    width: 100%; transform-origin: left;
    animation: fillBar 1s ease forwards;
}
.range-width-label { font-size: 0.75rem; color: var(--muted); text-align: center; }

/* ── Metrics Row ─────────────────────────────────────────────── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.25rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    padding: 1.25rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    position: relative; overflow: hidden;
    transition: transform 0.25s;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-icon { font-size: 1.8rem; }
.metric-body h3 { font-size: 0.72rem; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); }
.metric-val {
    font-size: clamp(1rem, 2vw, 1.75rem); font-weight: 900; color: var(--accent);
    line-height: 1; margin: 0.25rem 0;
    word-break: break-all; overflow-wrap: break-word;
}
.metric-body p { font-size: 0.72rem; color: var(--muted); }
.metric-badge {
    position: absolute; top: 0.6rem; right: 0.8rem;
    font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.5rem; border-radius: 20px;
}
.metric-badge.good { background: rgba(34,197,94,0.15); color: var(--green); }
.metric-badge.warn { background: rgba(234,179,8,0.15); color: var(--yellow); }
.metric-badge.bad  { background: rgba(239,68,68,0.15);  color: var(--red);   }

/* ── Chart Wrap ──────────────────────────────────────────────── */
.chart-wrap { position: relative; width: 100%; }
.chart-note { margin-top: 1rem; font-size: 0.78rem; color: var(--muted); line-height: 1.6; }

/* ── Gauge ───────────────────────────────────────────────────── */
.gauge-row {
    display: flex; align-items: center; gap: 3rem;
    padding: 1rem 0 0;
}
.gauge-box { width: 200px; flex-shrink: 0; }
.gauge-svg { width: 100%; overflow: visible; }
.backtest-stats { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.bstat { display: flex; flex-direction: column; gap: 0.2rem; }
.bstat-label { font-size: 0.72rem; color: var(--muted); }
.bstat-val { font-size: clamp(0.9rem, 1.5vw, 1.2rem); font-weight: 800; color: var(--text); word-break: break-all; }

/* ── Tables ──────────────────────────────────────────────────── */
.table-wrap { overflow-x: auto; border-radius: var(--radius-sm); }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
thead tr { background: var(--surface2); }
th { padding: 0.65rem 0.85rem; text-align: left; font-size: 0.7rem; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); white-space: nowrap; }
td { padding: 0.55rem 0.85rem; border-bottom: 1px solid var(--border); white-space: nowrap; }
tbody tr:hover { background: var(--surface2); }
.hit-yes { color: var(--green); font-weight: 700; }
.hit-no  { color: var(--red);   font-weight: 700; }
.hit-pending { color: var(--muted); }

/* ── Header Actions ──────────────────────────────────────────── */
.header-actions { display: flex; align-items: center; gap: 1rem; }
.accuracy-pill {
    font-size: 0.78rem; font-weight: 700;
    padding: 0.3rem 0.9rem; border-radius: 20px;
    background: rgba(16,185,129,0.12); color: var(--accent);
}
.btn-download {
    font-size: 0.8rem; font-weight: 600; text-decoration: none;
    padding: 0.4rem 1rem; border-radius: var(--radius-sm);
    background: var(--accent2); color: #fff;
    transition: opacity 0.2s;
}
.btn-download:hover { opacity: 0.85; }

/* ── Monte Carlo stats ───────────────────────────────────────── */
.mc-stats-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
    margin-bottom: 1.5rem;
}
.mc-stat {
    background: var(--surface2); border-radius: var(--radius-sm);
    padding: 0.85rem 1rem; text-align: center;
}
.mcs-label { display: block; font-size: 0.68rem; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.mcs-val { font-size: clamp(0.85rem, 1.5vw, 1.2rem); font-weight: 800; word-break: break-all; }
.mcs-val.red { color: var(--red); }
.mcs-val.green { color: var(--green); }
.mcs-val.accent { color: var(--accent2); }

/* ── Volatility Summary ──────────────────────────────────────── */
.vol-summary {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
    margin-bottom: 1.5rem;
}
.vol-stat {
    background: var(--surface2); border-radius: var(--radius-sm);
    padding: 0.85rem 1rem; text-align: center;
}
.vs-label { display: block; font-size: 0.68rem; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
.vs-val { font-size: 1.15rem; font-weight: 800; }

/* ── Spinner ─────────────────────────────────────────────────── */
.spinner-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    display: grid; place-items: center; z-index: 999;
    backdrop-filter: blur(4px);
    opacity: 0; pointer-events: none; transition: opacity 0.3s;
}
.spinner-overlay.active { opacity: 1; pointer-events: all; }
.spinner-box { text-align: center; }
.spinner-ring {
    width: 52px; height: 52px; border-radius: 50%;
    border: 4px solid var(--border);
    border-top-color: var(--accent);
    animation: spin 0.9s linear infinite;
    margin: 0 auto 1rem;
}
.spinner-box p { font-size: 0.9rem; color: var(--muted); }

/* ── Animations ──────────────────────────────────────────────── */
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}
@keyframes fillBar { from { transform: scaleX(0); } to { transform: scaleX(1); } }

.hidden { display: none !important; }

/* ── Unique Feature Additions ─────────────────────────────────── */
.nav-separator {
    font-size: 0.62rem; letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--accent2); padding: 1rem 0.85rem 0.4rem;
    font-weight: 700;
}
.unique-badge { background: rgba(249,115,22,0.12); color: var(--accent2); }

/* Countdown widget */
.countdown-widget { display: flex; align-items: center; gap: 0.75rem; }
.countdown-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); max-width: 60px; line-height: 1.3; }
.countdown-ring { position: relative; width: 52px; height: 52px; }
.countdown-ring svg { position: absolute; top: 0; left: 0; }
.countdown-time { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 800; }

/* Regime Classifier */
.regime-grid { display: grid; grid-template-columns: 220px 1fr; gap: 1.5rem; margin-bottom: 2rem; align-items: start; }
.regime-dominant { padding: 1.5rem; text-align: center; }
.regime-name { font-size: 2rem; font-weight: 900; color: var(--accent); margin: 0.5rem 0; }
.regime-desc { font-size: 0.78rem; color: var(--muted); }
.regime-signals { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem; }
.signal-card { background: var(--surface2); border-radius: var(--radius-sm); padding: 1rem; }
.signal-card h4 { font-size: 0.82rem; font-weight: 700; margin-bottom: 0.75rem; }
.sig-row { display: flex; justify-content: space-between; font-size: 0.78rem; padding: 0.3rem 0; border-bottom: 1px solid var(--border); }
.sig-row:last-child { border-bottom: none; }
.regime-prob-bar-wrap { margin-bottom: 0.6rem; }
.regime-prob-bar-wrap .rpb-label { display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 0.2rem; }
.regime-prob-bar-track { height: 10px; background: var(--surface2); border-radius: 5px; overflow: hidden; }
.regime-prob-bar-fill { height: 100%; border-radius: 5px; transition: width 0.8s ease; }

/* Scenario Studio */
.scenario-layout { display: grid; grid-template-columns: 320px 1fr; gap: 2rem; align-items: start; }
.scenario-controls { display: flex; flex-direction: column; gap: 1.5rem; }
.slider-group label { display: block; font-size: 0.82rem; margin-bottom: 0.6rem; color: var(--muted); }
.slider-group label strong { color: var(--text); }
.styled-slider { width: 100%; -webkit-appearance: none; height: 6px; border-radius: 3px; background: var(--surface2); outline: none; }
.styled-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%; background: var(--accent); cursor: pointer; box-shadow: 0 0 0 3px rgba(16,185,129,0.2); }
.slider-hints { display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--muted); margin-top: 0.3rem; }
.scenario-result {
    padding: 1.25rem; margin-top: 0.5rem;
    overflow: hidden;
}
.scenario-range {
    display: flex; align-items: baseline; flex-wrap: wrap;
    gap: 0.5rem; margin: 0.5rem 0;
    font-size: clamp(0.9rem, 2vw, 1.3rem); font-weight: 800;
    word-break: break-all; overflow-wrap: break-word;
}
.sc-sep { color: var(--muted); font-weight: 400; flex-shrink: 0; }
.scenario-width { font-size: 0.82rem; color: var(--muted); }
.scenario-vs { font-size: 0.82rem; margin-top: 0.4rem; color: var(--muted); }
.scenario-vs span:last-child { font-weight: 700; }

/* Tail Risk */
.tail-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 1.5rem; }
.tail-block { background: var(--surface2); border-radius: var(--radius-sm); padding: 1.25rem; }
.tail-block h3 { font-size: 0.82rem; font-weight: 700; margin-bottom: 1rem; }
.tail-stat-row { display: flex; justify-content: space-between; font-size: 0.82rem; padding: 0.35rem 0; border-bottom: 1px solid var(--border); }
.tail-stat-row:last-of-type { border: none; }
.tail-interp { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.interp-tag { font-size: 0.78rem; padding: 0.3rem 0.85rem; border-radius: 20px; background: var(--surface2); color: var(--muted); }

"""

APP_JS = """
/* ══════════════════════════════════════════════════════════
   BTC Range Predictor — Frontend Logic
   Sections: Dashboard, Backtest, History, Monte Carlo, Volatility
══════════════════════════════════════════════════════════ */

let priceChart = null;
let histogramChart = null;
let volChart = null;
let winklerChart = null;
let cachedPrediction = null;

// ── Utility ──────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const fmt = (n) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(n);
const fmtShort = (n) => `$${(+n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct = (n) => `${(n * 100).toFixed(2)}%`;
const getTheme = () => document.documentElement.getAttribute('data-theme') || 'dark';

function chartColors() {
    const dark = getTheme() === 'dark';
    return {
        text: dark ? '#a0a0a0' : '#555555',
        grid: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.07)',
        accent: dark ? '#10b981' : '#059669',
        accent2: dark ? '#f97316' : '#ea580c',
        red: dark ? '#ef4444' : '#dc2626',
        green: dark ? '#22c55e' : '#16a34a',
        yellow: dark ? '#eab308' : '#ca8a04',
    };
}

// ── Spinner ───────────────────────────────────────────────────────
function showSpinner(on) {
    $('spinner').classList.toggle('active', on);
}

// ── Navigation ────────────────────────────────────────────────────
function activateSection(name) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    $(`section-${name}`).classList.add('active');
    $(`nav-${name}`).classList.add('active');
    const titles = {
        dashboard: 'Dashboard',
        backtest: '30-Day Backtest',
        history: 'Prediction History',
        montecarlo: 'Monte Carlo Simulation',
        volatility: 'Volatility Regime',
    };
    $('page-title').textContent = titles[name] || name;

    // lazy render charts that need data
    if (name === 'backtest') loadBacktest();
    if (name === 'history') loadHistory();
    if (name === 'montecarlo' && cachedPrediction) renderHistogram(cachedPrediction);
    if (name === 'volatility' && cachedPrediction) renderVolChart(cachedPrediction);
}

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => activateSection(btn.dataset.section));
});

// ── Theme Toggle ─────────────────────────────────────────────────
$('theme-toggle').addEventListener('change', (e) => {
    const theme = e.target.checked ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);
    setTimeout(() => {
        if (cachedPrediction) {
            renderPriceChart(cachedPrediction);
            renderHistogram(cachedPrediction);
            renderVolChart(cachedPrediction);
        }
    }, 50);
});

// ── Refresh Button ────────────────────────────────────────────────
$('refresh-btn').addEventListener('click', () => {
    $('refresh-btn').classList.add('spinning');
    loadCurrentPrediction().finally(() => {
        $('refresh-btn').classList.remove('spinning');
    });
});

// ── LOAD CURRENT PREDICTION ───────────────────────────────────────
async function loadCurrentPrediction() {
    showSpinner(true);
    try {
        const res = await fetch('/api/prediction/current');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cachedPrediction = data;

        updateHeroCard(data);
        updateConfidenceBanner(data);
        updateTopbar(data);
        renderPriceChart(data);
        await loadMetrics();

        $('last-updated').textContent = new Date().toLocaleTimeString();
    } catch (err) {
        console.error('Failed to load prediction', err);
    } finally {
        showSpinner(false);
    }
}

// ── Hero Card ─────────────────────────────────────────────────────
function updateHeroCard(data) {
    $('hero-price').textContent = fmt(data.current_price);
    $('hero-lower').textContent = fmtShort(data.prediction.lower);
    $('hero-upper').textContent = fmtShort(data.prediction.upper);
    $('hero-timestamp').textContent = `as of ${new Date().toLocaleString()}`;
    $('range-width').textContent = `Range width: ${fmt(data.prediction.width)}`;
}

// ── Topbar ────────────────────────────────────────────────────────
function updateTopbar(data) {
    $('topbar-price').textContent = fmt(data.current_price);
}

// ── Confidence Banner ─────────────────────────────────────────────
function updateConfidenceBanner(data) {
    const banner = $('confidence-banner');
    banner.classList.remove('hidden', 'green', 'yellow', 'red');
    banner.classList.add(data.confidence_color);

    const icons = { High: '✅', Medium: '⚡', Low: '⚠️' };
    $('confidence-icon').textContent = icons[data.confidence] || '🔮';
    $('confidence-label').textContent = `Model Confidence: ${data.confidence}`;
    $('confidence-msg').textContent = data.confidence_msg;
}

// ── Metrics ────────────────────────────────────────────────────────
async function loadMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        if (data.coverage !== null) {
            $('metric-coverage').textContent = pct(data.coverage);
            $('metric-width').textContent = fmt(data.avg_width);
            $('metric-winkler').textContent = data.avg_winkler.toFixed(1);
            $('metric-preds').textContent = data.num_predictions || '—';

            // Coverage badge
            const badge = $('coverage-badge');
            const cov = data.coverage;
            if (cov >= 0.93 && cov <= 0.97) {
                badge.textContent = '✓ On Target'; badge.className = 'metric-badge good';
            } else if (cov > 0.97) {
                badge.textContent = '↑ Too wide'; badge.className = 'metric-badge warn';
            } else {
                badge.textContent = '↓ Too narrow'; badge.className = 'metric-badge bad';
            }
        }
    } catch (e) { console.warn('Metrics not ready', e); }
}

// ── PRICE CHART ────────────────────────────────────────────────────
function renderPriceChart(data) {
    const ctx = $('priceChart').getContext('2d');
    const C = chartColors();
    const chartData = data.chart_data;

    const labels = chartData.map(d => d.timestamp);
    const closes = chartData.map(d => d.close);

    // Add "Next Hour" prediction point
    labels.push('Next Hour');
    closes.push(null);

    const lowerLine = new Array(closes.length).fill(null);
    const upperLine = new Array(closes.length).fill(null);
    // Connect last actual price to prediction ribbon
    lowerLine[lowerLine.length - 2] = closes[closes.length - 2];
    upperLine[upperLine.length - 2] = closes[closes.length - 2];
    lowerLine[lowerLine.length - 1] = data.prediction.lower;
    upperLine[upperLine.length - 1] = data.prediction.upper;

    if (priceChart) priceChart.destroy();

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'BTC Close Price',
                    data: closes,
                    borderColor: C.accent2,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.15,
                    pointRadius: 0, pointHitRadius: 12,
                    order: 1,
                },
                {
                    label: '95% Lower Bound',
                    data: lowerLine,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(16,185,129,0.15)',
                    fill: '+1',
                    pointRadius: 0, tension: 0, order: 2,
                },
                {
                    label: '95% Upper Bound',
                    data: upperLine,
                    borderColor: 'rgba(16,185,129,0.5)',
                    borderDash: [5, 4],
                    backgroundColor: 'transparent',
                    pointRadius: 0, tension: 0, order: 3,
                },
            ],
        },
        options: baseChartOptions({
            plugins: {
                legend: { labels: { color: C.text, boxWidth: 14, font: { family: 'Outfit' } } },
                tooltip: { mode: 'index', intersect: false },
            },
            scales: buildScales(C, true),
        }),
    });
}

// ── HISTOGRAM (Monte Carlo) ────────────────────────────────────────
function renderHistogram(data) {
    const ctx = $('histogramChart').getContext('2d');
    const C = chartColors();
    const hist = data.histogram;
    const lower = data.prediction.lower;
    const upper = data.prediction.upper;
    const current = data.current_price;

    // Compute percentile stats from bin centers and counts
    let totalCount = hist.counts.reduce((a, b) => a + b, 0);
    let cumCount = 0;
    let p25 = null, p50 = null, p975 = null;
    for (let i = 0; i < hist.counts.length; i++) {
        cumCount += hist.counts[i];
        const frac = cumCount / totalCount;
        if (!p25 && frac >= 0.025) p25 = hist.bin_centers[i];
        if (!p50 && frac >= 0.5) p50 = hist.bin_centers[i];
        if (!p975 && frac >= 0.975) p975 = hist.bin_centers[i];
    }

    $('mc-p025').textContent = p25 ? fmtShort(p25) : '—';
    $('mc-p50').textContent = p50 ? fmtShort(p50) : '—';
    $('mc-p975').textContent = p975 ? fmtShort(p975) : '—';
    $('mc-current').textContent = fmtShort(current);

    // Color bars: green if inside 95% CI, grey otherwise
    const barColors = hist.bin_centers.map(bc =>
        bc >= lower && bc <= upper
            ? 'rgba(16,185,129,0.7)'
            : 'rgba(150,150,150,0.3)'
    );

    if (histogramChart) histogramChart.destroy();

    histogramChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hist.bin_centers.map(v => fmtShort(v)),
            datasets: [{
                label: 'Simulated Prices',
                data: hist.counts,
                backgroundColor: barColors,
                borderRadius: 3,
                borderSkipped: false,
            }],
        },
        options: {
            ...baseChartOptions(),
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Count: ${ctx.parsed.y.toLocaleString()}`,
                        title: ctx => `Price: ${ctx[0].label}`,
                    },
                },
                annotation: {
                    annotations: {
                        lower: {
                            type: 'line', xMin: hist.bin_centers.findIndex(v => v >= lower),
                            xMax: hist.bin_centers.findIndex(v => v >= lower),
                            borderColor: C.red, borderWidth: 2, borderDash: [5, 4],
                            label: { content: '2.5%', enabled: true, color: C.red, position: 'end', font: { size: 10 } }
                        },
                        upper: {
                            type: 'line', xMin: hist.bin_centers.findIndex(v => v >= upper),
                            xMax: hist.bin_centers.findIndex(v => v >= upper),
                            borderColor: C.green, borderWidth: 2, borderDash: [5, 4],
                        },
                    }
                }
            },
            scales: buildScales(C, false),
        },
    });
}

// ── VOLATILITY CHART ───────────────────────────────────────────────
function renderVolChart(data) {
    const ctx = $('volChart').getContext('2d');
    const C = chartColors();
    const vol = data.volatility;
    const last48 = vol.last_48;
    const threshold = vol.threshold;

    $('vol-current').textContent = (vol.current * 100).toFixed(4) + '%';
    $('vol-threshold').textContent = (threshold * 100).toFixed(4) + '%';
    const conf = data.confidence;
    const confColors = { High: C.green, Medium: C.yellow, Low: C.red };
    const regimeEl = $('vol-regime');
    regimeEl.textContent = conf;
    regimeEl.style.color = confColors[conf] || C.text;

    const labels = last48.map((_, i) => `T-${last48.length - i}h`);
    const barColors = last48.map(v => v > threshold ? C.red : C.accent);

    if (volChart) volChart.destroy();

    volChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Rolling 10-bar Vol',
                    data: last48.map(v => (v * 100).toFixed(5)),
                    backgroundColor: barColors,
                    borderRadius: 3,
                    order: 2,
                },
                {
                    label: 'Threshold (70th pct)',
                    data: last48.map(() => (threshold * 100).toFixed(5)),
                    type: 'line',
                    borderColor: C.yellow,
                    borderDash: [6, 4],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    order: 1,
                },
            ],
        },
        options: {
            ...baseChartOptions(),
            plugins: {
                legend: { labels: { color: C.text, font: { family: 'Outfit' } } },
                tooltip: { mode: 'index', intersect: false },
            },
            scales: buildScales(C, false),
        },
    });
}

// ── BACKTEST ──────────────────────────────────────────────────────
async function loadBacktest() {
    try {
        const [metricsRes, rowsRes] = await Promise.all([
            fetch('/api/metrics'),
            fetch('/api/backtest/results?limit=50'),
        ]);
        const metrics = await metricsRes.json();
        const rows = await rowsRes.json();

        if (metrics.coverage !== null) {
            const covPct = (metrics.coverage * 100).toFixed(2);
            $('bt-coverage').textContent = `${covPct}%`;
            $('bt-width').textContent = fmt(metrics.avg_width);
            $('bt-winkler').textContent = metrics.avg_winkler.toFixed(2);
            $('bt-preds').textContent = metrics.num_predictions || rows.length;

            // Gauge arc (semicircle ~251px circumference for half-circle)
            const arcLen = Math.min(metrics.coverage / 1.0, 1.0) * 251;
            const arcEl = $('gauge-arc');
            arcEl.style.strokeDasharray = `${arcLen} 251`;
            $('gauge-text').textContent = `${covPct}%`;
            arcEl.style.stroke = metrics.coverage >= 0.93 ? '#22c55e' : metrics.coverage >= 0.90 ? '#eab308' : '#ef4444';
        }

        renderWinklerChart(rows);
        renderBacktestTable(rows);
    } catch (e) { console.error('Backtest load failed', e); }
}

function renderBacktestTable(rows) {
    const tbody = $('backtest-tbody');
    tbody.innerHTML = '';
    rows.slice().reverse().slice(0, 50).forEach((r, i) => {
        const hit = r.covered;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${new Date(r.timestamp).toLocaleString()}</td>
            <td>${fmtShort(r.actual)}</td>
            <td class="hit-no">${fmtShort(r.lower)}</td>
            <td class="hit-yes">${fmtShort(r.upper)}</td>
            <td>${fmt(r.width)}</td>
            <td>${r.winkler.toFixed(2)}</td>
            <td class="${hit ? 'hit-yes' : 'hit-no'}">${hit ? '✓ Hit' : '✗ Miss'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderWinklerChart(rows) {
    const ctx = $('winklerChart').getContext('2d');
    const C = chartColors();
    const labels = rows.slice(-50).map((r, i) => `T-${50 - i}`);
    const winklerVals = rows.slice(-50).map(r => r.winkler.toFixed(2));
    const colors = rows.slice(-50).map(r => r.covered ? C.green : C.red);

    if (winklerChart) winklerChart.destroy();
    winklerChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Winkler Score',
                data: winklerVals,
                backgroundColor: colors,
                borderRadius: 3,
            }],
        },
        options: {
            ...baseChartOptions(),
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Winkler: ${parseFloat(ctx.parsed.y).toFixed(2)} — ${rows[ctx.dataIndex]?.covered ? 'Hit ✓' : 'Miss ✗'}`,
                    }
                }
            },
            scales: buildScales(C, false),
        },
    });
}

// ── HISTORY ────────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch('/api/prediction/history?limit=50');
        const rows = await res.json();

        const hits = rows.filter(r => r.hit === 1).length;
        const total = rows.filter(r => r.hit !== null).length;
        if (total > 0) {
            $('live-accuracy').textContent = `Live Accuracy: ${(hits / total * 100).toFixed(1)}%`;
        }

        const tbody = $('history-tbody');
        tbody.innerHTML = '';
        rows.forEach((r, i) => {
            const width = r.lower && r.upper ? r.upper - r.lower : null;
            let resultHtml = '<span class="hit-pending">Pending</span>';
            if (r.hit === 1) resultHtml = '<span class="hit-yes">✓ Hit</span>';
            else if (r.hit === 0) resultHtml = '<span class="hit-no">✗ Miss</span>';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td>${new Date(r.timestamp).toLocaleString()}</td>
                <td>${r.price ? fmtShort(r.price) : '—'}</td>
                <td class="hit-no">${r.lower ? fmtShort(r.lower) : '—'}</td>
                <td class="hit-yes">${r.upper ? fmtShort(r.upper) : '—'}</td>
                <td>${width ? fmt(width) : '—'}</td>
                <td>${r.actual ? fmtShort(r.actual) : '—'}</td>
                <td>${resultHtml}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) { console.error('History load failed', e); }
}

// ── Shared chart helpers ───────────────────────────────────────────
function baseChartOptions(extra = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        ...extra,
    };
}

function buildScales(C, withYLabel) {
    return {
        x: {
            grid: { color: C.grid },
            ticks: { color: C.text, font: { family: 'Outfit', size: 10 }, maxTicksLimit: 10 },
        },
        y: {
            grid: { color: C.grid },
            ticks: { color: C.text, font: { family: 'Outfit', size: 10 } },
        },
    };
}

// ── Auto-refresh every 60 seconds ──────────────────────────────────
setInterval(loadCurrentPrediction, 60000);

// ── Init ───────────────────────────────────────────────────────────
loadCurrentPrediction();

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 1 — Prediction Decay Countdown Timer
// ══════════════════════════════════════════════════════════════════
function startCountdown() {
    function tick() {
        const now = new Date();
        const secPast = now.getMinutes() * 60 + now.getSeconds();
        const secLeft = 3600 - (secPast % 3600);
        const min = Math.floor(secLeft / 60).toString().padStart(2, '0');
        const sec = (secLeft % 60).toString().padStart(2, '0');
        const textEl = document.getElementById('countdown-text');
        const arc = document.getElementById('countdown-arc');
        if (textEl) textEl.textContent = `${min}:${sec}`;
        if (arc) {
            const pct = secLeft / 3600;
            const circumference = 2 * Math.PI * 18; // r=18
            arc.style.strokeDashoffset = circumference * (1 - pct);
            arc.style.stroke = pct > 0.5 ? 'var(--accent)' : pct > 0.2 ? 'var(--yellow)' : 'var(--red)';
        }
    }
    tick();
    setInterval(tick, 1000);
}
startCountdown();

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 2 — Market Regime Classifier
// ══════════════════════════════════════════════════════════════════
async function loadRegime() {
    try {
        const res = await fetch('/api/regime');
        const d = await res.json();
        const $ = (id) => document.getElementById(id);
        const regimeColors = { 'Trending': '#f97316', 'Range-Bound': '#10b981', 'Pre-Breakout': '#eab308', 'High-Fear': '#ef4444' };
        const regimeDescs = {
            'Trending': 'Strong directional momentum. Model widens range in trend direction.',
            'Range-Bound': 'Price oscillating near mean. Mean-reversion likely. Expect tighter ranges.',
            'Pre-Breakout': 'Volatility accelerating from low base. A significant move may be imminent.',
            'High-Fear': 'Extreme negative momentum and high volatility. Exercise caution.'
        };
        $('regime-dominant').textContent = d.dominant_regime;
        $('regime-dominant').style.color = regimeColors[d.dominant_regime] || 'var(--accent)';
        $('regime-desc').textContent = regimeDescs[d.dominant_regime] || '';

        // Probability bars
        const barsEl = $('regime-prob-bars');
        barsEl.innerHTML = '';
        Object.entries(d.regime_probabilities).forEach(([name, pct]) => {
            const color = regimeColors[name] || 'var(--accent)';
            barsEl.innerHTML += `
                <div class="regime-prob-bar-wrap">
                    <div class="rpb-label"><span>${name}</span><span style="color:${color};font-weight:700">${pct}%</span></div>
                    <div class="regime-prob-bar-track">
                        <div class="regime-prob-bar-fill" style="width:${pct}%;background:${color}"></div>
                    </div>
                </div>`;
        });

        // Signal cards
        const m = d.momentum;
        $('sig-momentum-dir').textContent = m.direction;
        $('sig-momentum-dir').style.color = m.direction === 'Bullish' ? 'var(--green)' : 'var(--red)';
        $('sig-momentum-str').textContent = `${m.strength}%`;
        $('sig-momentum-10').textContent = `${m.rolling_10h_pct > 0 ? '+' : ''}${m.rolling_10h_pct}%`;
        $('sig-momentum-run').textContent = `${m.consecutive_bars} bars`;

        const mr = d.mean_reversion;
        $('sig-zscor').textContent = mr.z_score.toFixed(3);
        $('sig-mean').textContent = `$${mr['48h_mean'].toLocaleString()}`;
        const diff = mr.price_vs_48h_mean;
        $('sig-vs-mean').textContent = `${diff >= 0 ? '+' : ''}$${diff.toFixed(2)}`;
        $('sig-vs-mean').style.color = diff >= 0 ? 'var(--green)' : 'var(--red)';

        const va = d.volatility_accel;
        $('sig-vol-ratio').textContent = va.ratio.toFixed(3) + '×';
        $('sig-vol-status').textContent = va.interpretation;
        $('sig-range-pct').textContent = `${d.range_24h.range_pct.toFixed(3)}%`;

        const lev = d.levels;
        const fmtLev = (v) => v ? `$${v.toLocaleString()}` : '—';
        $('sig-r1').textContent = fmtLev(lev.resistances[0]);
        $('sig-r2').textContent = fmtLev(lev.resistances[1]);
        $('sig-s1').textContent = fmtLev(lev.supports[0]);
        $('sig-s2').textContent = fmtLev(lev.supports[1]);
    } catch (e) { console.error('Regime load failed', e); }
}

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 3 — What-If Scenario Studio
// ══════════════════════════════════════════════════════════════════
let scenarioChart = null;
let scenarioDebounce = null;
let baseModelWidth = null;

async function runScenario() {
    const volMultiplier = parseFloat(document.getElementById('sl-vol').value);
    const driftBias = parseFloat(document.getElementById('sl-drift').value);
    const confLevel = parseFloat(document.getElementById('sl-conf').value) / 100;

    document.getElementById('sl-vol-val').textContent = `${volMultiplier.toFixed(1)}×`;
    document.getElementById('sl-drift-val').textContent = `${driftBias >= 0 ? '+' : ''}${driftBias.toFixed(2)}%`;
    document.getElementById('sl-conf-val').textContent = `${(confLevel * 100).toFixed(0)}%`;

    try {
        const res = await fetch('/api/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vol_multiplier: volMultiplier, drift_bias_pct: driftBias, confidence_level: confLevel, num_simulations: 10000 })
        });
        const d = await res.json();
        document.getElementById('sc-lower').textContent = fmtShort(d.lower);
        document.getElementById('sc-upper').textContent = fmtShort(d.upper);
        document.getElementById('sc-width').textContent = `Width: ${fmt(d.width)}`;
        if (baseModelWidth !== null) {
            const delta = d.width - baseModelWidth;
            const vsEl = document.getElementById('sc-vs');
            vsEl.textContent = `${delta >= 0 ? '+' : ''}${fmt(delta)}`;
            vsEl.style.color = delta > 0 ? 'var(--red)' : 'var(--green)';
        }
        renderScenarioChart(d);
    } catch (e) { console.error('Scenario failed', e); }
}

function renderScenarioChart(d) {
    const ctx = document.getElementById('scenarioChart').getContext('2d');
    const C = chartColors();
    const hist = d.histogram;
    const inBand = hist.bin_centers.map(bc => bc >= d.lower && bc <= d.upper ? 'rgba(16,185,129,0.6)' : 'rgba(150,150,150,0.2)');
    if (scenarioChart) scenarioChart.destroy();
    scenarioChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hist.bin_centers.map(v => fmtShort(v)),
            datasets: [{ label: 'Simulated Prices', data: hist.counts, backgroundColor: inBand, borderRadius: 3 }]
        },
        options: { ...baseChartOptions(), plugins: { legend: { display: false } }, scales: buildScales(C, false) }
    });
}

function initScenarioSliders() {
    ['sl-vol', 'sl-drift', 'sl-conf'].forEach(id => {
        document.getElementById(id).addEventListener('input', () => {
            clearTimeout(scenarioDebounce);
            scenarioDebounce = setTimeout(runScenario, 300);
        });
    });
}
initScenarioSliders();

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 4 — Tail Risk Dashboard
// ══════════════════════════════════════════════════════════════════
let tailChart = null;

async function loadTailRisk() {
    try {
        const res = await fetch('/api/tail-risk');
        const d = await res.json();
        const $ = (id) => document.getElementById(id);

        $('tr-var95').textContent = `${fmt(d.var.var_95_dollar)} (${d.var.var_95_pct}%)`;
        $('tr-var99').textContent = `${fmt(d.var.var_99_dollar)} (${d.var.var_99_pct}%)`;
        $('tr-cvar95').textContent = `${fmt(d.cvar.cvar_95_dollar)} (${d.cvar.cvar_95_pct}%)`;
        $('tr-cvar99').textContent = `${fmt(d.cvar.cvar_99_dollar)} (${d.cvar.cvar_99_pct}%)`;

        const mp = d.move_probabilities;
        $('tr-up1').textContent = `${mp.up_1pct.toFixed(2)}%`;
        $('tr-dn1').textContent = `${mp.down_1pct.toFixed(2)}%`;
        $('tr-up2').textContent = `${mp.up_2pct.toFixed(2)}%`;
        $('tr-dn2').textContent = `${mp.down_2pct.toFixed(2)}%`;

        const ds = d.distribution_shape;
        $('tr-skew').textContent = ds.skewness.toFixed(4);
        $('tr-kurt').textContent = ds.excess_kurtosis.toFixed(4);
        $('tr-df').textContent = ds.student_t_df.toFixed(2);
        $('tr-annvol').textContent = `${d.annualised_vol_pct.toFixed(2)}%`;

        $('tr-skew-interp').textContent = `⟳ Skewness: ${ds.interpretation_skew}`;
        $('tr-kurt-interp').textContent = `⟳ Kurtosis: ${ds.interpretation_kurt}`;
        $('tr-maxdd').textContent = `📉 Max Drawdown (24h): ${d.max_drawdown_24h_pct.toFixed(4)}%`;

        // Percentile distribution chart
        const ctx = $('tailChart').getContext('2d');
        const C = chartColors();
        const pctData = d.price_percentile_distribution;
        const labels = Object.keys(pctData).map(k => `${k}th`);
        const values = Object.values(pctData);
        const barColors = values.map((v, i) => {
            const pct = parseInt(Object.keys(pctData)[i]);
            if (pct <= 5) return C.red;
            if (pct >= 95) return C.green;
            if (pct === 50) return C.accent;
            return 'rgba(150,150,150,0.4)';
        });
        if (tailChart) tailChart.destroy();
        tailChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Price at Percentile', data: values, backgroundColor: barColors, borderRadius: 4 }]
            },
            options: {
                ...baseChartOptions(),
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => `Price: ${fmtShort(ctx.parsed.y)}` } }
                },
                scales: buildScales(C, false)
            }
        });
    } catch (e) { console.error('Tail risk load failed', e); }
}

// ── Extend activateSection to load new sections ──────────────────
const _origActivate = activateSection;
// Patch nav to call new loaders
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const sec = btn.dataset.section;
        if (sec === 'regime') loadRegime();
        if (sec === 'scenario') { if (cachedPrediction) { baseModelWidth = cachedPrediction.prediction.width; } runScenario(); }
        if (sec === 'tailrisk') loadTailRisk();
    });
});

// ══════════════════════════════════════════════════════════════════
// CURRENCY SYSTEM — Live exchange rates from frankfurter.app
// ══════════════════════════════════════════════════════════════════

// Currency config: symbol, decimals
const CURRENCY_META = {
    USD: { symbol: '$',  locale: 'en-US', decimals: 2 },
    EUR: { symbol: '€',  locale: 'de-DE', decimals: 2 },
    GBP: { symbol: '£',  locale: 'en-GB', decimals: 2 },
    INR: { symbol: '₹',  locale: 'en-IN', decimals: 0 },
    JPY: { symbol: '¥',  locale: 'ja-JP', decimals: 0 },
    CNY: { symbol: '¥',  locale: 'zh-CN', decimals: 2 },
    AUD: { symbol: 'A$', locale: 'en-AU', decimals: 2 },
    CAD: { symbol: 'C$', locale: 'en-CA', decimals: 2 },
    CHF: { symbol: 'Fr', locale: 'de-CH', decimals: 2 },
    SGD: { symbol: 'S$', locale: 'en-SG', decimals: 2 },
    AED: { symbol: 'د.إ',locale: 'ar-AE', decimals: 2 },
    SAR: { symbol: '﷼',  locale: 'ar-SA', decimals: 2 },
    BRL: { symbol: 'R$', locale: 'pt-BR', decimals: 2 },
    MXN: { symbol: '$',  locale: 'es-MX', decimals: 2 },
    KRW: { symbol: '₩',  locale: 'ko-KR', decimals: 0 },
    HKD: { symbol: 'HK$',locale: 'zh-HK', decimals: 2 },
    SEK: { symbol: 'kr', locale: 'sv-SE', decimals: 2 },
    NOK: { symbol: 'kr', locale: 'nb-NO', decimals: 2 },
    DKK: { symbol: 'kr', locale: 'da-DK', decimals: 2 },
    NZD: { symbol: 'NZ$',locale: 'en-NZ', decimals: 2 },
    ZAR: { symbol: 'R',  locale: 'en-ZA', decimals: 2 },
    TRY: { symbol: '₺',  locale: 'tr-TR', decimals: 2 },
    IDR: { symbol: 'Rp', locale: 'id-ID', decimals: 0 },
    MYR: { symbol: 'RM', locale: 'ms-MY', decimals: 2 },
    THB: { symbol: '฿',  locale: 'th-TH', decimals: 2 },
    PHP: { symbol: '₱',  locale: 'fil-PH',decimals: 2 },
    PLN: { symbol: 'zł', locale: 'pl-PL', decimals: 2 },
    CZK: { symbol: 'Kč', locale: 'cs-CZ', decimals: 2 },
    HUF: { symbol: 'Ft', locale: 'hu-HU', decimals: 0 },
    ILS: { symbol: '₪',  locale: 'he-IL', decimals: 2 },
};

let activeCurrency = 'USD';
let exchangeRates = { USD: 1 }; // rates relative to USD

async function fetchExchangeRates() {
    try {
        // frankfurter.app — free, no key, updated daily
        const res = await fetch('https://api.frankfurter.app/latest?from=USD');
        if (!res.ok) throw new Error('Rate fetch failed');
        const data = await res.json();
        exchangeRates = { USD: 1, ...data.rates };
        console.log('Exchange rates loaded:', exchangeRates);
    } catch (e) {
        console.warn('Could not fetch live rates, using fallback', e);
        // Reasonable fallback rates (approximate)
        exchangeRates = {
            USD:1, EUR:0.92, GBP:0.79, INR:83.5, JPY:149.8, CNY:7.24,
            AUD:1.53, CAD:1.36, CHF:0.90, SGD:1.34, AED:3.67, SAR:3.75,
            BRL:4.97, MXN:17.2, KRW:1330, HKD:7.82, SEK:10.4, NOK:10.6,
            DKK:6.89, NZD:1.63, ZAR:18.7, TRY:32.1, IDR:15700, MYR:4.72,
            THB:35.2, PHP:56.5, PLN:3.97, CZK:23.1, HUF:360, ILS:3.65,
        };
    }
    updateRateBadge();
    refreshAllDisplayedPrices();
}

function updateRateBadge() {
    const rate = exchangeRates[activeCurrency] || 1;
    const meta = CURRENCY_META[activeCurrency] || CURRENCY_META.USD;
    const badge = document.getElementById('exchange-rate-badge');
    if (badge) badge.textContent = `1 USD = ${rate.toLocaleString(meta.locale, { maximumFractionDigits: 4 })} ${activeCurrency}`;
}

// Override the global fmt/fmtShort to respect currency
window.convertPrice = function(usdAmount) {
    const rate = exchangeRates[activeCurrency] || 1;
    return usdAmount * rate;
};

window.fmt = function(n) {
    const converted = convertPrice(n);
    const meta = CURRENCY_META[activeCurrency] || CURRENCY_META.USD;
    return new Intl.NumberFormat(meta.locale, {
        style: 'currency', currency: activeCurrency,
        minimumFractionDigits: meta.decimals,
        maximumFractionDigits: meta.decimals,
    }).format(converted);
};

window.fmtShort = function(n) {
    const converted = convertPrice(n);
    const meta = CURRENCY_META[activeCurrency] || CURRENCY_META.USD;
    return new Intl.NumberFormat(meta.locale, {
        style: 'currency', currency: activeCurrency,
        minimumFractionDigits: meta.decimals,
        maximumFractionDigits: meta.decimals,
    }).format(converted);
};

function refreshAllDisplayedPrices() {
    if (!cachedPrediction) return;
    updateHeroCard(cachedPrediction);
    updateTopbar(cachedPrediction);
    updateRateBadge();
    renderPriceChart(cachedPrediction);
    // Re-load metrics if visible
    loadMetrics();
}

// Currency selector listener
document.getElementById('currency-select').addEventListener('change', (e) => {
    activeCurrency = e.target.value;
    updateRateBadge();
    refreshAllDisplayedPrices();
    // Also update visible section values
    const activeSection = document.querySelector('.section.active');
    if (activeSection) {
        const id = activeSection.id;
        if (id === 'section-backtest') loadBacktest();
        if (id === 'section-history') loadHistory();
        if (id === 'section-tailrisk') loadTailRisk();
        if (id === 'section-scenario') runScenario();
    }
});

// Init exchange rates on load
fetchExchangeRates();

"""

@app.get("/style.css")
def get_css():
    return Response(content=STYLE_CSS, media_type="text/css")

@app.get("/app.js")
def get_js():
    return Response(content=APP_JS, media_type="application/javascript")

@app.get("/")
def get_html():
    return HTMLResponse(content=INDEX_HTML)

