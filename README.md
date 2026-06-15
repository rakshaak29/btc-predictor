# ₿ BTC Range Predictor

##  Link
https://btc-predictor-k5pp.onrender.com/

Hey there! Welcome to the **BTC Range Predictor** — a full-stack, real-time Bitcoin price prediction engine.

I built this project for the **AlphaI × Polaris Build Challenge**. While the core requirement was to build a prediction model using Geometric Brownian Motion (GBM), I wanted to take things several steps further. Instead of just a basic chart, I built a complete, institution-grade analytics dashboard that feels alive, interactive, and gives you deep insights into the market.

##  What Makes This Special?

I didn't just stop at predicting the next hour's price. I added several **unique, stand-out features** that you won't find in standard reference tools:

*    **Regime AI Classifier:** Automatically analyzes the current market structure (Trending, Range-Bound, Pre-Breakout, or High-Fear) using momentum, mean-reversion, and volatility acceleration signals. It even calculates local support and resistance levels for you!
*   **What-If Scenario Studio:** An interactive playground. Drag the sliders to change the Volatility Multiplier, Drift Bias, or Confidence Level, and watch the GBM Monte Carlo simulation re-run in **real-time** right before your eyes.
*    **Tail Risk Dashboard:** Serious risk metrics. It calculates Value at Risk (VaR), Expected Shortfall (CVaR), Skewness, Kurtosis, and the exact probability of +/- 1% and 2% price moves.
*    **Live Global Currencies:** A built-in currency switcher that fetches live exchange rates, allowing you to view all predictions and metrics in over 30 global currencies (USD, EUR, INR, JPY, GBP, etc.).
*    **Prediction Decay Timer:** A live countdown ring that shows exactly how fresh (or stale) the current hourly prediction is.

##  Tech Stack

I kept the stack lean, fast, and dependency-light:

*   **Backend:** Python, FastAPI, Pandas, NumPy, SciPy
*   **Model:** Geometric Brownian Motion (GBM) with Student-t fat tails and EWMA (Exponentially Weighted Moving Average) volatility clustering.
*   **Frontend:** Pure HTML, CSS (Vanilla), and Vanilla JavaScript. (No heavy JS frameworks!)
*   **Charts:** Chart.js
*   **Database:** SQLite (for tracking historical predictions and backtesting accuracy)
*   **Data Source:** Binance Public API (No API keys required)

##  Design

The UI was built from scratch with a strict **"No Blue"** dark mode policy. It uses a sleek combination of Charcoal surfaces, Emerald Green for bullish indicators, and Sunset Orange for bearish indicators/accents. 
