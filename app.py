import json
import os
import time
from datetime import datetime

import google.generativeai as genai
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request

# --- CONFIGURATION ---
app = Flask(__name__)

# API KEYS
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")

if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

# --- TARGET ASSETS ---
STOCK_TICKERS = [
    "TSLA",
    "NVDA",
    "AMD",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NFLX",
    "COIN",
    "MARA",
    "RIOT",
    "PLTR",
    "SOFI",
    "HOOD",
    "GME",
    "AMC",
    "SPY",
    "QQQ",
    "IWM",
]

CRYPTO_TICKERS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "DOGE-USD",
    "SHIB-USD",
    "ADA-USD",
    "AVAX-USD",
    "DOT-USD",
    "MATIC-USD",
    "LINK-USD",
    "LTC-USD",
    "BCH-USD",
    "UNI-USD",
    "ATOM-USD",
    "ETC-USD",
    "FIL-USD",
    "ICP-USD",
    "NEAR-USD",
    "APE-USD",
]


def get_market_status():
    """Returns simple market status string"""
    now = datetime.now()
    # Simple check for US Market hours (approximate)
    if 9 <= now.hour < 16 and now.weekday() < 5:
        return "MARKET OPEN"
    return "MARKET CLOSED"


# --- CORE SCANNER LOGIC ---
def analyze_market_data(ticker_list):
    """
    Fetches data and performs technical analysis.
    Returns a list of dictionaries.
    """
    results = []

    # Batch fetch for speed
    try:
        data = yf.download(
            tickers=" ".join(ticker_list),
            period="5d",
            interval="15m",
            group_by="ticker",
            auto_adjust=True,
            prepost=True,
            threads=True,
        )
    except Exception as e:
        print(f"Batch download failed: {e}")
        return []

    for ticker in ticker_list:
        try:
            # Handle single vs multi-ticker structure in yfinance
            if len(ticker_list) == 1:
                df = data
            else:
                df = data[ticker]

            if df.empty:
                continue

            # Technical Indicators
            # 1. VWAP (Approximation)
            df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
            df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

            # 2. RSI (14)
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            # Current values
            current_price = df["Close"].iloc[-1]
            current_rsi = df["RSI"].iloc[-1]
            current_vwap = df["VWAP"].iloc[-1]

            # --- SIGNAL LOGIC (THE "CEREBRO" ENGINE) ---
            signal = "NEUTRAL"
            probability = 50
            setup_type = "Consolidation"

            # Oversold Bounce
            if current_rsi < 30:
                signal = "BULLISH (OVERSOLD)"
                probability = 75
                setup_type = "Mean Reversion"

            # Overbought Pullback
            elif current_rsi > 70:
                signal = "BEARISH (OVERBOUGHT)"
                probability = 70
                setup_type = "Top Reversal"

            # Trend Continuation
            elif current_price > current_vwap * 1.01:
                signal = "BULLISH (TREND)"
                probability = 65
                setup_type = "VWAP Hold"

            elif current_price < current_vwap * 0.99:
                signal = "BEARISH (TREND)"
                probability = 65
                setup_type = "VWAP Reject"

            # Entry/Stop Logic
            stop_loss = (
                current_price * 0.98 if "BULLISH" in signal else current_price * 1.02
            )
            target = (
                current_price * 1.05 if "BULLISH" in signal else current_price * 0.95
            )

            results.append(
                {
                    "ticker": ticker,
                    "price": round(float(current_price), 2),
                    "signal": signal,
                    "probability": probability,
                    "vwap": round(float(current_vwap), 2),
                    "setup": {
                        "type": setup_type,
                        "entry": f"MKT {round(float(current_price), 2)}",
                        "stop": f"{round(float(stop_loss), 2)}",
                        "target": f"{round(float(target), 2)}",
                    },
                }
            )

        except Exception as e:
            continue

    # Sort by probability (Highest first)
    return sorted(results, key=lambda x: x["probability"], reverse=True)


def get_ai_rationale(ticker_data):
    """
    Generates a professional analyst rationale using Gemini.
    """
    if not GENAI_API_KEY:
        return "AI Module Offline: Analysis unavailable."

    try:
        model = genai.GenerativeModel("gemini-pro")

        prompt = (
            f"Act as a senior quantitative analyst for a hedge fund. "
            f"Analyze this technical setup for {ticker_data['ticker']}: "
            f"Price: {ticker_data['price']}, Signal: {ticker_data['signal']}, "
            f"Setup Type: {ticker_data['setup']['type']}. "
            f"Provide a concise, 2-sentence rationale focusing on risk/reward and market structure. "
            f"Do not use slang. Be professional, direct, and institutional."
        )

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "AI Analysis timed out."


# --- ROUTES ---
@app.route("/")
def home():
    return render_template(
        "index.html", market_status=get_market_status(), current_mode="stock"
    )


@app.route("/scan")
def scan():
    mode = request.args.get("mode", "stock")
    tickers = CRYPTO_TICKERS if mode == "crypto" else STOCK_TICKERS

    start_time = time.time()
    results = analyze_market_data(tickers)
    scan_time = time.time() - start_time
    print(f"Scan complete in {round(scan_time, 2)}s")

    # Select the "High Conviction" Pick (Top result)
    chosen_one = None
    if results:
        chosen_one = results[0]
        # Enrich the top pick with AI and "Social" data
        chosen_one["rationale"] = get_ai_rationale(chosen_one)

        # Simulated Social Sentiment (Placeholder for Reddit/Twitter API)
        chosen_one["social"] = {"score": "HIGH_MENTION_VOLUME", "icon": "🔥"}

    return render_template(
        "index.html",
        market_status=get_market_status(),
        current_mode=mode,
        results=results,
        chosen_one=chosen_one,
    )


@app.route("/api/vix")
def get_vix():
    # Simple VIX fetch for the header widget
    try:
        vix = yf.Ticker("^VIX").history(period="1d")
        if not vix.empty:
            price = round(vix["Close"].iloc[-1], 2)
            label = "ELEVATED" if price > 20 else "STABLE"
            return jsonify({"price": price, "label": label})
    except:
        pass
    return jsonify({"price": "...", "label": "N/A"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
