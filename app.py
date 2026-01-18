import random
import sqlite3
from datetime import datetime, time

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from flask import Flask, redirect, render_template, request, url_for
from scipy.stats import norm

app = Flask(__name__)
DB_NAME = "watchlist.db"

# --- THE MATRIX UNIVERSE ---
UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMD",
    "AMZN",
    "GOOGL",
    "META",
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "DOGE-USD",
    "SHIB-USD",
    "GME",
    "AMC",
    "PLTR",
    "COIN",
    "MSTR",
    "SPY",
    "QQQ",
    "IWM",
]


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY, ticker TEXT, signal TEXT,
                  price REAL, strategy TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


init_db()


# --- HELPERS ---
def get_current_time():
    """Returns formatted live NY time"""
    tz = pytz.timezone("US/Eastern")
    return datetime.now(tz).strftime("%H:%M:%S EST")


def get_market_status_color():
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return "#f28b82"
    current_time = now.time()
    if time(9, 30) <= current_time <= time(16, 0):
        return "#81c995"
    return "#f28b82"


def get_market_data(ticker):
    try:
        # Fetch 5 days of data to ensure we have enough for volatility
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 5:
            return None
        return df
    except Exception:
        return None


def calculate_probability(price, target, std_dev, days=3):
    """
    Calculates probability of touching target within 'days'.
    Fixed logic: Uses 3-day volatility window for realistic swing trade odds.
    """
    # Volatility scales with square root of time
    volatility = std_dev * np.sqrt(days)

    if volatility == 0:
        return 50.0

    # Z-Score: How many standard deviations away is the target?
    z_score = abs(target - price) / volatility

    # Survival Function (sf) gives probability of exceeding Z
    # We multiply by 2 for "touch" probability (up or down path)
    prob = 2 * norm.sf(z_score)

    # Scale to percentage and clamp
    pop = prob * 100

    # Realistic Caps: Even the best trade isn't 100%, worst isn't 0%
    return round(min(max(pop, 15), 88), 1)


def get_social_sentiment(ticker, rsi, volume_spike):
    if rsi > 70 and volume_spike:
        return {
            "score": "MAX HYPE",
            "comment": random.choice(["🚀 MOON MISSION", "💎🙌 HODL"]),
            "icon": "🔥",
        }
    elif rsi < 30 and volume_spike:
        return {
            "score": "FEAR SPIKE",
            "comment": random.choice(["🩸 Blood in streets", "Capitulation?"]),
            "icon": "🩸",
        }
    elif volume_spike:
        return {"score": "TRENDING", "comment": "Whales Entering", "icon": "👀"}
    else:
        return {"score": "QUIET", "comment": "Accumulation Zone", "icon": "💤"}


def get_vix_data():
    df = get_market_data("^VIX")
    if df is None:
        return {"price": "ERR", "color": "grey"}
    price = df.iloc[-1]["Close"]
    if price < 15:
        return {"price": round(price, 2), "color": "#00e676"}
    elif 15 <= price < 20:
        return {"price": round(price, 2), "color": "#ffea00"}
    elif 20 <= price < 30:
        return {"price": round(price, 2), "color": "#ff9100"}
    else:
        return {"price": round(price, 2), "color": "#ff1744"}


# --- ANALYZER ---
def analyze_ticker(ticker_input):
    ticker = ticker_input.upper().strip()
    df = get_market_data(ticker)

    if df is None:
        if not ticker.endswith("-USD"):
            crypto_try = f"{ticker}-USD"
            df_crypto = get_market_data(crypto_try)
            if df_crypto is not None:
                df = df_crypto
                ticker = crypto_try

    if df is None:
        return None

    # Technicals
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    price = latest["Close"]
    std_dev = latest["Std_Dev"]

    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        avg_vol = 1
    vol_spike = latest["Volume"] > (avg_vol * 1.5)

    # Signal Logic
    signal = "NEUTRAL"
    suggestion = "Wait"
    setup_type = "NONE"

    if price < latest["BB_Lower"] and latest["RSI"] < 30:
        signal = "BULLISH"
        suggestion = "Oversold Reversion"
        setup_type = "LONG"
    elif price > latest["BB_Upper"] and latest["RSI"] > 70:
        signal = "BEARISH"
        suggestion = "Overbought Rejection"
        setup_type = "SHORT"
    elif price > latest["VWAP"] and vol_spike:
        signal = "BULLISH (MOMENTUM)"
        suggestion = "Volume Breakout"
        setup_type = "LONG"

    # AI Setup - Target 1.5 Std Dev (Realistic Swing)
    target_dist = std_dev * 1.5
    target_price = price + target_dist if setup_type == "LONG" else price - target_dist

    # Calculate Probability (3 Day Outlook)
    probability = calculate_probability(price, target_price, std_dev, days=3)

    social = get_social_sentiment(ticker, latest["RSI"], vol_spike)

    setup_text = {"type": "NO SETUP", "entry": "-", "target": "-", "stop": "-"}
    if setup_type == "LONG":
        setup_text = {
            "type": "SWING LONG",
            "entry": f"${price:,.2f}",
            "target": f"${target_price:,.2f}",
            "stop": f"${(price - std_dev):,.2f}",
        }
    elif setup_type == "SHORT":
        setup_text = {
            "type": "SWING SHORT",
            "entry": f"${price:,.2f}",
            "target": f"${target_price:,.2f}",
            "stop": f"${(price + std_dev):,.2f}",
        }

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "rsi": round(latest["RSI"], 2),
        "vwap": round(latest["VWAP"], 2),
        "signal": signal,
        "suggestion": suggestion,
        "probability": probability,
        "social": social,
        "setup": setup_text,
    }


# --- ROUTES ---
@app.route("/")
def index():
    return render_template(
        "index.html",
        vix=get_vix_data(),
        status_color=get_market_status_color(),
        timestamp=get_current_time(),
    )


@app.route("/scan")
def scan():
    scan_list = random.sample(UNIVERSE, 8)
    if "BTC-USD" not in scan_list:
        scan_list.append("BTC-USD")

    raw_results = [analyze_ticker(t) for t in scan_list]
    results = [r for r in raw_results if r]

    active_setups = [r for r in results if r["signal"] != "NEUTRAL"]
    if active_setups:
        chosen_one = sorted(
            active_setups, key=lambda x: x["probability"], reverse=True
        )[0]
    else:
        chosen_one = results[0] if results else None

    bulls = sum(1 for r in results if "BULLISH" in r["signal"])
    bears = sum(1 for r in results if "BEARISH" in r["signal"])
    mood = "BULL" if bulls > bears else "BEAR"

    return render_template(
        "index.html",
        results=results,
        chosen_one=chosen_one,
        mood=mood,
        vix=get_vix_data(),
        status_color=get_market_status_color(),
        timestamp=get_current_time(),
    )


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return redirect(url_for("index"))

    result = analyze_ticker(query)

    if result is None:
        return render_template(
            "index.html",
            results=[],
            error=f"Could not find data for '{query}'",
            vix=get_vix_data(),
            status_color=get_market_status_color(),
            timestamp=get_current_time(),
        )

    results = [result]
    mood = "NEUTRAL"
    if "BULLISH" in result["signal"]:
        mood = "BULL"
    elif "BEARISH" in result["signal"]:
        mood = "BEAR"

    return render_template(
        "index.html",
        results=results,
        chosen_one=result,
        mood=mood,
        vix=get_vix_data(),
        status_color=get_market_status_color(),
        timestamp=get_current_time(),
    )


if __name__ == "__main__":
    app.run(debug=True)
