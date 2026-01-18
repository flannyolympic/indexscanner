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
# A mix of Blue Chip, Crypto, and WSB Favorites
UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMD",
    "AMZN",
    "GOOGL",
    "META",  # Tech
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "DOGE-USD",
    "SHIB-USD",
    "PEPE-USD",  # Crypto
    "GME",
    "AMC",
    "PLTR",
    "COIN",
    "MSTR",
    "HOOD",
    "SPY",
    "QQQ",
    "IWM",  # Memes/ETFs
]


# --- DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY, ticker TEXT, signal TEXT,
                  price REAL, strategy TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


init_db()

# --- AI ENGINES ---


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
        # Fetch data for volatility and volume analysis
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def calculate_probability(price, target, std_dev, days=1):
    """
    Calculates 'Probability of Profit' (POP) using Z-Score & Normal Distribution.
    Logic: What are the odds price stays within or hits target based on current volatility?
    """
    # Simple Z-score model for target touch probability
    # Z = (Target - Price) / (Volatility * sqrt(Time))
    volatility = std_dev * np.sqrt(days)
    if volatility == 0:
        return 50.0

    z_score = abs(target - price) / volatility
    probability = 2 * (1 - norm.cdf(z_score))  # Two-tailed touch probability

    # Cap between 10% and 95% for realism
    pop = min(max(probability * 100, 12), 94)
    return round(pop, 1)


def get_social_sentiment(ticker, rsi, volume_spike):
    """
    Simulates WallStreetBets/Social Sentiment based on Hype Factors.
    """
    sentiments = []

    if rsi > 70 and volume_spike:
        score = "EXTREME FOMO"
        comment = random.choice(
            ["🚀 TO THE MOON", "💎🙌 DIAMOND HANDS", "Gamma Squeeze Incoming?"]
        )
        icon = "🔥"
    elif rsi < 30 and volume_spike:
        score = "PANIC SELLING"
        comment = random.choice(["GUH.", "Buy the Dip?", "Capitulation Detected"])
        icon = "🩸"
    elif volume_spike:
        score = "TRENDING"
        comment = random.choice(
            ["Apes are watching", "High Volume Alert", "Whale Activity"]
        )
        icon = "👀"
    else:
        score = "QUIET"
        comment = "Under the radar."
        icon = "💤"

    return {"score": score, "comment": comment, "icon": icon}


def analyze_ticker(ticker):
    ticker = ticker.upper().strip()
    df = get_market_data(ticker)

    if df is None:
        if not ticker.endswith("-USD"):
            ticker = f"{ticker}-USD"
            df = get_market_data(ticker)

    if df is None or len(df) < 20:
        return None

    # Technicals
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    # RSI
    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    prev = df.iloc[-5]  # 5 periods ago
    price = latest["Close"]
    std_dev = latest["Std_Dev"]

    # Volume Spike Detection (Current Vol vs 20-period Avg)
    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
    vol_spike = latest["Volume"] > (avg_vol * 1.5)

    # --- LOGIC CORE ---
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

    # --- AI PROBABILITY & SOCIAL ---
    target_price = (
        price + (std_dev * 2) if setup_type == "LONG" else price - (std_dev * 2)
    )
    pop = calculate_probability(price, target_price, std_dev)
    social = get_social_sentiment(ticker, latest["RSI"], vol_spike)

    # Generate Setup Text
    setup_text = {}
    if setup_type == "LONG":
        setup_text = {
            "type": "SNIPER LONG",
            "entry": f"${price:,.2f}",
            "target": f"${target_price:,.2f}",
            "stop": f"${(price - std_dev):,.2f}",
        }
    elif setup_type == "SHORT":
        setup_text = {
            "type": "SNIPER SHORT",
            "entry": f"${price:,.2f}",
            "target": f"${target_price:,.2f}",
            "stop": f"${(price + std_dev):,.2f}",
        }
    else:
        setup_text = {"type": "NO SETUP", "entry": "-", "target": "-", "stop": "-"}

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "rsi": round(latest["RSI"], 2),
        "vwap": round(latest["VWAP"], 2),
        "signal": signal,
        "suggestion": suggestion,
        "pop": pop,
        "social": social,
        "setup": setup_text,
        "vol_spike": vol_spike,
    }


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


# --- ROUTES ---
@app.route("/")
def index():
    return render_template(
        "index.html", vix=get_vix_data(), status_color=get_market_status_color()
    )


@app.route("/scan")
def scan():
    # 1. SCAN THE UNIVERSE
    # To keep it fast, we shuffle and pick 8 randoms + Bitcoin every time
    scan_list = random.sample(UNIVERSE, 8)
    if "BTC-USD" not in scan_list:
        scan_list.append("BTC-USD")

    raw_results = [analyze_ticker(t) for t in scan_list]
    results = [r for r in raw_results if r]  # Filter nones

    # 2. FIND "THE CHOSEN ONE" (Highest Probability Setup)
    # Filter for active signals first
    active_setups = [r for r in results if r["signal"] != "NEUTRAL"]

    # If no active setups, just pick the one with highest Volatility/Action
    if active_setups:
        # Sort by Probability (POP) descending
        chosen_one = sorted(active_setups, key=lambda x: x["pop"], reverse=True)[0]
    else:
        chosen_one = results[0] if results else None

    # 3. CALCULATE OVERALL MOOD
    bulls = sum(1 for r in results if "BULLISH" in r["signal"])
    bears = sum(1 for r in results if "BEARISH" in r["signal"])
    mood = "BULL" if bulls > bears else "BEAR"
    if bulls == bears:
        mood = "NEUTRAL"

    return render_template(
        "index.html",
        results=results,
        chosen_one=chosen_one,
        mood=mood,
        vix=get_vix_data(),
        status_color=get_market_status_color(),
    )


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return redirect(url_for("index"))
    result = analyze_ticker(query)
    results = [result] if result else []

    mood = "NEUTRAL"
    if result:
        if "BULLISH" in result["signal"]:
            mood = "BULL"
        elif "BEARISH" in result["signal"]:
            mood = "BEAR"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "rsi": round(latest["RSI"], 2),
        "vwap": round(latest["VWAP"], 2),
        "signal": signal,
        "suggestion": suggestion,
        "probability": pop,  # <--- CHANGED NAME HERE (Was "pop": pop)
        "social": social,
        "setup": setup_text,
        "vol_spike": vol_spike,
    }


@app.route("/add_to_watchlist", methods=["POST"])
def add_to_watchlist():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
