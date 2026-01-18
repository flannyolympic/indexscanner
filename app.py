import sqlite3

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
DB_NAME = "watchlist.db"

# --- CONFIGURATION ---
QQQ_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "GOOG",
    "AVGO",
    "PEP",
]


# --- DATABASE SETUP (Kept in background) ---
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
def get_market_data(ticker):
    try:
        # Fetch data
        df = yf.download(ticker, period="1d", interval="1m", progress=False)

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 5:
            return None
        return df
    except Exception:
        return None


def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --- ANALYSIS LOGIC ---
def analyze_ticker(ticker):
    ticker = ticker.upper().strip()
    df = get_market_data(ticker)

    # Smart Retry for Crypto (e.g., if "BTC" fails, try "BTC-USD")
    if df is None:
        if not ticker.endswith("-USD"):
            ticker = f"{ticker}-USD"
            df = get_market_data(ticker)

    if df is None or len(df) < 20:
        return None

    # Indicators
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["RSI"] = calculate_rsi(df["Close"])
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    latest = df.iloc[-1]
    price = latest["Close"]
    rsi = latest["RSI"]
    vwap = latest["VWAP"]

    # Logic
    signal = "NEUTRAL"
    suggestion = "Wait for Setup"

    if price < latest["BB_Lower"] and rsi < 30:
        signal = "BULLISH"
        suggestion = f"Oversold Bounce | Call Strike: ${np.ceil(price)}"
    elif price > latest["BB_Upper"] and rsi > 70:
        signal = "BEARISH"
        suggestion = f"Overbought Reject | Put Strike: ${np.floor(price)}"
    elif price > vwap and df.iloc[-2]["Close"] < df.iloc[-2]["VWAP"]:
        signal = "BULLISH (VWAP)"
        suggestion = f"Momentum Breakout | Call Strike: ${np.ceil(price)}"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "rsi": round(rsi, 2),
        "vwap": round(vwap, 2),
        "signal": signal,
        "suggestion": suggestion,
    }


# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan")
def scan():
    # The Classic QQQ Scan
    results = [analyze_ticker(t) for t in QQQ_TICKERS]
    results = [r for r in results if r and r["signal"] != "NEUTRAL"]
    return render_template("index.html", results=results, title="QQQ Scan Results")


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return redirect(url_for("index"))

    result = analyze_ticker(query)
    results = [result] if result else []

    return render_template(
        "index.html", results=results, title=f"Search: {query.upper()}"
    )


@app.route("/vix")
def vix():
    df = get_market_data("^VIX")
    if df is None:
        return "Error fetching VIX"
    price = df.iloc[-1]["Close"]

    if price < 15:
        sentiment = "LOW"
        message = "Market Complacent (Risk On)"
    elif 15 <= price < 20:
        sentiment = "ELEVATED"
        message = "Normal Volatility"
    elif 20 <= price < 30:
        sentiment = "FEAR"
        message = "High Fear (Puts Expensive)"
    else:
        sentiment = "EXTREME"
        message = "Capitulation / Panic"

    return render_template(
        "vix.html", price=round(price, 2), sentiment=sentiment, message=message
    )


# (Kept hidden route just in case you want to save later)
@app.route("/add_to_watchlist", methods=["POST"])
def add_to_watchlist():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
