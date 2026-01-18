import sqlite3
from datetime import datetime, time

import numpy as np
import pandas as pd
import pytz  # Library to handle Timezones
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


# --- HELPERS ---
def get_market_status_color():
    """Checks if US Market is Open (Green) or Closed (Red)"""
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)

    # Check Weekday (0=Mon, 6=Sun)
    if now.weekday() >= 5:
        return "#f28b82"  # Red (Closed - Weekend)

    # Check Time (09:30 - 16:00)
    current_time = now.time()
    market_open = time(9, 30)
    market_close = time(16, 0)

    if market_open <= current_time <= market_close:
        return "#81c995"  # Green (Open)
    else:
        return "#f28b82"  # Red (Closed - After Hours)


def get_market_data(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            return None
        return df
    except Exception:
        return None


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


def analyze_ticker(ticker):
    ticker = ticker.upper().strip()
    df = get_market_data(ticker)

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

    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    latest = df.iloc[-1]
    price = latest["Close"]

    signal = "NEUTRAL"
    suggestion = "Wait for Setup"

    if price < latest["BB_Lower"] and latest["RSI"] < 30:
        signal = "BULLISH"
        suggestion = f"Oversold Bounce | Call Strike: ${np.ceil(price)}"
    elif price > latest["BB_Upper"] and latest["RSI"] > 70:
        signal = "BEARISH"
        suggestion = f"Overbought Reject | Put Strike: ${np.floor(price)}"
    elif price > latest["VWAP"] and df.iloc[-2]["Close"] < df.iloc[-2]["VWAP"]:
        signal = "BULLISH (VWAP)"
        suggestion = f"Momentum Breakout | Call Strike: ${np.ceil(price)}"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "rsi": round(latest["RSI"], 2),
        "vwap": round(latest["VWAP"], 2),
        "signal": signal,
        "suggestion": suggestion,
    }


# --- ROUTES ---
@app.route("/")
def index():
    return render_template(
        "index.html", vix=get_vix_data(), status_color=get_market_status_color()
    )


@app.route("/scan")
def scan():
    results = [analyze_ticker(t) for t in QQQ_TICKERS]
    results = [r for r in results if r and r["signal"] != "NEUTRAL"]
    return render_template(
        "index.html",
        results=results,
        title="QQQ Scan Results",
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
    return render_template(
        "index.html",
        results=results,
        title=f"Search: {query.upper()}",
        vix=get_vix_data(),
        status_color=get_market_status_color(),
    )


@app.route("/add_to_watchlist", methods=["POST"])
def add_to_watchlist():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
