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


# --- DATA HELPERS ---
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
    """Fetches live VIX and determines color/sentiment for the UI"""
    df = get_market_data("^VIX")
    if df is None:
        return {"price": "ERR", "color": "grey", "label": "Offline"}

    price = df.iloc[-1]["Close"]

    # Dynamic Coloring Logic
    if price < 15:
        return {
            "price": round(price, 2),
            "color": "#00e676",
            "label": "LOW",
        }  # Bright Green
    elif 15 <= price < 20:
        return {
            "price": round(price, 2),
            "color": "#ffea00",
            "label": "ELEVATED",
        }  # Bright Yellow
    elif 20 <= price < 30:
        return {"price": round(price, 2), "color": "#ff9100", "label": "FEAR"}  # Orange
    else:
        return {"price": round(price, 2), "color": "#ff1744", "label": "PANIC"}  # Red


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

    # RSI
    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # VWAP
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    latest = df.iloc[-1]
    price = latest["Close"]

    # Logic
    signal = "NEUTRAL"
    suggestion = "Wait for Setup"

    if price < latest["BB_Lower"] and latest["RSI"] < 30:
        signal = "BULLISH"
        suggestion = f"Oversold Bounce | Call Strike: ${np.ceil(price)}"
    elif price > latest["BB_Upper"] and latest["RSI"] < 70:  # Fixed logic typo
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
    # Pass VIX data to home page
    return render_template("index.html", vix=get_vix_data())


@app.route("/scan")
def scan():
    results = [analyze_ticker(t) for t in QQQ_TICKERS]
    results = [r for r in results if r and r["signal"] != "NEUTRAL"]
    # Pass VIX data to scan results
    return render_template(
        "index.html", results=results, title="QQQ Scan Results", vix=get_vix_data()
    )


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    vix = get_vix_data()  # Fetch VIX even during search

    if not query:
        return redirect(url_for("index"))

    result = analyze_ticker(query)
    results = [result] if result else []

    return render_template(
        "index.html", results=results, title=f"Search: {query.upper()}", vix=vix
    )


@app.route("/add_to_watchlist", methods=["POST"])
def add_to_watchlist():
    # Placeholder for database save
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
