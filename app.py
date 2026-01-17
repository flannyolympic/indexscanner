import sqlite3

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
DB_NAME = "watchlist.db"

# Top 10 QQQ holdings
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


# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY, ticker TEXT, signal TEXT,
                  price REAL, strategy TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


# --- Technical Analysis Helpers ---
def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def get_market_data(ticker):
    try:
        # Download data
        df = yf.download(ticker, period="1d", interval="1m", progress=False)

        # --- THE FIX: Flatten complex headers ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # ----------------------------------------

        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def analyze_ticker(ticker):
    df = get_market_data(ticker)
    if df is None or len(df) < 20:
        return None

    # Calculate Indicators
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["RSI"] = calculate_rsi(df["Close"])

    latest = df.iloc[-1]
    price = latest["Close"]
    rsi = latest["RSI"]

    # Logic
    signal = "NEUTRAL"
    suggestion = "No Trade"

    if price < latest["BB_Lower"] and rsi < 30:
        signal = "BULLISH"
        suggestion = f"Buy 0DTE CALL | Strike: ${np.ceil(price)}"
    elif price > latest["BB_Upper"] and rsi > 70:
        signal = "BEARISH"
        suggestion = f"Buy 0DTE PUT | Strike: ${np.floor(price)}"
    elif price > latest["VWAP"] and df.iloc[-2]["Close"] < df.iloc[-2]["VWAP"]:
        signal = "BULLISH (VWAP)"
        suggestion = f"Buy 0DTE CALL | Strike: ${np.ceil(price)}"

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "rsi": round(rsi, 2),
        "vwap": round(latest["VWAP"], 2),
        "signal": signal,
        "suggestion": suggestion,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan")
def scan():
    results = [analyze_ticker(t) for t in QQQ_TICKERS]
    results = [r for r in results if r and r["signal"] != "NEUTRAL"]
    return render_template("index.html", results=results)


@app.route("/watchlist")
def watchlist():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    rows = (
        conn.cursor()
        .execute("SELECT * FROM watchlist ORDER BY timestamp DESC")
        .fetchall()
    )
    conn.close()
    return render_template("watchlist.html", rows=rows)


@app.route("/add_to_watchlist", methods=["POST"])
def add_to_watchlist():
    conn = sqlite3.connect(DB_NAME)
    conn.cursor().execute(
        "INSERT INTO watchlist (ticker, signal, price, strategy) VALUES (?, ?, ?, ?)",
        (
            request.form["ticker"],
            request.form["signal"],
            request.form["price"],
            request.form["strategy"],
        ),
    )
    conn.commit()
    conn.close()
    return redirect(url_for("watchlist"))


# --- Initialize DB on startup ---
init_db()  # <--- Moved this UP here!

if __name__ == "__main__":
    app.run(debug=True)
