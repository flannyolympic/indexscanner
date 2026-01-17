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


# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist
                 (id INTEGER PRIMARY KEY, ticker TEXT, signal TEXT,
                  price REAL, strategy TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


# Initialize DB immediately
init_db()


# --- HELPERS ---
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


def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --- ANALYSIS LOGIC ---
def analyze_ticker(ticker):
    df = get_market_data(ticker)
    if df is None or len(df) < 20:
        return None

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["RSI"] = calculate_rsi(df["Close"])

    # VWAP Calculation
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    latest = df.iloc[-1]
    price = latest["Close"]
    rsi = latest["RSI"]

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


# --- ROUTES ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scan")
def scan():
    results = [analyze_ticker(t) for t in QQQ_TICKERS]
    results = [r for r in results if r and r["signal"] != "NEUTRAL"]
    return render_template("index.html", results=results)


@app.route("/vix")
def vix():
    # Fetch VIX Data
    df = get_market_data("^VIX")
    if df is None:
        return "Error fetching VIX data"

    price = df.iloc[-1]["Close"]

    # Determine Sentiment
    if price < 15:
        sentiment = "LOW"
        message = "Market is complacent. Grind up likely."
    elif 15 <= price < 20:
        sentiment = "ELEVATED"
        message = "Normal volatility. Watch for chops."
    elif 20 <= price < 30:
        sentiment = "FEAR"
        message = "High Fear. Puts are expensive. Careful."
    else:
        sentiment = "EXTREME"
        message = "Panic selling. Potential capitulation bottom."

    return render_template(
        "vix.html", price=round(price, 2), sentiment=sentiment, message=message
    )


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


if __name__ == "__main__":
    app.run(debug=True)
