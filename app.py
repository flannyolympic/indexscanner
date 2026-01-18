import sqlite3
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytz
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
        # Fetch 5 days of 15m data for better volatility calculation
        df = yf.download(ticker, period="5d", interval="15m", progress=False)
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


# --- AI TRADE ARCHITECT ---
def generate_trade_setup(ticker, price, signal, upper_bb, lower_bb, std_dev):
    """
    Constructs a specific Option or Crypto spread based on Volatility.
    """
    setup = {}
    is_crypto = "-USD" in ticker or ticker in ["BTC", "ETH", "SOL", "DOGE"]

    # 1. EXPIRATION LOGIC (For Options)
    # If today is Friday, aim for next week. Otherwise 0DTE or 1DTE.
    today = datetime.now().weekday()
    expiry = "0DTE (Today)" if today < 4 else "Next Friday"

    # 2. STRATEGY CONSTRUCTION
    if "BULLISH" in signal:
        if is_crypto:
            # Crypto Long Setup
            risk = std_dev * 1.5
            entry = price
            stop_loss = entry - risk
            take_profit = entry + (risk * 2)  # 2:1 Reward Ratio
            setup = {
                "type": "CRYPTO LONG",
                "leg1": f"Entry: ${entry:,.2f}",
                "leg2": f"Stop: ${stop_loss:,.2f}",
                "leg3": f"Target: ${take_profit:,.2f} (2:1)",
            }
        else:
            # Stock Call Debit Spread
            # Buy ATM, Sell resistance (Upper BB)
            buy_strike = np.ceil(price)
            sell_strike = np.ceil(upper_bb)
            if sell_strike <= buy_strike:
                sell_strike = buy_strike + 2.5  # Minimum width

            setup = {
                "type": f"BULL CALL SPREAD ({expiry})",
                "leg1": f"BUY Call Strike: ${buy_strike}",
                "leg2": f"SELL Call Strike: ${sell_strike}",
                "leg3": "Net Debit (Low Risk)",
            }

    elif "BEARISH" in signal:
        if is_crypto:
            # Crypto Short Setup
            risk = std_dev * 1.5
            entry = price
            stop_loss = entry + risk
            take_profit = entry - (risk * 2)
            setup = {
                "type": "CRYPTO SHORT",
                "leg1": f"Entry: ${entry:,.2f}",
                "leg2": f"Stop: ${stop_loss:,.2f}",
                "leg3": f"Target: ${take_profit:,.2f} (2:1)",
            }
        else:
            # Stock Put Debit Spread
            # Buy ATM, Sell Support (Lower BB)
            buy_strike = np.floor(price)
            sell_strike = np.floor(lower_bb)
            if sell_strike >= buy_strike:
                sell_strike = buy_strike - 2.5

            setup = {
                "type": f"BEAR PUT SPREAD ({expiry})",
                "leg1": f"BUY Put Strike: ${buy_strike}",
                "leg2": f"SELL Put Strike: ${sell_strike}",
                "leg3": "Net Debit (Low Risk)",
            }

    else:
        # Neutral / Chop
        setup = {
            "type": "WAIT / CASH",
            "leg1": "No clear edge.",
            "leg2": " volatility low.",
            "leg3": "Preserve Capital.",
        }

    return setup


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

    # Logic
    signal = "NEUTRAL"
    suggestion = "Wait for Setup"

    if price < latest["BB_Lower"] and latest["RSI"] < 30:
        signal = "BULLISH"
        suggestion = "Oversold | Reversion Likely"
    elif price > latest["BB_Upper"] and latest["RSI"] > 70:
        signal = "BEARISH"
        suggestion = "Overbought | Rejection Likely"
    elif price > latest["VWAP"] and df.iloc[-2]["Close"] < df.iloc[-2]["VWAP"]:
        signal = "BULLISH (VWAP)"
        suggestion = "Momentum Breakout"

    # --- GENERATE AI SETUP ---
    ai_setup = generate_trade_setup(
        ticker, price, signal, latest["BB_Upper"], latest["BB_Lower"], latest["Std_Dev"]
    )

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "rsi": round(latest["RSI"], 2),
        "vwap": round(latest["VWAP"], 2),
        "signal": signal,
        "suggestion": suggestion,
        "setup": ai_setup,  # Pass the new setup to the UI
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

    bulls = sum(1 for r in results if "BULLISH" in r["signal"])
    bears = sum(1 for r in results if "BEARISH" in r["signal"])

    mood = "NEUTRAL"
    if bulls > bears:
        mood = "BULL"
    elif bears > bulls:
        mood = "BEAR"

    return render_template(
        "index.html",
        results=results,
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

    return render_template(
        "index.html",
        results=results,
        mood=mood,
        vix=get_vix_data(),
        status_color=get_market_status_color(),
    )


@app.route("/add_to_watchlist", methods=["POST"])
def add_to_watchlist():
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
