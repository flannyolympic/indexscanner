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
# Added High Beta names for better strategy variety
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
    "SMCI",
    "ARM",
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
        # Fetch 1 month to calculate long-term averages
        df = yf.download(ticker, period="1mo", interval="90m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 20:
            return None
        return df
    except Exception:
        return None


def calculate_probability(price, target, std_dev, rsi, trend):
    """
    Advanced POP with Volatility Floor for Weekend Math.
    """
    # Safety Floor: Assume at least 1.5% volatility per week to prevent flatline math
    safe_vol = max(std_dev, price * 0.015)

    # 5-Day Outlook
    volatility = safe_vol * np.sqrt(5)
    z_score = abs(target - price) / volatility
    stat_prob = 2 * norm.sf(z_score) * 100

    # RSI Bias
    rsi_edge = 0
    if trend == "LONG" and rsi < 40:
        rsi_edge = 15
    elif trend == "SHORT" and rsi > 60:
        rsi_edge = 15

    final_pop = stat_prob + rsi_edge

    # Clamp results to realistic trading probabilities
    return round(min(max(final_pop, 42.5), 91.0), 1)


def determine_strategy(
    price, bb_upper, bb_lower, current_width, avg_width, signal, is_crypto
):
    """
    RELATIVE STRATEGY ENGINE:
    Compares Current Band Width vs Average Band Width.
    """
    # If bands are wider than normal -> Expect Mean Reversion -> SELL PREMIUM (Credit)
    # If bands are tighter than normal -> Expect Explosion -> BUY PREMIUM (Debit)
    is_high_vol = current_width > avg_width

    setup_text = {"type": "WAIT", "entry": "-", "target": "-", "stop": "-"}

    if is_crypto:
        risk = current_width / 3
        if "BULLISH" in signal:
            setup_text = {
                "type": "CRYPTO LONG (Lev)",
                "entry": f"${price:,.2f}",
                "target": f"${(price + (risk * 2)):,.2f}",
                "stop": f"${(price - risk):,.2f}",
            }
        elif "BEARISH" in signal:
            setup_text = {
                "type": "CRYPTO SHORT (Lev)",
                "entry": f"${price:,.2f}",
                "target": f"${(price - (risk * 2)):,.2f}",
                "stop": f"${(price + risk):,.2f}",
            }

    else:  # STOCK OPTIONS
        if "BULLISH" in signal:
            if is_high_vol:
                # Wide Bands -> Bull Put Spread (Credit)
                sell_strike = np.floor(bb_lower)
                buy_strike = sell_strike - 2.5
                setup_text = {
                    "type": "BULL PUT CREDIT SPREAD",
                    "entry": f"SELL Put ${sell_strike}",
                    "target": "Expire Worthless",
                    "stop": f"Close below ${buy_strike}",
                }
            else:
                # Tight Bands -> Bull Call Spread (Debit)
                buy_strike = np.ceil(price)
                sell_strike = np.ceil(bb_upper)
                setup_text = {
                    "type": "BULL CALL DEBIT SPREAD",
                    "entry": f"BUY Call ${buy_strike}",
                    "target": f"Target ${sell_strike}",
                    "stop": "-40% Premium",
                }

        elif "BEARISH" in signal:
            if is_high_vol:
                # Wide Bands -> Bear Call Spread (Credit)
                sell_strike = np.ceil(bb_upper)
                buy_strike = sell_strike + 2.5
                setup_text = {
                    "type": "BEAR CALL CREDIT SPREAD",
                    "entry": f"SELL Call ${sell_strike}",
                    "target": "Expire Worthless",
                    "stop": f"Close above ${buy_strike}",
                }
            else:
                # Tight Bands -> Bear Put Spread (Debit)
                buy_strike = np.floor(price)
                sell_strike = np.floor(bb_lower)
                setup_text = {
                    "type": "BEAR PUT DEBIT SPREAD",
                    "entry": f"BUY Put ${buy_strike}",
                    "target": f"Target ${sell_strike}",
                    "stop": "-40% Premium",
                }

        else:  # NEUTRAL SIGNAL
            if is_high_vol:
                # High Vol + Neutral = IRON CONDOR
                call_side = np.ceil(bb_upper)
                put_side = np.floor(bb_lower)
                setup_text = {
                    "type": "IRON CONDOR (Income)",
                    "entry": f"SELL Call ${call_side} / Put ${put_side}",
                    "target": "Theta Decay",
                    "stop": "Band Breach",
                }
            else:
                # Low Vol + Neutral = CALENDAR / WAIT
                setup_text = {
                    "type": "CASH / WAIT",
                    "entry": "Volatility Low",
                    "target": "Wait for squeeze",
                    "stop": "-",
                }

    return setup_text


def get_social_sentiment(rsi, vol_spike):
    if rsi > 75 and vol_spike:
        return {
            "score": "MAX HYPE",
            "comment": random.choice(["🚀 MOON MISSION", "Short Squeeze"]),
            "icon": "🔥",
        }
    if rsi < 25 and vol_spike:
        return {
            "score": "MAX FEAR",
            "comment": random.choice(["🩸 Capitulation", "Oversold"]),
            "icon": "🩸",
        }
    if vol_spike:
        return {"score": "TRENDING", "comment": "Whales Active", "icon": "👀"}
    return {"score": "QUIET", "comment": "Accumulation", "icon": "💤"}


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
            df = get_market_data(f"{ticker}-USD")
            if df is not None:
                ticker = f"{ticker}-USD"
    if df is None:
        return None

    # Technicals
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]  # Band Width
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    price = latest["Close"]

    # Volume Logic
    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1] or 1
    vol_spike = latest["Volume"] > (avg_vol * 1.3)  # 30% above average

    # Relative Volatility Logic (Current Width vs 20-period Avg Width)
    avg_width = df["BB_Width"].rolling(window=20).mean().iloc[-1]
    current_width = latest["BB_Width"]

    # Signal Logic
    signal = "NEUTRAL"
    suggestion = "Neutral"
    trend = "FLAT"

    if price < latest["BB_Lower"] or latest["RSI"] < 30:
        signal = "BULLISH"
        suggestion = "Oversold Reversion"
        trend = "LONG"
    elif price > latest["BB_Upper"] or latest["RSI"] > 70:
        signal = "BEARISH"
        suggestion = "Overbought Rejection"
        trend = "SHORT"
    elif vol_spike and price > latest["VWAP"]:
        signal = "BULLISH (VOL)"
        suggestion = "Momentum Breakout"
        trend = "LONG"

    # Strategy & POP
    is_crypto = "-USD" in ticker
    setup_text = determine_strategy(
        price,
        latest["BB_Upper"],
        latest["BB_Lower"],
        current_width,
        avg_width,
        signal,
        is_crypto,
    )

    target_price = latest["BB_Upper"] if trend == "LONG" else latest["BB_Lower"]
    probability = calculate_probability(
        price, target_price, latest["Std_Dev"], latest["RSI"], trend
    )
    social = get_social_sentiment(latest["RSI"], vol_spike)

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
        # Prioritize High Probability + Active Signal
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
            error=f"Could not find '{query}'",
            vix=get_vix_data(),
            status_color=get_market_status_color(),
            timestamp=get_current_time(),
        )

    mood = "NEUTRAL"
    if "BULLISH" in result["signal"]:
        mood = "BULL"
    elif "BEARISH" in result["signal"]:
        mood = "BEAR"

    return render_template(
        "index.html",
        results=[result],
        chosen_one=result,
        mood=mood,
        vix=get_vix_data(),
        status_color=get_market_status_color(),
        timestamp=get_current_time(),
    )


if __name__ == "__main__":
    app.run(debug=True)
