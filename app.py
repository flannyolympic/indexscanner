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
    "SMCI",
    "ARM",
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
        # Fetch enough data for reliable volatility calc
        df = yf.download(ticker, period="1mo", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 20:
            return None
        return df
    except Exception:
        return None


def calculate_probability(price, target, std_dev, rsi, trend):
    """
    Advanced POP Calculation:
    Combines Gaussian Probability with RSI Confluence.
    """
    # 1. Base Statistical Probability (Z-Score)
    days = 5  # Swing trade duration
    volatility = std_dev * np.sqrt(days)
    if volatility == 0:
        return 50.0
    z_score = abs(target - price) / volatility
    stat_prob = 2 * norm.sf(z_score) * 100

    # 2. RSI Confluence Adjustment
    # If we are betting LONG and RSI is LOW (Oversold), Probability INCREASES.
    rsi_edge = 0
    if trend == "LONG":
        if rsi < 30:
            rsi_edge = 25  # Massive edge
        elif rsi < 40:
            rsi_edge = 15
        elif rsi > 70:
            rsi_edge = -20  # Betting against momentum is hard
    elif trend == "SHORT":
        if rsi > 70:
            rsi_edge = 25
        elif rsi > 60:
            rsi_edge = 15
        elif rsi < 30:
            rsi_edge = -20

    # Combine and Clamp
    final_pop = stat_prob + rsi_edge
    return round(min(max(final_pop, 18.5), 92.4), 1)


def determine_strategy(price, bb_upper, bb_lower, bb_width, rsi, signal, is_crypto):
    """
    THE STRATEGY ENGINE:
    Decides between Debit, Credit, or Iron Condors based on Volatility Width.
    """
    setup_text = {"type": "WAIT", "entry": "-", "target": "-", "stop": "-"}
    expiry = "Next Friday"

    # Check Volatility State (Squeeze vs Expansion)
    # This is a simplified relative width check
    high_vol = bb_width > (price * 0.05)  # If bands are >5% of price wide, Vol is high

    if is_crypto:
        risk = (bb_upper - bb_lower) / 4
        if "BULLISH" in signal:
            setup_text = {
                "type": "CRYPTO SWING LONG",
                "entry": f"${price:,.2f}",
                "target": f"${(price + (risk * 2)):,.2f}",
                "stop": f"${(price - risk):,.2f}",
            }
        elif "BEARISH" in signal:
            setup_text = {
                "type": "CRYPTO SWING SHORT",
                "entry": f"${price:,.2f}",
                "target": f"${(price - (risk * 2)):,.2f}",
                "stop": f"${(price + risk):,.2f}",
            }

    else:  # STOCK OPTIONS LOGIC
        if "BULLISH" in signal:
            if high_vol:
                # High Vol? SELL PREMIUM (Bull Put Credit Spread)
                # Sell Put at Support, Buy lower Put for protection
                sell_strike = np.floor(bb_lower)
                buy_strike = sell_strike - 2.5
                setup_text = {
                    "type": f"BULL PUT SPREAD (Credit)",
                    "entry": f"SELL Put ${sell_strike}",
                    "target": "Expire Worthless",
                    "stop": f"Below ${buy_strike}",
                }
            else:
                # Low Vol? BUY PREMIUM (Bull Call Debit Spread)
                # Buy ATM Call, Sell Resistance Call
                buy_strike = np.ceil(price)
                sell_strike = np.ceil(bb_upper)
                setup_text = {
                    "type": f"BULL CALL SPREAD (Debit)",
                    "entry": f"BUY Call ${buy_strike}",
                    "target": f"SELL Call ${sell_strike}",
                    "stop": "-50% Premium",
                }

        elif "BEARISH" in signal:
            if high_vol:
                # High Vol? SELL PREMIUM (Bear Call Credit Spread)
                sell_strike = np.ceil(bb_upper)
                buy_strike = sell_strike + 2.5
                setup_text = {
                    "type": f"BEAR CALL SPREAD (Credit)",
                    "entry": f"SELL Call ${sell_strike}",
                    "target": "Expire Worthless",
                    "stop": f"Above ${buy_strike}",
                }
            else:
                # Low Vol? BUY PREMIUM (Bear Put Debit Spread)
                buy_strike = np.floor(price)
                sell_strike = np.floor(bb_lower)
                setup_text = {
                    "type": f"BEAR PUT SPREAD (Debit)",
                    "entry": f"BUY Put ${buy_strike}",
                    "target": f"SELL Put ${sell_strike}",
                    "stop": "-50% Premium",
                }

        else:  # NEUTRAL SIGNAL
            if high_vol:
                # Neutral + High Vol = IRON CONDOR (Rangebound)
                call_side = np.ceil(bb_upper)
                put_side = np.floor(bb_lower)
                setup_text = {
                    "type": "IRON CONDOR (Income)",
                    "entry": f"SELL Call ${call_side} / Put ${put_side}",
                    "target": "Range Hold",
                    "stop": "Band Breakout",
                }

    return setup_text


def get_social_sentiment(rsi, vol_spike):
    if rsi > 70 and vol_spike:
        return {
            "score": "MAX HYPE",
            "comment": random.choice(["🚀 MOON", "Short Squeeze"]),
            "icon": "🔥",
        }
    if rsi < 30 and vol_spike:
        return {
            "score": "FEAR",
            "comment": random.choice(["🩸 Capitulation", "Oversold"]),
            "icon": "🩸",
        }
    if vol_spike:
        return {"score": "ACTIVE", "comment": "Whales Accumulating", "icon": "👀"}
    return {"score": "QUIET", "comment": "Retail Sleeping", "icon": "💤"}


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
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    price = latest["Close"]
    std_dev = latest["Std_Dev"]
    bb_upper = latest["BB_Upper"]
    bb_lower = latest["BB_Lower"]
    bb_width = bb_upper - bb_lower

    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
    if pd.isna(avg_vol) or avg_vol == 0:
        avg_vol = 1
    vol_spike = latest["Volume"] > (avg_vol * 1.5)

    # Signal Logic
    signal = "NEUTRAL"
    suggestion = "Neutral"
    trend = "FLAT"

    if price < bb_lower or latest["RSI"] < 35:
        signal = "BULLISH"
        suggestion = "Oversold Reversion"
        trend = "LONG"
    elif price > bb_upper or latest["RSI"] > 65:
        signal = "BEARISH"
        suggestion = "Overbought Rejection"
        trend = "SHORT"
    elif vol_spike and price > latest["VWAP"]:
        signal = "BULLISH (VOL)"
        suggestion = "Volume Breakout"
        trend = "LONG"

    # --- ADVANCED AI CALCULATIONS ---
    # 1. Determine Strategy based on Volatility Width + Crypto vs Stock
    is_crypto = "-USD" in ticker
    setup_text = determine_strategy(
        price, bb_upper, bb_lower, bb_width, latest["RSI"], signal, is_crypto
    )

    # 2. Calculate Dynamic POP based on Trend + RSI Edge
    target_price = bb_upper if trend == "LONG" else bb_lower
    probability = calculate_probability(
        price, target_price, std_dev, latest["RSI"], trend
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

    # Pick the Highest Probability Trade
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
