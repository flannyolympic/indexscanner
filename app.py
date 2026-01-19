import logging
import random
import sqlite3
import threading
import time as t_module
from datetime import datetime, time

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from scipy.stats import norm

app = Flask(__name__)
DB_NAME = "watchlist.db"

# --- VERSION 1.0.7 CLEAN PROTOCOL ---
APP_VERSION = "v1.0.7 Clean"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HFT_Scanner")

# Robust Cache
VIX_CACHE = {
    "data": {
        "price": "WAIT",
        "color": "grey",
        "label": "INITIALIZING",
        "market_status": "grey",
    },
    "last_updated": 0,
}

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
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
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
        return "#ff5252"
    current_time = now.time()
    if time(9, 30) <= current_time <= time(16, 0):
        return "#00e676"
    return "#ff5252"


# --- ROBUST DATA FETCHING ---
def get_market_data(ticker, retries=3):
    attempt = 0
    while attempt <= retries:
        try:
            jitter = random.uniform(0.1, 0.5)
            t_module.sleep(jitter)

            df = yf.download(ticker, period="5d", interval="5m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not df.empty and len(df) >= 20:
                return df

        except Exception as e:
            if attempt > 0:
                logger.warning(f"Retry {attempt}/{retries} for {ticker}: {e}")
            t_module.sleep(2**attempt)

        attempt += 1

    return None


# --- VIX SPECTRUM LOGIC ---
def get_vix_data(force_update=False):
    global VIX_CACHE
    if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60:
        return VIX_CACHE["data"]

    try:
        status_color = get_market_status_color()
        df = get_market_data("^VIX", retries=3)

        if df is None:
            if VIX_CACHE["last_updated"] > 0:
                return VIX_CACHE["data"]
            return {
                "price": "N/A",
                "color": "grey",
                "label": "OFFLINE",
                "market_status": status_color,
            }

        price = df.iloc[-1]["Close"]

        if price < 12:
            color, label = "#00E676", "COMPLACENCY"
        elif 12 <= price < 15:
            color, label = "#66BB6A", "CALM"
        elif 15 <= price < 20:
            color, label = "#FFD600", "MILD"
        elif 20 <= price < 30:
            color, label = "#FF9100", "ELEVATED"
        elif 30 <= price < 40:
            color, label = "#FF3D00", "ANXIETY"
        elif 40 <= price < 50:
            color, label = "#D50000", "CRISIS"
        else:
            color, label = "#880E4F", "SHOCK"

        new_data = {
            "price": round(price, 2),
            "color": color,
            "label": label,
            "market_status": status_color,
        }
        VIX_CACHE["data"] = new_data
        VIX_CACHE["last_updated"] = t_module.time()
        return new_data
    except Exception:
        return VIX_CACHE["data"]


# --- CORE LOGIC ---
def calculate_probability(price, target, std_dev, rsi, trend):
    safe_vol = max(std_dev, price * 0.005)
    z_score = abs(target - price) / (safe_vol * np.sqrt(3))
    stat_prob = 2 * norm.sf(z_score) * 100

    rsi_edge = 0
    if trend == "LONG" and rsi < 40:
        rsi_edge = 12
    elif trend == "SHORT" and rsi > 60:
        rsi_edge = 12

    final_pop = stat_prob + rsi_edge
    vol_percent = (std_dev / price) * 100
    pulse = random.uniform(-min(vol_percent, 2.0), min(vol_percent, 2.0))
    return round(min(max(final_pop + pulse, 35.5), 96.2), 1)


def determine_strategy(
    price, bb_upper, bb_lower, current_width, avg_width, signal, is_crypto
):
    vol_ratio = current_width / avg_width if avg_width > 0 else 1.0
    setup_text = {"type": "WAIT", "entry": "-", "target": "-", "stop": "-"}

    if is_crypto:
        risk = current_width / 2.0
        if "BULLISH" in signal:
            setup_text = {
                "type": "CRYPTO SCALP LONG",
                "entry": f"${price:,.2f}",
                "target": f"${(price + (risk * 1.5)):,.2f}",
                "stop": f"${(price - risk):,.2f}",
            }
        elif "BEARISH" in signal:
            setup_text = {
                "type": "CRYPTO SCALP SHORT",
                "entry": f"${price:,.2f}",
                "target": f"${(price - (risk * 1.5)):,.2f}",
                "stop": f"${(price + risk):,.2f}",
            }
    else:
        if "BULLISH" in signal:
            if vol_ratio > 1.25:
                setup_text = {
                    "type": "BULL PUT SPREAD",
                    "entry": f"SELL Put ${np.floor(bb_lower)}",
                    "target": "Expire Worthless",
                    "stop": "Close < Strike",
                }
            elif vol_ratio < 0.75:
                setup_text = {
                    "type": "LONG CALL BUTTERFLY",
                    "entry": f"Center ${np.ceil(price)}",
                    "target": "Pin Strike",
                    "stop": "Wing Breach",
                }
            else:
                setup_text = {
                    "type": "BULL CALL SPREAD",
                    "entry": f"BUY Call ${np.ceil(price)}",
                    "target": f"${np.ceil(bb_upper)}",
                    "stop": "-40% Prem",
                }
        elif "BEARISH" in signal:
            if vol_ratio > 1.25:
                setup_text = {
                    "type": "BEAR CALL SPREAD",
                    "entry": f"SELL Call ${np.ceil(bb_upper)}",
                    "target": "Expire Worthless",
                    "stop": "Close > Strike",
                }
            elif vol_ratio < 0.75:
                setup_text = {
                    "type": "LONG PUT BUTTERFLY",
                    "entry": f"Center ${np.floor(price)}",
                    "target": "Pin Strike",
                    "stop": "Wing Breach",
                }
            else:
                setup_text = {
                    "type": "BEAR PUT SPREAD",
                    "entry": f"BUY Put ${np.floor(price)}",
                    "target": f"${np.floor(bb_lower)}",
                    "stop": "-40% Prem",
                }
        else:
            if vol_ratio > 1.1:
                setup_text = {
                    "type": "IRON CONDOR",
                    "entry": f"SELL Call ${np.ceil(bb_upper)}",
                    "target": "Range Bound",
                    "stop": "Wing Breach",
                }
            else:
                setup_text = {
                    "type": "CALENDAR SPREAD",
                    "entry": f"Strike ${np.round(price, 0)}",
                    "target": "Vol Expansion",
                    "stop": "Price Runaway",
                }
    return setup_text


def get_social_sentiment(rsi, vol_ratio):
    if rsi > 75 and vol_ratio > 1.1:
        return {
            "score": "MAX HYPE",
            "comment": random.choice(["🚀 GAMMA SQUEEZE", "FOMO"]),
            "icon": "🔥",
        }
    if rsi < 25 and vol_ratio > 1.1:
        return {
            "score": "MAX FEAR",
            "comment": random.choice(["🩸 LIQUIDATION", "DUMP"]),
            "icon": "🩸",
        }
    if vol_ratio > 1.1:
        return {"score": "TRENDING", "comment": "High Volume", "icon": "👀"}
    return {"score": "QUIET", "comment": "Consolidation", "icon": "💤"}


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

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Std_Dev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    delta = df["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    price = latest["Close"]
    avg_width = df["BB_Width"].rolling(window=20).mean().iloc[-1]
    current_width = latest["BB_Width"]
    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1] or 1
    vol_spike = latest["Volume"] > (avg_vol * 1.5)

    signal = "NEUTRAL"
    suggestion = "Neutral"
    trend = "FLAT"

    if price < latest["BB_Lower"] or latest["RSI"] < 30:
        signal = "BULLISH"
        suggestion = "Oversold Bounce"
        trend = "LONG"
    elif price > latest["BB_Upper"] or latest["RSI"] > 70:
        signal = "BEARISH"
        suggestion = "Overbought Reject"
        trend = "SHORT"
    elif vol_spike and price > latest["VWAP"]:
        signal = "BULLISH (VOL)"
        suggestion = "Mom. Breakout"
        trend = "LONG"

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


@app.route("/suggest")
def suggest():
    query = request.args.get("q", "")
    if not query:
        return jsonify([])
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        suggestions = []
        if "quotes" in data:
            for item in data["quotes"]:
                if "symbol" in item:
                    suggestions.append(
                        {
                            "symbol": item["symbol"],
                            "name": item.get("shortname", item.get("longname", "")),
                            "exch": item.get("exchDisp", item.get("exchange", "")),
                        }
                    )
        return jsonify(suggestions)
    except:
        return jsonify([])


@app.after_request
def add_header(response):
    if request.path.startswith("/static"):
        response.headers["Cache-Control"] = "public, max-age=31536000"
    else:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    return render_template(
        "index.html",
        vix=get_vix_data(),
        status_color=get_market_status_color(),
        timestamp=get_current_time(),
        version=APP_VERSION,
    )


@app.route("/scan")
def scan():
    start_time = t_module.time()
    scan_list = random.sample(UNIVERSE, 8)
    if "BTC-USD" not in scan_list:
        scan_list.append("BTC-USD")
    raw_results = [analyze_ticker(t) for t in scan_list]
    results = [r for r in raw_results if r]
    active_setups = [r for r in results if r["signal"] != "NEUTRAL"]
    chosen_one = (
        sorted(active_setups, key=lambda x: x["probability"], reverse=True)[0]
        if active_setups
        else (results[0] if results else None)
    )
    bulls = sum(1 for r in results if "BULLISH" in r["signal"])
    bears = sum(1 for r in results if "BEARISH" in r["signal"])
    mood = "BULL" if bulls > bears else "BEAR"
    elapsed = round(t_module.time() - start_time, 2)
    logger.info(f"Scan complete in {elapsed}s")
    return render_template(
        "index.html",
        results=results,
        chosen_one=chosen_one,
        mood=mood,
        vix=get_vix_data(),
        status_color=get_market_status_color(),
        timestamp=get_current_time(),
        version=APP_VERSION,
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
            version=APP_VERSION,
        )
    mood = (
        "BULL"
        if "BULLISH" in result["signal"]
        else "BEAR"
        if "BEARISH" in result["signal"]
        else "NEUTRAL"
    )
    return render_template(
        "index.html",
        results=[result],
        chosen_one=result,
        mood=mood,
        vix=get_vix_data(),
        status_color=get_market_status_color(),
        timestamp=get_current_time(),
        version=APP_VERSION,
    )


@app.route("/api/vix")
def api_vix():
    return jsonify(get_vix_data())


def background_vix_updater():
    """Fetches VIX data in the background to prevent startup lag/errors"""
    t_module.sleep(3)
    while True:
        try:
            get_vix_data(force_update=True)
        except Exception as e:
            logger.warning(f"VIX BG Update Failed: {e}")
        t_module.sleep(60)


if __name__ != "__main__":
    threading.Thread(target=background_vix_updater, daemon=True).start()

if __name__ == "__main__":
    threading.Thread(target=background_vix_updater, daemon=True).start()
    app.run(debug=True)
