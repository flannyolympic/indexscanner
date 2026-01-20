import logging
import random
import sqlite3
import threading
import time as t_module
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# --- VERSION 1.8.0 DATA FOCUS ---
APP_VERSION = "v1.8.0 Pro"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HFT_Scanner")

# Robust Cache & Lock
VIX_LOCK = threading.Lock()
VIX_CACHE = {
    "data": {
        "price": "...",
        "color": "grey",
        "label": "LOADING",
        "market_status": "grey",
    },
    "last_updated": 0,
}

# UNIVERSE
STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMD",
    "AMZN",
    "GOOGL",
    "META",
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
    "NFLX",
    "INTC",
    "BA",
    "DIS",
    "JPM",
    "GS",
    "V",
    "MA",
    "WMT",
    "JNJ",
    "PG",
    "XOM",
    "CVX",
    "HD",
    "KO",
    "PEP",
    "COST",
    "AVGO",
    "ORCL",
    "IBM",
]

CRYPTO = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "DOGE-USD",
    "SHIB-USD",
    "XRP-USD",
    "ADA-USD",
    "AVAX-USD",
    "LINK-USD",
    "LTC-USD",
    "DOT-USD",
    "MATIC-USD",
    "UNI-USD",
    "ATOM-USD",
    "ETC-USD",
    "XLM-USD",
    "BCH-USD",
    "ALGO-USD",
    "VET-USD",
    "ICP-USD",
    "FIL-USD",
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


def get_market_status():
    tz = pytz.timezone("US/Eastern")
    now = datetime.now(tz)
    current_time = now.time()

    if now.weekday() >= 5:
        return "WEEKEND"
    if time(4, 0) <= current_time < time(9, 30):
        return "PRE-MARKET"
    if time(9, 30) <= current_time <= time(16, 0):
        return "OPEN"
    if time(16, 0) < current_time <= time(20, 0):
        return "AFTER-HOURS"
    return "CLOSED"


def get_market_status_color():
    status = get_market_status()
    if status == "OPEN":
        return "#00e676"
    if status in ["PRE-MARKET", "AFTER-HOURS"]:
        return "#ffd700"
    return "#ff5252"


# --- DATA FETCHING ---
def get_market_data(ticker, retries=3):
    attempt = 0
    while attempt <= retries:
        try:
            jitter = random.uniform(0.05, 0.2)
            t_module.sleep(jitter)
            # Native yfinance handling (Best for Render)
            df = yf.download(
                ticker, period="5d", interval="5m", prepost=True, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 10:
                return df
        except Exception as e:
            if attempt > 0:
                logger.warning(f"Retry {attempt}/{retries} for {ticker}: {e}")
            t_module.sleep(1 + attempt)
        attempt += 1
    return None


def get_ticker_news(ticker):
    try:
        clean_ticker = ticker.replace("-USD", "")
        search_term = (
            f"{clean_ticker} crypto" if "-USD" in ticker else f"{clean_ticker} stock"
        )
        url = f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en"

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=3)

        root = ET.fromstring(response.content)
        items = root.findall("./channel/item")

        news_data = []
        for item in items[:3]:
            title = (
                item.find("title").text
                if item.find("title") is not None
                else "Market Update"
            )
            link = item.find("link").text if item.find("link") is not None else "#"
            source = item.find("source")
            publisher = source.text if source is not None else "Financial Wire"
            news_data.append({"title": title, "publisher": publisher, "link": link})

        return news_data
    except Exception as e:
        logger.error(f"News Error: {e}")
        return []


# --- VIX ---
def get_vix_data(force_update=False):
    global VIX_CACHE
    if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60:
        return VIX_CACHE["data"]

    with VIX_LOCK:
        if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60:
            return VIX_CACHE["data"]
        try:
            status_color = get_market_status_color()
            df = get_market_data("^VIX", retries=2)
            if df is None:
                if VIX_CACHE["last_updated"] > 0:
                    return VIX_CACHE["data"]
                return {
                    "price": "...",
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


# --- AI CORE ---
class NeuralEngine:
    @staticmethod
    def calculate_technical_score(df):
        latest = df.iloc[-1]
        rsi = latest["RSI"]
        rsi_score = 90 if (rsi < 30 or rsi > 70) else (50 - abs(50 - rsi))

        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        vol_score = 50
        if not pd.isna(avg_vol) and avg_vol > 0:
            vol_score = min(100, (latest["Volume"] / avg_vol) * 50)

        trend_score = 100 if latest["Close"] > latest["SMA_20"] else 0
        return (rsi_score * 0.3) + (vol_score * 0.3) + (trend_score * 0.4)

    @staticmethod
    def generate_narrative(signal, rsi, vol_spike, price, vwap):
        dist_to_vwap = ((price - vwap) / vwap) * 100
        narrative = []
        if "BULLISH" in signal:
            narrative.append("Bullish momentum detected")
            if vol_spike:
                narrative.append("with high volume")
            if rsi < 35:
                narrative.append("rebounding from oversold")
            elif rsi > 60:
                narrative.append("confirming trend")
            if dist_to_vwap > 0:
                narrative.append(f"holding {dist_to_vwap:.2f}% above VWAP")
        elif "BEARISH" in signal:
            narrative.append("Bearish rejection detected")
            if vol_spike:
                narrative.append("on high sell volume")
            if rsi > 65:
                narrative.append("at overextended levels")
            if dist_to_vwap < 0:
                narrative.append(f"rejected {abs(dist_to_vwap):.2f}% below VWAP")
        else:
            narrative.append("Consolidation phase")
        return f"{', '.join(narrative)}."


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
        if not ticker.endswith("-USD") and ticker in CRYPTO:
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
    if pd.isna(latest["Close"]) or pd.isna(latest["RSI"]):
        return None

    price = latest["Close"]
    avg_width = df["BB_Width"].rolling(window=20).mean().iloc[-1]
    current_width = latest["BB_Width"]
    avg_vol = df["Volume"].rolling(window=20).mean().iloc[-1]
    avg_vol = avg_vol if (not pd.isna(avg_vol) and avg_vol > 0) else 1
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

    tech_score = NeuralEngine.calculate_technical_score(df)
    rationale = NeuralEngine.generate_narrative(
        signal, latest["RSI"], vol_spike, price, latest["VWAP"]
    )
    probability = round(tech_score, 1)
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
        "rationale": rationale,
    }


@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").upper()
    if not query:
        return jsonify([])

    local_suggestions = []
    ALL_TICKERS = list(set(STOCKS + CRYPTO))
    for t in ALL_TICKERS:
        if t.startswith(query):
            name = "Crypto" if "-USD" in t else "Stock"
            local_suggestions.append({"symbol": t, "name": name, "exch": "Global"})

    if local_suggestions:
        return jsonify(local_suggestions[:5])
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
        market_status=get_market_status(),
    )


@app.route("/scan")
def scan():
    start_time = t_module.time()
    mode = request.args.get("mode", "stock")
    scan_source = CRYPTO if mode == "crypto" else STOCKS

    scan_list = random.sample(scan_source, min(len(scan_source), 8))
    leader = "BTC-USD" if mode == "crypto" else "SPY"
    if leader not in scan_list:
        scan_list.append(leader)

    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in scan_list}
        for future in as_completed(future_to_ticker):
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logger.error(f"Scan Error: {e}")

    active_setups = [r for r in results if r["signal"] != "NEUTRAL"]
    chosen_one = (
        sorted(active_setups, key=lambda x: x["probability"], reverse=True)[0]
        if active_setups
        else (results[0] if results else None)
    )

    if chosen_one:
        chosen_one["news"] = get_ticker_news(chosen_one["ticker"])

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
        market_status=get_market_status(),
        current_mode=mode,
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
            market_status=get_market_status(),
        )
    result["news"] = get_ticker_news(result["ticker"])
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
        market_status=get_market_status(),
    )


@app.route("/api/vix")
def api_vix():
    return jsonify(get_vix_data())


def background_vix_updater():
    t_module.sleep(5)
    while True:
        try:
            get_vix_data(force_update=True)
        except Exception as e:
            logger.error(f"VIX BG Error: {e}")
        t_module.sleep(60)


if __name__ != "__main__":
    threading.Thread(target=background_vix_updater, daemon=True).start()

if __name__ == "__main__":
    threading.Thread(target=background_vix_updater, daemon=True).start()
    app.run(debug=True)
