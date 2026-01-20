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

# --- VERSION 1.7.0 CEREBRO ---
APP_VERSION = "v1.7.0 Cerebro"

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

# EXPANDED UNIVERSE FOR INSTANT SEARCH
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


# --- ROBUST DATA FETCHING ---
def get_market_data(ticker, retries=3):
    attempt = 0
    while attempt <= retries:
        try:
            jitter = random.uniform(0.05, 0.2)
            t_module.sleep(jitter)
            df = yf.download(
                ticker, period="5d", interval="5m", prepost=True, progress=False
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 20:
                return df
        except Exception as e:
            if attempt > 0:
                logger.warning(f"Retry {attempt}/{retries} for {ticker}: {e}")
            t_module.sleep(1.5**attempt)
        attempt += 1
    return None


def get_ticker_news(ticker):
    try:
        query = f"{ticker} stock news"
        if "-USD" in ticker:
            query = f"{ticker.replace('-USD', '')} crypto news"

        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=4)
        response.raise_for_status()

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

            publisher = "Financial Wire"
            source = item.find("source")
            if source is not None:
                publisher = source.text
            elif " - " in title:
                publisher = title.split(" - ")[-1]
                title = title.split(" - ")[0]

            news_data.append({"title": title, "publisher": publisher, "link": link})

        return news_data
    except Exception as e:
        logger.error(f"News Fetch Error ({ticker}): {e}")
        return []


# --- VIX SPECTRUM LOGIC ---
def get_vix_data(force_update=False):
    global VIX_CACHE
    if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60:
        return VIX_CACHE["data"]

    with VIX_LOCK:
        if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60:
            return VIX_CACHE["data"]

        try:
            status_color = get_market_status_color()
            df = get_market_data("^VIX", retries=3)

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


# --- THE NEURAL ENGINE (AI CORE) ---
class NeuralEngine:
    @staticmethod
    def calculate_technical_score(df):
        """Generates a composite technical score (0-100) based on weighted factors."""
        latest = df.iloc[-1]

        # 1. RSI Score (30%)
        rsi = latest["RSI"]
        rsi_score = 50 - abs(
            50 - rsi
        )  # Center is best for trending, extremes for mean rev
        if rsi < 30:
            rsi_score = 90  # Oversold bounce potential
        elif rsi > 70:
            rsi_score = 90  # Momentum breakout potential

        # 2. Volume Score (30%)
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        vol_score = min(100, (latest["Volume"] / avg_vol) * 50) if avg_vol > 0 else 50

        # 3. Trend Score (40%)
        sma20 = latest["SMA_20"]
        price = latest["Close"]
        trend_score = 100 if price > sma20 else 0

        # Composite
        final_score = (rsi_score * 0.3) + (vol_score * 0.3) + (trend_score * 0.4)
        return final_score

    @staticmethod
    def generate_narrative(signal, rsi, vol_spike, price, vwap):
        """Constructs a natural language explanation."""
        dist_to_vwap = ((price - vwap) / vwap) * 100

        narrative = []
        if "BULLISH" in signal:
            narrative.append("Bullish divergence detected")
            if vol_spike:
                narrative.append("driven by institutional volume")
            if rsi < 35:
                narrative.append("from deep oversold territory")
            elif rsi > 60:
                narrative.append("riding strong momentum")
            if dist_to_vwap > 0.5:
                narrative.append(f"holding {dist_to_vwap:.1f}% above VWAP support")
        elif "BEARISH" in signal:
            narrative.append("Bearish rejection confirmed")
            if vol_spike:
                narrative.append("on heavy selling pressure")
            if rsi > 65:
                narrative.append("at overbought extremes")
            if dist_to_vwap < -0.5:
                narrative.append(
                    f"trading {abs(dist_to_vwap):.1f}% below VWAP resistance"
                )
        else:
            narrative.append("Price is consolidating")
            narrative.append("awaiting volatility expansion")

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
        if not ticker.endswith("-USD"):
            df = get_market_data(f"{ticker}-USD")
            if df is not None:
                ticker = f"{ticker}-USD"
    if df is None:
        return None

    # Technical Calculations
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

    # Signal Logic
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

    # NEURAL ENGINE EXECUTION
    tech_score = NeuralEngine.calculate_technical_score(df)
    rationale = NeuralEngine.generate_narrative(
        signal, latest["RSI"], vol_spike, price, latest["VWAP"]
    )

    # Probability is influenced by the Neural Score
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

    # 1. INSTANT LOCAL SEARCH (Google-like Speed)
    local_suggestions = []
    # Combine lists for search
    ALL_TICKERS = list(set(STOCKS + CRYPTO))

    for t in ALL_TICKERS:
        if t.startswith(query):
            local_suggestions.append(
                {"symbol": t, "name": "Popular Asset", "exch": "Global"}
            )

    # If we found matches locally, return them INSTANTLY
    if local_suggestions:
        return jsonify(local_suggestions[:5])

    # 2. Fallback to API if no local match
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
    try:
        response = requests.get(url, headers=headers, timeout=2)
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
        market_status=get_market_status(),
    )


@app.route("/scan")
def scan():
    start_time = t_module.time()
    mode = request.args.get("mode", "stock")
    scan_source = CRYPTO if mode == "crypto" else STOCKS

    scan_list = random.sample(scan_source, 8)
    leader = "BTC-USD" if mode == "crypto" else "SPY"
    if leader not in scan_list:
        scan_list.append(leader)

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
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
