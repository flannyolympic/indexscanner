import logging
import random
import sqlite3
import time as t_module
import threading
import json
from datetime import datetime, time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# --- VERSION 2.4.0 GOOGLE INTELLIGENCE ---
APP_VERSION = "v2.4.0 Google Intelligence" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HFT_Scanner")

VIX_LOCK = threading.Lock()
VIX_CACHE = {"data": {"price": "...", "color": "grey", "label": "LOADING", "market_status": "grey"}, "last_updated": 0}

# --- ASSET LISTS ---
STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META", "GME", "AMC", "PLTR", "COIN", "MSTR", "SMCI", "ARM", "SPY", "QQQ", "IWM", "NFLX", "INTC", "BA", "DIS", "JPM", "GS", "V", "MA", "WMT", "JNJ", "PG", "XOM", "CVX", "HD", "KO", "PEP", "COST", "AVGO", "ORCL", "IBM"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "SHIB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "DOT-USD", "MATIC-USD", "UNI-USD", "ATOM-USD", "ETC-USD", "XLM-USD", "BCH-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD"]

def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY, ticker TEXT, signal TEXT, price REAL, strategy TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()
init_db()

def get_current_time():
    return datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S EST")

def get_market_status():
    now = datetime.now(pytz.timezone("US/Eastern"))
    current_time = now.time()
    if now.weekday() >= 5: return "WEEKEND"
    if time(4, 0) <= current_time < time(9, 30): return "PRE-MARKET"
    if time(9, 30) <= current_time <= time(16, 0): return "OPEN"
    if time(16, 0) < current_time <= time(20, 0): return "AFTER-HOURS"
    return "CLOSED"

def get_market_status_color():
    status = get_market_status()
    if status == "OPEN": return "#00e676"
    if status in ["PRE-MARKET", "AFTER-HOURS"]: return "#ffd700"
    return "#ff5252" 

def get_market_data(ticker, retries=3):
    attempt = 0
    while attempt <= retries:
        try:
            t_module.sleep(random.uniform(0.05, 0.2))
            df = yf.download(ticker, period="5d", interval="5m", prepost=True, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if not df.empty and len(df) >= 10: return df
        except Exception as e:
            if attempt > 0: logger.warning(f"Retry {attempt}/{retries} for {ticker}: {e}")
            t_module.sleep(1 + attempt)
        attempt += 1
    return None

def get_ticker_news(ticker):
    try:
        clean_ticker = ticker.replace("-USD", "")
        search_term = f"{clean_ticker} crypto" if "-USD" in ticker else f"{clean_ticker} stock"
        url = f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
        root = ET.fromstring(response.content)
        items = root.findall("./channel/item")
        news_data = []
        for item in items[:3]: 
            title = item.find("title").text if item.find("title") is not None else "Market Update"
            link = item.find("link").text if item.find("link") is not None else "#"
            source = item.find("source")
            publisher = source.text if source is not None else "Financial Wire"
            news_data.append({"title": title, "publisher": publisher, "link": link})
        return news_data
    except: return []

def get_vix_data(force_update=False):
    global VIX_CACHE
    if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60: return VIX_CACHE["data"]
    with VIX_LOCK:
        if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60: return VIX_CACHE["data"]
        try:
            status_color = get_market_status_color()
            df = get_market_data("^VIX", retries=2)
            if df is None: return {"price": "...", "color": "grey", "label": "OFFLINE", "market_status": status_color}
            price = df.iloc[-1]["Close"]
            if price < 12: color, label = "#00E676", "COMPLACENCY"
            elif 12 <= price < 15: color, label = "#66BB6A", "CALM"
            elif 15 <= price < 20: color, label = "#FFD600", "MILD"
            elif 20 <= price < 30: color, label = "#FF9100", "ELEVATED"
            elif 30 <= price < 40: color, label = "#FF3D00", "ANXIETY"
            elif 40 <= price < 50: color, label = "#D50000", "CRISIS"
            else: color, label = "#880E4F", "SHOCK"
            new_data = {"price": round(price, 2), "color": color, "label": label, "market_status": status_color}
            VIX_CACHE["data"] = new_data
            VIX_CACHE["last_updated"] = t_module.time()
            return new_data
        except: return VIX_CACHE["data"]

class NeuralEngine:
    @staticmethod
    def calculate_technical_score(df):
        latest = df.iloc[-1]
        rsi = latest["RSI"]
        rsi_score = 50 - abs(50 - rsi)
        if rsi < 30 or rsi > 70: rsi_score = 90
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        vol_score = 50
        if not pd.isna(avg_vol) and avg_vol > 0: vol_score = min(100, (latest["Volume"] / avg_vol) * 50)
        trend_score = 100 if latest["Close"] > latest["SMA_20"] else 0
        return round((rsi_score * 0.3) + (vol_score * 0.3) + (trend_score * 0.4), 1)

    @staticmethod
    def generate_narrative(signal, rsi, vol_spike, price, vwap):
        dist_to_vwap = ((price - vwap) / vwap) * 100
        narrative = []
        if "BULLISH" in signal:
            narrative.append("Bullish momentum detected")
            if vol_spike: narrative.append("fueled by high volume")
            if rsi < 35: narrative.append("rebounding from oversold conditions")
            elif rsi > 60: narrative.append("confirming trend strength")
            if dist_to_vwap > 0: narrative.append(f"holding {dist_to_vwap:.2f}% above VWAP")
        elif "BEARISH" in signal:
            narrative.append("Bearish rejection detected")
            if vol_spike: narrative.append("on heavy selling pressure")
            if rsi > 65: narrative.append("at overextended levels")
            if dist_to_vwap < 0: narrative.append(f"rejected {abs(dist_to_vwap):.2f}% below VWAP")
        else:
            narrative.append("Price is consolidating")
            narrative.append("awaiting a volatility trigger")
        return f"{', '.join(narrative)}."

def determine_strategy(price, bb_upper, bb_lower, current_width, avg_width, signal, is_crypto):
    vol_ratio = current_width / avg_width if avg_width > 0 else 1.0
    setup_text = {"type": "WAIT", "entry": "-", "target": "-", "stop": "-"}
    if is_crypto:
        risk = current_width / 2.0
        if "BULLISH" in signal: setup_text = {"type": "CRYPTO SCALP LONG", "entry": f"${price:,.2f}", "target": f"${(price+(risk*1.5)):,.2f}", "stop": f"${(price-risk):,.2f}"}
        elif "BEARISH" in signal: setup_text = {"type": "CRYPTO SCALP SHORT", "entry": f"${price:,.2f}", "target": f"${(price-(risk*1.5)):,.2f}", "stop": f"${(price+risk):,.2f}"}
    else:
        if "BULLISH" in signal:
            if vol_ratio > 1.25: setup_text = {"type": "BULL PUT SPREAD", "entry": f"SELL Put ${np.floor(bb_lower)}", "target": "Expire Worthless", "stop": "Close < Strike"}
            elif vol_ratio < 0.75: setup_text = {"type": "LONG CALL BUTTERFLY", "entry": f"Center ${np.ceil(price)}", "target": "Pin Strike", "stop": "Wing Breach"}
            else: setup_text = {"type": "BULL CALL SPREAD", "entry": f"BUY Call ${np.ceil(price)}", "target": f"${np.ceil(bb_upper)}", "stop": "-40% Prem"}
        elif "BEARISH" in signal:
            if vol_ratio > 1.25: setup_text = {"type": "BEAR CALL SPREAD", "entry": f"SELL Call ${np.ceil(bb_upper)}", "target": "Expire Worthless", "stop": "Close > Strike"}
            elif vol_ratio < 0.75: setup_text = {"type": "LONG PUT BUTTERFLY", "entry": f"Center ${np.floor(price)}", "target": "Pin Strike", "stop": "Wing Breach"}
            else: setup_text = {"type": "BEAR PUT SPREAD", "entry": f"BUY Put ${np.floor(price)}", "target": f"${np.floor(bb_lower)}", "stop": "-40% Prem"}
        else:
            if vol_ratio > 1.1: setup_text = {"type": "IRON CONDOR", "entry": f"SELL Call ${np.ceil(bb_upper)}", "target": "Range Bound", "stop": "Wing Breach"}
            else: setup_text = {"type": "CALENDAR SPREAD", "entry": f"Strike ${np.round(price,0)}", "target": "Vol Expansion", "stop": "Price Runaway"}
    return setup_text

def get_social_sentiment(rsi, vol_ratio):
    if rsi > 75 and vol_ratio > 1.1: return {"score": "MAX HYPE", "comment": random.choice(["🚀 GAMMA SQUEEZE", "FOMO"]), "icon": "🔥"}
    if rsi < 25 and vol_ratio > 1.1: return {"score": "MAX FEAR", "comment": random.choice(["🩸 LIQUIDATION", "DUMP"]), "icon": "🩸"}
    if vol_ratio > 1.1: return {"score": "TRENDING", "comment": "High Volume", "icon": "👀"}
    return {"score": "QUIET", "comment": "Consolidation", "icon": "💤"}

def analyze_ticker(ticker_input):
    ticker = ticker_input.upper().strip()
    df = get_market_data(ticker)
    if df is None:
        if not ticker.endswith("-USD") and ticker in CRYPTO:
             df = get_market_data(f"{ticker}-USD")
             if df is not None: ticker = f"{ticker}-USD"
        if df is None: return None

    close_prices = df["Close"]
    volume_data = df["Volume"]
    df["SMA_20"] = close_prices.rolling(window=20).mean()
    df["Std_Dev"] = close_prices.rolling(window=20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["VWAP"] = (close_prices * volume_data).cumsum() / volume_data.cumsum()

    delta = close_prices.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    if pd.isna(latest["Close"]) or pd.isna(latest["RSI"]): return None

    price, rsi_val = latest["Close"], latest["RSI"]
    vol_spike = volume_data.iloc[-1] > (volume_data.rolling(window=20).mean().iloc[-1] * 1.5)
    
    signal, suggestion, trend = "NEUTRAL", "Neutral", "FLAT"
    if price < latest["BB_Lower"] or rsi_val < 30: signal, suggestion, trend = "BULLISH", "Oversold Bounce", "LONG"
    elif price > latest["BB_Upper"] or rsi_val > 70: signal, suggestion, trend = "BEARISH", "Overbought Reject", "SHORT"
    elif vol_spike and price > latest["VWAP"]: signal, suggestion, trend = "BULLISH (VOL)", "Mom. Breakout", "LONG"

    is_crypto = "-USD" in ticker
    setup_text = determine_strategy(price, latest["BB_Upper"], latest["BB_Lower"], latest["BB_Width"], df["BB_Width"].rolling(window=20).mean().iloc[-1], signal, is_crypto)
    
    return {
        "ticker": ticker, "price": round(price, 2), "rsi": round(rsi_val, 2), "vwap": round(latest["VWAP"], 2),
        "signal": signal, "suggestion": suggestion, "probability": NeuralEngine.calculate_technical_score(df),
        "social": get_social_sentiment(rsi_val, vol_spike), "setup": setup_text,
        "rationale": NeuralEngine.generate_narrative(signal, rsi_val, vol_spike, price, latest["VWAP"])
    }

# --- GOOGLE INTELLIGENCE SEARCH PROXY ---
@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").strip()
    if not query: return jsonify([])
    
    # 1. Try Yahoo Autocomplete (High Quality, but blocked sometimes)
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=6&newsCount=0&enableFuzzyQuery=false&quotesQueryId=tss_match_phrase_query"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=2)
        if response.status_code == 200:
            data = response.json()
            suggestions = []
            if "quotes" in data:
                for item in data["quotes"]:
                    suggestions.append({"symbol": item.get("symbol"), "name": item.get("shortname", item.get("longname", item.get("symbol"))), "exch": item.get("exchDisp", "")})
            return jsonify(suggestions)
    except Exception:
        pass # Fallback silently

    # 2. Fallback: Local Intelligence (Google-like Speed)
    # Matches query against our expanded local lists if the API fails
    local_suggestions = []
    ALL_TICKERS = list(set(STOCKS + CRYPTO + ["CVNA", "EURUSD=X", "GC=F", "CL=F", "SI=F"])) # Add common futures/forex
    for t in ALL_TICKERS:
        if t.startswith(query.upper()):
            name = "Crypto Asset" if "-USD" in t else "Stock Asset"
            local_suggestions.append({"symbol": t, "name": name, "exch": "Global"})
    return jsonify(local_suggestions[:5])

@app.after_request
def add_header(response):
    if request.path.startswith('/static'): response.headers["Cache-Control"] = "public, max-age=31536000"
    else: response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route("/")
def index():
    return render_template("index.html", vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, market_status=get_market_status())

@app.route("/scan")
def scan():
    start_time = t_module.time()
    mode = request.args.get('mode', 'stock')
    
    scan_source = CRYPTO if mode == 'crypto' else STOCKS
    leader = "BTC-USD" if mode == 'crypto' else "SPY"
    scan_list = random.sample(scan_source, min(len(scan_source), 8))
    if leader not in scan_list: scan_list.append(leader)
    
    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in scan_list}
        for future in as_completed(future_to_ticker):
            try:
                data = future.result()
                if data: results.append(data)
            except Exception as e: logger.error(f"Scan Error: {e}")

    active_setups = [r for r in results if r["signal"] != "NEUTRAL"]
    chosen_one = sorted(active_setups, key=lambda x: x["probability"], reverse=True)[0] if active_setups else (results[0] if results else None)
    if chosen_one: chosen_one["news"] = get_ticker_news(chosen_one["ticker"])

    bulls = sum(1 for r in results if "BULLISH" in r["signal"])
    bears = sum(1 for r in results if "BEARISH" in r["signal"])
    mood = "BULL" if bulls > bears else "BEAR"
    
    logger.info(f"Scan complete in {round(t_module.time() - start_time, 2)}s")
    return render_template("index.html", results=results, chosen_one=chosen_one, mood=mood, vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, market_status=get_market_status(), current_mode=mode)

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query: return redirect(url_for("index"))
    result = analyze_ticker(query)
    if result is None: return render_template("index.html", results=[], error=f"Could not find '{query}'", vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, market_status=get_market_status())
    result["news"] = get_ticker_news(result["ticker"])
    return render_template("index.html", results=[result], chosen_one=result, mood=("BULL" if "BULLISH" in result["signal"] else "BEAR" if "BEARISH" in result["signal"] else "NEUTRAL"), vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, market_status=get_market_status())

@app.route("/api/vix")
def api_vix(): return jsonify(get_vix_data())

def background_vix_updater():
    t_module.sleep(5) 
    while True:
        try: get_vix_data(force_update=True)
        except: pass
        t_module.sleep(60)

if __name__ != '__main__': threading.Thread(target=background_vix_updater, daemon=True).start()
if __name__ == "__main__":
    threading.Thread(target=background_vix_updater, daemon=True).start()
    app.run(debug=True)