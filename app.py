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

# --- VERSION 2.5.0 SENTIENT ---
APP_VERSION = "v2.5.0 Sentient"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HFT_Scanner")

VIX_LOCK = threading.Lock()
VIX_CACHE = {"data": {"price": "...", "color": "grey", "label": "LOADING", "market_status": "grey"}, "last_updated": 0}

# --- THE "GOOGLE-LIKE" GLOBAL INDEX (Local Cache for 0ms Search) ---
# This ensures common searches NEVER fail, even if the API is blocked.
GLOBAL_INDEX = [
    # TECH GIANTS
    {"s": "AAPL", "n": "Apple Inc.", "e": "Stock"}, {"s": "MSFT", "n": "Microsoft Corp.", "e": "Stock"},
    {"s": "NVDA", "n": "NVIDIA Corp.", "e": "Stock"}, {"s": "TSLA", "n": "Tesla Inc.", "e": "Stock"},
    {"s": "GOOGL", "n": "Alphabet Inc.", "e": "Stock"}, {"s": "AMZN", "n": "Amazon.com", "e": "Stock"},
    {"s": "META", "n": "Meta Platforms", "e": "Stock"}, {"s": "AMD", "n": "Advanced Micro Devices", "e": "Stock"},
    # POPULAR / MEME
    {"s": "GME", "n": "GameStop Corp.", "e": "Stock"}, {"s": "AMC", "n": "AMC Entertainment", "e": "Stock"},
    {"s": "PLTR", "n": "Palantir Tech", "e": "Stock"}, {"s": "COIN", "n": "Coinbase Global", "e": "Stock"},
    {"s": "CVNA", "n": "Carvana Co.", "e": "Stock"}, {"s": "MSTR", "n": "MicroStrategy", "e": "Stock"},
    # CRYPTO
    {"s": "BTC-USD", "n": "Bitcoin", "e": "Crypto"}, {"s": "ETH-USD", "n": "Ethereum", "e": "Crypto"},
    {"s": "SOL-USD", "n": "Solana", "e": "Crypto"}, {"s": "DOGE-USD", "n": "Dogecoin", "e": "Crypto"},
    {"s": "SHIB-USD", "n": "Shiba Inu", "e": "Crypto"}, {"s": "XRP-USD", "n": "XRP", "e": "Crypto"},
    {"s": "PEPE-USD", "n": "Pepe", "e": "Crypto"},
    # INDICES & FUTURES (The "Smart" Stuff)
    {"s": "^GSPC", "n": "S&P 500", "e": "Index"}, {"s": "^IXIC", "n": "Nasdaq Composite", "e": "Index"},
    {"s": "^VIX", "n": "Volatility Index", "e": "Index"}, {"s": "GC=F", "n": "Gold Futures", "e": "Future"},
    {"s": "CL=F", "n": "Crude Oil", "e": "Future"}, {"s": "EURUSD=X", "n": "EUR/USD", "e": "Forex"}
]

# Simple lists for the random scanner
STOCKS = [x["s"] for x in GLOBAL_INDEX if x["e"] == "Stock"]
CRYPTO = [x["s"] for x in GLOBAL_INDEX if x["e"] == "Crypto"]

def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY, ticker TEXT, signal TEXT, price REAL, strategy TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()
init_db()

# --- HELPERS ---
def get_current_time(): return datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S EST")

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

# --- ROBUST DATA FETCHING ---
def get_market_data(ticker, retries=2):
    attempt = 0
    while attempt <= retries:
        try:
            # Native yfinance with auto_adjust to prevent multi-column errors
            df = yf.download(ticker, period="5d", interval="5m", prepost=True, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()] # Safety deduplication
            if not df.empty and len(df) >= 10: return df
        except Exception:
            t_module.sleep(0.5)
        attempt += 1
    return None

def get_ticker_news(ticker):
    try:
        clean_ticker = ticker.replace("-USD", "")
        search_term = f"{clean_ticker} crypto" if "-USD" in ticker else f"{clean_ticker} stock"
        url = f"https://news.google.com/rss/search?q={search_term}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=2)
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
            df = get_market_data("^VIX", retries=1)
            if df is None: return {"price": "...", "color": "grey", "label": "OFFLINE", "market_status": status_color}
            price = df.iloc[-1]["Close"]
            if price < 12: color, label = "#00E676", "COMPLACENCY"
            elif 12 <= price < 15: color, label = "#66BB6A", "CALM"
            elif 15 <= price < 20: color, label = "#FFD600", "MILD"
            elif 20 <= price < 30: color, label = "#FF9100", "ELEVATED"
            elif 30 <= price < 40: color, label = "#FF3D00", "ANXIETY"
            else: color, label = "#880E4F", "SHOCK"
            new_data = {"price": round(price, 2), "color": color, "label": label, "market_status": status_color}
            VIX_CACHE["data"] = new_data
            VIX_CACHE["last_updated"] = t_module.time()
            return new_data
        except: return VIX_CACHE["data"]

# --- TRADER BRAIN (AI ENGINE) ---
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
            narrative.append("Bullish divergence")
            if vol_spike: narrative.append("on high volume")
            if rsi < 35: narrative.append("from oversold")
            elif rsi > 60: narrative.append("with strong momentum")
            if dist_to_vwap > 0: narrative.append(f"holding {dist_to_vwap:.1f}% > VWAP")
        elif "BEARISH" in signal:
            narrative.append("Bearish rejection")
            if vol_spike: narrative.append("on heavy selling")
            if rsi > 65: narrative.append("at overbought")
            if dist_to_vwap < 0: narrative.append(f"trading {abs(dist_to_vwap):.1f}% < VWAP")
        else:
            narrative.append("Consolidation awaiting vol expansion")
        return f"{', '.join(narrative)}."

def determine_strategy(price, bb_upper, bb_lower, current_width, avg_width, signal, is_crypto):
    vol_ratio = current_width / avg_width if avg_width > 0 else 1.0
    setup_text = {"type": "WAIT", "entry": "-", "target": "-", "stop": "-"}
    if is_crypto:
        risk = current_width / 2.0
        if "BULLISH" in signal: setup_text = {"type": "SCALP LONG", "entry": f"${price:,.2f}", "target": f"${(price+(risk*1.5)):,.2f}", "stop": f"${(price-risk):,.2f}"}
        elif "BEARISH" in signal: setup_text = {"type": "SCALP SHORT", "entry": f"${price:,.2f}", "target": f"${(price-(risk*1.5)):,.2f}", "stop": f"${(price+risk):,.2f}"}
    else:
        if "BULLISH" in signal:
            if vol_ratio > 1.25: setup_text = {"type": "BULL PUT SPREAD", "entry": f"Sell Put ${int(bb_lower)}", "target": "Expire", "stop": "Close < Strike"}
            else: setup_text = {"type": "CALL DEBIT SPREAD", "entry": f"Buy Call ${int(price)}", "target": f"${int(bb_upper)}", "stop": "-40% Prem"}
        elif "BEARISH" in signal:
            if vol_ratio > 1.25: setup_text = {"type": "BEAR CALL SPREAD", "entry": f"Sell Call ${int(bb_upper)}", "target": "Expire", "stop": "Close > Strike"}
            else: setup_text = {"type": "PUT DEBIT SPREAD", "entry": f"Buy Put ${int(price)}", "target": f"${int(bb_lower)}", "stop": "-40% Prem"}
    return setup_text

def get_social_sentiment(rsi, vol_ratio):
    if rsi > 75 and vol_ratio > 1.1: return {"score": "MAX HYPE", "comment": "FOMO Squeeze", "icon": "🔥"}
    if rsi < 25 and vol_ratio > 1.1: return {"score": "MAX FEAR", "comment": "Capitulation", "icon": "🩸"}
    if vol_ratio > 1.1: return {"score": "ACTIVE", "comment": "High Volume", "icon": "👀"}
    return {"score": "QUIET", "comment": "Choppy", "icon": "💤"}

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
    
    signal, suggestion = "NEUTRAL", "Neutral"
    if price < latest["BB_Lower"] or rsi_val < 30: signal, suggestion = "BULLISH", "Oversold Bounce"
    elif price > latest["BB_Upper"] or rsi_val > 70: signal, suggestion = "BEARISH", "Overbought Reject"
    elif vol_spike and price > latest["VWAP"]: signal, suggestion = "BULLISH (VOL)", "Mom. Breakout"

    is_crypto = "-USD" in ticker
    setup_text = determine_strategy(price, latest["BB_Upper"], latest["BB_Lower"], latest["BB_Width"], df["BB_Width"].rolling(window=20).mean().iloc[-1], signal, is_crypto)
    
    return {
        "ticker": ticker, "price": round(price, 2), "rsi": round(rsi_val, 2), "vwap": round(latest["VWAP"], 2),
        "signal": signal, "suggestion": suggestion, "probability": NeuralEngine.calculate_technical_score(df),
        "social": get_social_sentiment(rsi_val, vol_spike), "setup": setup_text,
        "rationale": NeuralEngine.generate_narrative(signal, rsi_val, vol_spike, price, latest["VWAP"])
    }

# --- HYBRID SEARCH ENGINE (LOCAL FIRST + FALLBACK) ---
@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").strip().upper()
    if not query: return jsonify([])
    
    # 1. INSTANT LOCAL MATCH (0ms Latency)
    matches = []
    for item in GLOBAL_INDEX:
        if item["s"].startswith(query) or query in item["n"].upper():
            matches.append({"symbol": item["s"], "name": item["n"], "exch": item["e"]})
    
    # 2. IF LOCAL FAILS, TRY YAHOO PROXY
    if not matches and len(query) > 1:
        try:
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&lang=en-US&region=US&quotesCount=5"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                for q in data.get("quotes", []):
                    matches.append({"symbol": q["symbol"], "name": q.get("shortname", q["symbol"]), "exch": q.get("exchDisp", "Global")})
        except: pass

    return jsonify(matches[:6])

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
    
    # CRITICAL FIX: FORCE LIST SEPARATION
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
            except: pass

    active = [r for r in results if r["signal"] != "NEUTRAL"]
    chosen = sorted(active, key=lambda x: x["probability"], reverse=True)[0] if active else (results[0] if results else None)
    if chosen: chosen["news"] = get_ticker_news(chosen["ticker"])

    mood = "BULL" if sum(1 for r in results if "BULLISH" in r["signal"]) > sum(1 for r in results if "BEARISH" in r["signal"]) else "BEAR"
    return render_template("index.html", results=results, chosen_one=chosen, mood=mood, vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, market_status=get_market_status(), current_mode=mode)

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query: return redirect(url_for("index"))
    result = analyze_ticker(query)
    if result is None: return render_template("index.html", results=[], error=f"Asset '{query}' not found.", vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, market_status=get_market_status())
    result["news"] = get_ticker_news(result["ticker"])
    return render_template("index.html", results=[result], chosen_one=result, mood="NEUTRAL", vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, market_status=get_market_status())

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