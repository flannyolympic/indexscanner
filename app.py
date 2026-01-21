import logging
import random
import sqlite3
import time as t_module
import threading
import xml.etree.ElementTree as ET
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

# --- VERSION 1.6.2 STABLE RESTORE ---
APP_VERSION = "v1.6.2 Stable" 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HFT_Scanner")

VIX_LOCK = threading.Lock()
VIX_CACHE = {"data": {"price": "...", "color": "grey", "label": "LOADING", "market_status": "grey"}, "last_updated": 0}

# --- CLASSIC LISTS ---
STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "GOOGL", "META", "GME", "AMC", "PLTR", "COIN", "MSTR", "SMCI", "ARM", "SPY", "QQQ", "IWM", "NFLX", "INTC", "BA", "DIS", "JPM", "GS", "V", "MA", "WMT", "JNJ"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "SHIB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "DOT-USD", "MATIC-USD", "UNI-USD", "ATOM-USD", "ETC-USD", "XLM-USD", "BCH-USD", "ALGO-USD"]

def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY, ticker TEXT, signal TEXT, price REAL, strategy TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()
init_db()

def get_current_time(): return datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S EST")

def get_market_status():
    now = datetime.now(pytz.timezone("US/Eastern"))
    if now.weekday() >= 5: return "WEEKEND"
    t = now.time()
    if time(4,0) <= t < time(9,30): return "PRE-MARKET"
    if time(9,30) <= t <= time(16,0): return "OPEN"
    if time(16,0) < t <= time(20,0): return "AFTER-HOURS"
    return "CLOSED"

def get_market_status_color():
    status = get_market_status()
    if status == "OPEN": return "#00e676"
    if status in ["PRE-MARKET", "AFTER-HOURS"]: return "#ffd700"
    return "#ff5252" 

# --- NATIVE DATA FETCH (STABLE) ---
def get_market_data(ticker, retries=3):
    attempt = 0
    while attempt <= retries:
        try:
            t_module.sleep(random.uniform(0.05, 0.2))
            # Native fetch, no complex headers
            df = yf.download(ticker, period="5d", interval="5m", prepost=True, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()] # Prevent crash
            if not df.empty and len(df) >= 10: return df
        except:
            t_module.sleep(1)
        attempt += 1
    return None

def get_ticker_news(ticker):
    try:
        clean = ticker.replace("-USD", "")
        url = f"https://news.google.com/rss/search?q={clean}+stock&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=2)
        root = ET.fromstring(r.content)
        items = root.findall("./channel/item")[:3]
        return [{"title": i.find("title").text, "publisher": i.find("source").text, "link": i.find("link").text} for i in items]
    except: return []

def get_vix_data(force_update=False):
    global VIX_CACHE
    if not force_update and t_module.time() - VIX_CACHE["last_updated"] < 60: return VIX_CACHE["data"]
    with VIX_LOCK:
        try:
            df = get_market_data("^VIX", retries=1)
            if df is None: return VIX_CACHE["data"]
            price = df.iloc[-1]["Close"]
            color, label = ("#00E676", "COMPLACENCY") if price < 12 else ("#66BB6A", "CALM") if price < 15 else ("#FFD600", "MILD") if price < 20 else ("#FF9100", "ELEVATED") if price < 30 else ("#D50000", "CRISIS")
            VIX_CACHE["data"] = {"price": round(price, 2), "color": color, "label": label, "market_status": get_market_status_color()}
            VIX_CACHE["last_updated"] = t_module.time()
            return VIX_CACHE["data"]
        except: return VIX_CACHE["data"]

# --- CLASSIC MATH LOGIC (No Neural) ---
def calculate_probability(price, target, std_dev, rsi, trend):
    safe_vol = max(std_dev, price * 0.005)
    z_score = abs(target - price) / (safe_vol * np.sqrt(3))
    stat_prob = 2 * norm.sf(z_score) * 100
    rsi_edge = 12 if (trend == "LONG" and rsi < 40) or (trend == "SHORT" and rsi > 60) else 0
    return round(min(max(stat_prob + rsi_edge, 35.5), 96.2), 1)

def determine_strategy(price, bb_upper, bb_lower, width, avg_width, signal, is_crypto):
    vol_ratio = width / avg_width if avg_width > 0 else 1.0
    if is_crypto:
        risk = width / 2.0
        if "BULLISH" in signal: return {"type": "SCALP LONG", "entry": f"${price:,.2f}", "target": f"${(price+risk*1.5):,.2f}", "stop": f"${(price-risk):,.2f}"}
        if "BEARISH" in signal: return {"type": "SCALP SHORT", "entry": f"${price:,.2f}", "target": f"${(price-risk*1.5):,.2f}", "stop": f"${(price+risk):,.2f}"}
    else:
        if "BULLISH" in signal: return {"type": "BULL PUT SPREAD", "entry": f"Sell Put ${int(bb_lower)}", "target": "Expire", "stop": "Close < Strike"}
        if "BEARISH" in signal: return {"type": "BEAR CALL SPREAD", "entry": f"Sell Call ${int(bb_upper)}", "target": "Expire", "stop": "Close > Strike"}
    return {"type": "WAIT", "entry": "-", "target": "-", "stop": "-"}

def analyze_ticker(ticker_input):
    ticker = ticker_input.upper().strip()
    df = get_market_data(ticker)
    if df is None:
        if ticker in CRYPTO and not ticker.endswith("-USD"): df = get_market_data(f"{ticker}-USD")
        if df is None: return None

    # Classic Technicals
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Std_Dev"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + (df["Std_Dev"] * 2)
    df["BB_Lower"] = df["SMA_20"] - (df["Std_Dev"] * 2)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain/loss)))

    latest = df.iloc[-1]
    if pd.isna(latest["Close"]) or pd.isna(latest["RSI"]): return None

    price, rsi = latest["Close"], latest["RSI"]
    signal, trend = "NEUTRAL", "FLAT"
    
    if price < latest["BB_Lower"] or rsi < 30: signal, trend = "BULLISH", "LONG"
    elif price > latest["BB_Upper"] or rsi > 70: signal, trend = "BEARISH", "SHORT"

    setup = determine_strategy(price, latest["BB_Upper"], latest["BB_Lower"], latest["BB_Width"], df["BB_Width"].mean(), signal, "-USD" in ticker)
    target = latest["BB_Upper"] if trend == "LONG" else latest["BB_Lower"]
    prob = calculate_probability(price, target, latest["Std_Dev"], rsi, trend)
    
    # Simple Sentiment (No AI)
    social = {"score": "HYPE", "icon": "🔥"} if rsi > 70 else {"score": "FEAR", "icon": "🩸"} if rsi < 30 else {"score": "QUIET", "icon": "💤"}

    return {
        "ticker": ticker, "price": round(price, 2), "rsi": round(rsi, 2), "vwap": 0, # VWAP removed for simplicity
        "signal": signal, "probability": prob, "social": social, "setup": setup
    }

@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").upper()
    if not query: return jsonify([])
    # Simple Local Suggestion (Fast & Reliable)
    all_assets = sorted(list(set(STOCKS + CRYPTO)))
    return jsonify([{"symbol": t, "name": "Crypto" if "-USD" in t else "Stock", "exch": "US"} for t in all_assets if t.startswith(query)][:5])

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return r

@app.route("/")
def index():
    return render_template("index.html", vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION)

@app.route("/scan")
def scan():
    mode = request.args.get('mode', 'stock')
    source = CRYPTO if mode == 'crypto' else STOCKS
    # Exact Logic from v1.6.2
    scan_list = random.sample(source, min(len(source), 8))
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_ticker, t): t for t in scan_list}
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)

    active = [r for r in results if r["signal"] != "NEUTRAL"]
    chosen = sorted(active, key=lambda x: x["probability"], reverse=True)[0] if active else (results[0] if results else None)
    if chosen: chosen["news"] = get_ticker_news(chosen["ticker"])
    
    mood = "BULL" if sum(1 for r in results if "BULLISH" in r["signal"]) > sum(1 for r in results if "BEARISH" in r["signal"]) else "BEAR"
    return render_template("index.html", results=results, chosen_one=chosen, mood=mood, vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, current_mode=mode)

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    res = analyze_ticker(query)
    if not res: return redirect(url_for("index"))
    res["news"] = get_ticker_news(res["ticker"])
    return render_template("index.html", results=[res], chosen_one=res, mood="NEUTRAL", vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION)

@app.route("/api/vix")
def api_vix(): return jsonify(get_vix_data())

if __name__ == "__main__":
    app.run(debug=True)