import logging
import random
import sqlite3
import time as t_module
import threading
from datetime import datetime, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pytz
import requests
import yfinance as yf
from flask import Flask, jsonify, redirect, render_template, request, url_for
from scipy.stats import norm

app = Flask(__name__)

# --- RESTORE POINT: SONIC NEBULA v1.4.2 ---
APP_VERSION = "v1.4.2 Sonic Nebula (Origin)" 

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HFT_Scanner")

# VIX Cache (For the UI Animation)
VIX_LOCK = threading.Lock()
VIX_CACHE = {"data": {"price": 20.00, "color": "grey", "label": "LOADING", "market_status": "grey"}, "last_updated": 0}

# --- MASTER LIST (Instant Search Fix) ---
# This replaces the broken API proxy with a hardcoded list of the most popular assets.
MASTER_INDEX = [
    {"s": "AAPL", "n": "Apple", "t": "stock"}, {"s": "MSFT", "n": "Microsoft", "t": "stock"},
    {"s": "NVDA", "n": "NVIDIA", "t": "stock"}, {"s": "TSLA", "n": "Tesla", "t": "stock"},
    {"s": "GOOGL", "n": "Google", "t": "stock"}, {"s": "AMZN", "n": "Amazon", "t": "stock"},
    {"s": "META", "n": "Meta", "t": "stock"}, {"s": "AMD", "n": "AMD", "t": "stock"},
    {"s": "GME", "n": "GameStop", "t": "stock"}, {"s": "AMC", "n": "AMC", "t": "stock"},
    {"s": "PLTR", "n": "Palantir", "t": "stock"}, {"s": "COIN", "n": "Coinbase", "t": "stock"},
    {"s": "SPY", "n": "S&P 500", "t": "stock"}, {"s": "QQQ", "n": "Nasdaq", "t": "stock"},
    {"s": "IWM", "n": "Russell 2000", "t": "stock"}, {"s": "VIX", "n": "Volatility", "t": "stock"},
    {"s": "BTC-USD", "n": "Bitcoin", "t": "crypto"}, {"s": "ETH-USD", "n": "Ethereum", "t": "crypto"},
    {"s": "SOL-USD", "n": "Solana", "t": "crypto"}, {"s": "DOGE-USD", "n": "Dogecoin", "t": "crypto"},
    {"s": "SHIB-USD", "n": "Shiba Inu", "t": "crypto"}, {"s": "XRP-USD", "n": "XRP", "t": "crypto"}
]

STOCKS = [x["s"] for x in MASTER_INDEX if x["t"] == "stock"]
CRYPTO = [x["s"] for x in MASTER_INDEX if x["t"] == "crypto"]

def get_current_time():
    return datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M:%S EST")

def get_market_status_color():
    now = datetime.now(pytz.timezone("US/Eastern"))
    if now.weekday() >= 5: return "#ff5252" # Weekend
    t = now.time()
    if time(9,30) <= t <= time(16,0): return "#00e676" # Open
    return "#ffd700" # Pre/Post

# --- THE DATA PATCH ---
# This is the ONLY "new" code. It fixes the 403 Forbidden errors.
def get_market_data(ticker):
    try:
        t_module.sleep(0.1)
        # auto_adjust=True fixes the multi-column crash
        df = yf.download(ticker, period="5d", interval="5m", progress=False, auto_adjust=True)
        
        # Deduplicate columns (Safety Fix)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if len(df) > 10: return df
    except:
        pass
    return None

def get_vix_data():
    global VIX_CACHE
    if t_module.time() - VIX_CACHE["last_updated"] < 60: return VIX_CACHE["data"]
    
    with VIX_LOCK:
        try:
            df = get_market_data("^VIX")
            if df is not None:
                price = df.iloc[-1]["Close"]
                color = "#00E676" if price < 15 else "#FFD600" if price < 20 else "#D50000"
                label = "CALM" if price < 15 else "MILD" if price < 20 else "FEAR"
                VIX_CACHE["data"] = {"price": round(price, 2), "color": color, "label": label}
                VIX_CACHE["last_updated"] = t_module.time()
        except: pass
        return VIX_CACHE["data"]

# --- ORIGINAL MATH LOGIC (v1.4.2) ---
def analyze_ticker(ticker_input):
    ticker = ticker_input.upper().strip()
    df = get_market_data(ticker)
    if df is None: return None

    close = df["Close"]
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain/loss)))

    latest = df.iloc[-1]
    price = latest["Close"]
    rsi_val = rsi.iloc[-1]
    
    # Classic Signal Logic
    signal = "NEUTRAL"
    if price < lower.iloc[-1] or rsi_val < 30: signal = "BULLISH"
    elif price > upper.iloc[-1] or rsi_val > 70: signal = "BEARISH"

    # Probability Math
    z_score = abs(upper.iloc[-1] - price) / (std.iloc[-1] * 1.7)
    prob = round(norm.sf(z_score) * 200, 1) # Simple Bell Curve Mapping
    if prob > 98: prob = 98.5
    if prob < 35: prob = 35.5

    return {
        "ticker": ticker, "price": round(price, 2), "rsi": round(rsi_val, 2),
        "signal": signal, "probability": prob, 
        "setup": "SCALP" if "-USD" in ticker else "SWING"
    }

@app.route("/suggest")
def suggest():
    q = request.args.get("q", "").upper()
    if not q: return jsonify([])
    # Instant Local Search
    return jsonify([x for x in MASTER_INDEX if x["s"].startswith(q)][:5])

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
    scan_list = random.sample(source, min(len(source), 8))
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_ticker, t): t for t in scan_list}
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)

    # Sort by Probability (High to Low)
    chosen = sorted(results, key=lambda x: x["probability"], reverse=True)[0] if results else None
    
    return render_template("index.html", results=results, chosen_one=chosen, vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION, current_mode=mode)

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    res = analyze_ticker(query)
    if not res: return redirect(url_for("index"))
    return render_template("index.html", results=[res], chosen_one=res, vix=get_vix_data(), status_color=get_market_status_color(), timestamp=get_current_time(), version=APP_VERSION)

@app.route("/api/vix")
def api_vix(): return jsonify(get_vix_data())

if __name__ == "__main__":
    app.run(debug=True)