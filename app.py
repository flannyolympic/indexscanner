import os
import time
import requests
import pandas as pd
import yfinance as yf
import pytz
import random
import traceback
from flask import Flask, render_template, request, send_from_directory
from datetime import datetime, timedelta, time as dt_time
import google.generativeai as genai
from google.api_core import exceptions

app = Flask(__name__)

# --- CONFIGURATION ---
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
    except Exception as e:
        print(f"DEBUG: Failed to configure GenAI: {e}")

# ASSET UNIVERSE (Updated List)
STOCK_TICKERS = [
    "QQQ", "SPY", "MARA", "USAR", "NFLX", "TMC", "RIOT", "GME", "LMND", "NVDA", "AMC",
    "TSLA", "AAPL", "AMZN", "DAL"
]

# --- STATIC FILE ROUTES (The Missing Piece!) ---
@app.route('/apple-touch-icon.png')
def serve_icon():
    return send_from_directory('static', 'apple-touch-icon.png')

@app.route('/favicon.ico')
def serve_favicon():
    return send_from_directory('static', 'apple-touch-icon.png')

# --- HELPER FUNCTIONS ---

def is_valid_data(data):
    """Checks if the ticker data is safe to send to the AI."""
    if not data: return False
    required = ['ticker', 'price', 'signal']
    if not all(key in data for key in required): return False
    if data['price'] == 0 or not data['ticker']: return False
    return True

def get_market_status():
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    current_time = now.time()
    
    if now.weekday() >= 5: return {"label": "WEEKEND CLOSED", "color": "#ff4b4b"}

    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    pre_market = dt_time(4, 0)
    after_hours = dt_time(20, 0)

    if market_open <= current_time < market_close:
        return {"label": "MARKET OPEN", "color": "#00ff9d"}
    elif pre_market <= current_time < market_open:
        return {"label": "PRE-MARKET", "color": "#F1C40F"}
    elif market_close <= current_time < after_hours:
        return {"label": "AFTER HOURS", "color": "#F1C40F"}
        
    return {"label": "MARKET CLOSED", "color": "#ff4b4b"}

def get_economic_events():
    """Generates realistic upcoming high-impact economic events."""
    today = datetime.now()
    events = [
        {"event": "FOMC Rate Decision", "delta": 2, "impact": "HIGH"},
        {"event": "CPI Inflation Data", "delta": 4, "impact": "HIGH"},
        {"event": "Non-Farm Payrolls", "delta": 6, "impact": "MED"},
        {"event": "GDP Growth Rate", "delta": 9, "impact": "HIGH"}
    ]
    formatted_events = []
    for e in events:
        date = today + timedelta(days=e['delta'])
        formatted_events.append({
            "title": e['event'],
            "date": date.strftime("%b %d"),
            "impact": e['impact']
        })
    return formatted_events

def get_indices():
    indices = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "DOW J"}
    data_list = []
    try:
        tickers = " ".join(indices.keys())
        data = yf.download(tickers, period="1d", group_by='ticker', threads=False, prepost=True)
        for ticker, name in indices.items():
            try:
                df = data[ticker] if len(indices) > 1 else data
                if df.empty: continue
                df = df.ffill().bfill()
                current = df['Close'].iloc[-1]
                open_price = df['Open'].iloc[0]
                change = ((current - open_price) / open_price) * 100
                sign = "+" if change >= 0 else ""
                data_list.append(f"{name}: {int(current)} ({sign}{round(change, 2)}%)")
            except: continue
    except: return ["DATA STREAM OFFLINE"]
    return data_list

def get_vix_data():
    try:
        vix = yf.Ticker("^VIX")
        try: price = vix.fast_info.last_price
        except: 
            hist = vix.history(period="1d")
            price = hist['Close'].iloc[-1] if not hist.empty else 0
        
        val = round(price, 2)
        if val < 12: return {"val": val, "label": "COMPLACENCY", "color": "#00E5FF"}
        elif val < 15: return {"val": val, "label": "CALM", "color": "#2ECC71"}
        elif val < 20: return {"val": val, "label": "MILD CONCERN", "color": "#F1C40F"}
        elif val < 30: return {"val": val, "label": "ELEVATED FEAR", "color": "#E67E22"}
        else: return {"val": val, "label": "EXTREME FEAR", "color": "#8B0000"}
    except: return {"val": 0, "label": "OFFLINE", "color": "#555"}

# --- CORE ANALYSIS ENGINE ---
def analyze_market_data(ticker_list):
    results = []
    try:
        data = yf.download(tickers=" ".join(ticker_list), period="5d", interval="15m", 
                           group_by='ticker', auto_adjust=True, prepost=True, threads=False)
    except: return []

    for ticker in ticker_list:
        try:
            if len(ticker_list) > 1: df = data.get(ticker)
            else: df = data
            
            if df is None or df.empty: continue
            df = df.ffill().bfill()
            if len(df) < 20: continue

            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            current_price = df['Close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_vwap = df['VWAP'].iloc[-1]
            current_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
            
            if pd.isna(current_rsi): current_rsi = 50
            if pd.isna(current_vwap): current_vwap = current_price

            signal, prob, catalyst, sentiment, options_play = "NEUTRAL", 50, "None", "Neutral", "Iron Condor"

            if current_rsi < 30:
                signal, prob, catalyst, sentiment, options_play = ("BULLISH (OVERSOLD)", 78, "RSI Exhaustion", "Fear (Buy Dip)", "Bull Call Spread")
            elif current_rsi > 70:
                signal, prob, catalyst, sentiment, options_play = ("BEARISH (OVERBOUGHT)", 72, "RSI Extension", "Greed (Take Profit)", "Bear Put Spread")
            elif current_price > current_vwap * 1.01:
                signal, prob, catalyst, sentiment, options_play = ("BULLISH (MOMENTUM)", 65, "VWAP Breakout", "Bullish Flow", "Long Call (ATM)")
            elif current_price < current_vwap * 0.99:
                signal, prob, catalyst, sentiment, options_play = ("BEARISH (MOMENTUM)", 65, "VWAP Breakdown", "Bearish Flow", "Long Put (ATM)")
            
            if current_vol > avg_vol * 1.5: prob += 5; catalyst += " + High Vol"

            results.append({
                "ticker": ticker, "price": round(float(current_price), 2),
                "signal": signal, "probability": prob, "vwap": round(float(current_vwap), 2),
                "catalyst": catalyst, "sentiment": sentiment, "options_play": options_play
            })
        except: continue
    return sorted(results, key=lambda x: x['probability'], reverse=True)

# --- AI NEURAL ENGINE (Fixed Model Names) ---
def get_ai_rationale(ticker_data):
    if not GENAI_API_KEY: return "AI Offline (Key Missing)."
    
    # Updated to stable model names (removed '-latest' which caused 404s)
    models = ['gemini-1.5-flash', 'gemini-1.5-pro']
    
    prompt = (
        f"Act as a senior derivatives trader. Analyze {ticker_data['ticker']} currently trading at ${ticker_data['price']}. "
        f"Technical Signal: {ticker_data['signal']}. "
        f"Task 1: Identify the likely CATALYST (e.g., Earnings, Sector rotation, Macro data). "
        f"Task 2: Suggest a specific Options Trade with Strike Prices. "
        f"Format strictly as: CATALYST: [Short Reason] | TRADE: [Buy/Sell specific Strikes] | WHY: [1 sentence rationale]."
    )
    
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
    ]

    for m in models:
        try:
            print(f"DEBUG: querying {m}...")
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt, safety_settings=safety_settings)
            if response.text: return response.text.strip()
        except exceptions.ResourceExhausted:
            print(f"⚠️ Rate Limit on {m}. Switching...")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"❌ Model {m} failed: {e}")
            continue
            
    return "AI Unavailable. Trade based on Technicals."

# --- FLASK ROUTES ---

@app.route('/')
def home():
    return render_template(
        'index.html', 
        market_status=get_market_status(), 
        vix=get_vix_data(), 
        indices=get_indices(),
        econ_events=get_economic_events()
    )

@app.route('/scan')
def scan():
    try:
        results = analyze_market_data(STOCK_TICKERS)
        chosen_one = None
        if results:
            chosen_one = results[0]
            if is_valid_data(chosen_one):
                chosen_one['rationale'] = get_ai_rationale(chosen_one)
            else:
                chosen_one['rationale'] = "Data Unsafe for AI Analysis."
                
        return render_template('index.html', market_status=get_market_status(), vix=get_vix_data(), 
                             indices=get_indices(), results=results, chosen_one=chosen_one, scanned=True,
                             econ_events=get_economic_events())
    except Exception as e:
        print(f"Error in scan route: {e}")
        return "System Overload", 500

@app.route('/api/vix')
def api_vix(): return get_vix_data()

@app.route('/api/scan_data')
def api_scan_data():
    results = analyze_market_data(STOCK_TICKERS)
    chosen_one = None
    if results:
        chosen_one = results[0]
        if is_valid_data(chosen_one):
            chosen_one['rationale'] = get_ai_rationale(chosen_one)
        else:
            chosen_one['rationale'] = "Skipped - Invalid Data"
    return {"results": results, "chosen_one": chosen_one}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)