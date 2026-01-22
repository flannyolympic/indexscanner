import os
import time
import requests
import pandas as pd
import yfinance as yf
import pytz
from flask import Flask, render_template, request
from datetime import datetime, time as dt_time
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

# ASSET UNIVERSES (Default Lists)
STOCK_TICKERS = [
    "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "COIN", "MARA", "RIOT", "PLTR", "SOFI", "HOOD", "GME", "AMC", "SPY", "QQQ"
]
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "SHIB-USD",
    "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD", "UNI-USD"
]

def get_market_status():
    tz = pytz.timezone('America/New_York')
    now = datetime.now(tz)
    current_time = now.time()
    
    status = "MARKET CLOSED"
    color = "#ff4b4b" 
    
    if now.weekday() >= 5: return {"label": "WEEKEND CLOSED", "color": color}

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
        
    return {"label": status, "color": color}

def get_indices():
    indices = {"^GSPC": "S&P 500", "^IXIC": "NASDAQ", "^DJI": "DOW J"}
    data_list = []
    try:
        tickers = " ".join(indices.keys())
        data = yf.download(tickers, period="1d", group_by='ticker', threads=False)
        
        for ticker, name in indices.items():
            try:
                df = data[ticker] if len(indices) > 1 else data
                if df.empty: continue
                current = df['Close'].iloc[-1]
                open_price = df['Open'].iloc[0]
                change = ((current - open_price) / open_price) * 100
                sign = "+" if change >= 0 else ""
                data_list.append(f"{name}: {int(current)} ({sign}{round(change, 2)}%)")
            except: continue
    except: return ["MARKET DATA OFFLINE"]
    return data_list

def get_vix_data():
    try:
        vix = yf.Ticker("^VIX")
        price = vix.fast_info.last_price
        if not price:
            hist = vix.history(period="1d")
            price = hist['Close'].iloc[-1]
        val = round(price, 2)
        
        if val < 12: return {"val": val, "label": "COMPLACENCY", "color": "#00E5FF", "desc": "Overly Chill"}
        elif val < 15: return {"val": val, "label": "CALM / HEALTHY", "color": "#2ECC71", "desc": "Goldilocks Zone"}
        elif val < 20: return {"val": val, "label": "MILD CONCERN", "color": "#F1C40F", "desc": "Cautious Vibes"}
        elif val < 30: return {"val": val, "label": "ELEVATED FEAR", "color": "#E67E22", "desc": "Volatility Spikes"}
        elif val < 40: return {"val": val, "label": "HIGH ANXIETY", "color": "#D35400", "desc": "Serious Stress"}
        elif val < 50: return {"val": val, "label": "CRISIS MODE", "color": "#C0392B", "desc": "Full-Blown Panic"}
        else: return {"val": val, "label": "SYSTEM SHOCK", "color": "#8B0000", "desc": "Total Meltdown"}
    except:
        return {"val": 0, "label": "OFFLINE", "color": "#555", "desc": "No Connection"}

def analyze_market_data(ticker_list):
    results = []
    if not ticker_list: return []
    
    try:
        # Threads=False is critical for stability on free tier
        data = yf.download(tickers=" ".join(ticker_list), period="5d", interval="15m", group_by='ticker', auto_adjust=True, prepost=True, threads=False)
    except: return []

    for ticker in ticker_list:
        try:
            df = data[ticker] if len(ticker_list) > 1 else data
            if df.empty: continue

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
            
            # --- CATALYST LOGIC ---
            signal = "NEUTRAL"
            prob = 50
            setup = "Consolidation"
            catalyst = "No Clear Trigger"
            
            if current_rsi < 30:
                signal = "BULLISH"
                prob = 75
                setup = "Oversold Bounce"
                catalyst = "RSI Exhaustion (<30)"
            elif current_rsi > 70:
                signal = "BEARISH"
                prob = 70
                setup = "Overbought Pullback"
                catalyst = "RSI Extension (>70)"
            elif current_price > current_vwap * 1.01:
                signal = "BULLISH"
                prob = 60
                setup = "Momentum Breakout"
                catalyst = "Price > VWAP Support"
            elif current_price < current_vwap * 0.99:
                signal = "BEARISH"
                prob = 60
                setup = "Momentum Breakdown"
                catalyst = "Price < VWAP Resistance"

            stop = current_price * 0.98 if "BULLISH" in signal else current_price * 1.02
            target = current_price * 1.05 if "BULLISH" in signal else current_price * 0.95

            results.append({
                "ticker": ticker,
                "price": round(float(current_price), 2),
                "signal": signal,
                "probability": prob,
                "vwap": round(float(current_vwap), 2),
                "catalyst": catalyst, # NEW FIELD
                "setup": {
                    "type": setup,
                    "entry": f"${round(float(current_price), 2)}",
                    "stop": f"${round(float(stop), 2)}",
                    "target": f"${round(float(target), 2)}"
                }
            })
        except: continue
    return sorted(results, key=lambda x: x['probability'], reverse=True)

def get_ai_rationale(ticker_data):
    if not GENAI_API_KEY: return "AI Module Offline."
    models_to_try = ['gemini-2.0-flash-lite-preview-02-05', 'gemini-flash-latest', 'gemini-pro']
    
    prompt = (
        f"Act as a quantitative analyst. Analyze {ticker_data['ticker']}. "
        f"Price: {ticker_data['price']}, Signal: {ticker_data['signal']}, Catalyst: {ticker_data['catalyst']}. "
        f"Provide a 2-sentence institutional rationale. Be concise. No fluff."
    )

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response.text: return response.text
        except: continue
    return "AI Unavailable. Technical signal valid."

@app.route('/')
def home():
    return render_template('index.html', market_status=get_market_status(), vix=get_vix_data(), indices=get_indices(), current_mode="stock")

@app.route('/scan')
def scan():
    mode = request.args.get('mode', 'stock')
    user_query = request.args.get('ticker', '').upper().strip()
    
    # DETERMINE TICKER LIST
    if user_query:
        tickers = [user_query] # Search specific
    elif mode == 'crypto':
        tickers = CRYPTO_TICKERS
    else:
        tickers = STOCK_TICKERS # Default stock list
    
    results = analyze_market_data(tickers)
    
    chosen_one = None
    if results:
        chosen_one = results[0]
        chosen_one['rationale'] = get_ai_rationale(chosen_one)

    return render_template(
        'index.html',
        market_status=get_market_status(),
        vix=get_vix_data(),
        indices=get_indices(),
        current_mode=mode,
        results=results,
        chosen_one=chosen_one,
        search_term=user_query
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)