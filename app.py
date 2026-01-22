import os
import time
import requests
import pandas as pd
import yfinance as yf
import pytz
import random
from flask import Flask, render_template, request
from datetime import datetime, time as dt_time
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if GENAI_API_KEY:
    try:
        genai.configure(api_key=GENAI_API_KEY)
    except Exception as e:
        print(f"DEBUG: Failed to configure GenAI: {e}")

# ASSET UNIVERSE
STOCK_TICKERS = [
    "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "COIN", "MARA", "PLTR", "SOFI", "HOOD", "GME", "AMC", "SPY", "QQQ", "IWM",
    "MSTR", "DKNG", "UBER", "ABNB", "ROKU", "PYPL"
]

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
    except: return ["DATA STREAM OFFLINE"]
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
    try:
        data = yf.download(tickers=" ".join(ticker_list), period="5d", interval="15m", group_by='ticker', auto_adjust=True, prepost=True, threads=False)
    except: return []

    for ticker in ticker_list:
        try:
            df = data[ticker] if len(ticker_list) > 1 else data
            if df.empty: continue

            df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP'] = (df['TP'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            current_price = df['Close'].iloc[-1]
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            current_rsi = df['RSI'].iloc[-1]
            current_vwap = df['VWAP'].iloc[-1]
            current_vol = df['Volume'].iloc[-1]
            avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
            
            signal = "NEUTRAL"
            prob = 50
            catalyst = "No Clear Trigger"
            sentiment = "Neutral"
            options_play = "Iron Condor"

            if current_rsi < 30:
                signal, prob, catalyst, sentiment, options_play = ("BULLISH (OVERSOLD)", 78, "RSI Exhaustion", "Fear (Buy Dip)", "Sell Puts")
            elif current_rsi > 70:
                signal, prob, catalyst, sentiment, options_play = ("BEARISH (OVERBOUGHT)", 72, "RSI Extension", "Greed (Sell)", "Buy Puts")
            elif current_price > current_vwap * 1.01:
                signal, prob, catalyst, sentiment, options_play = ("BULLISH (MOMENTUM)", 65, "VWAP Breakout", "Bullish Flow", "Debit Call Spreads")
            elif current_price < current_vwap * 0.99:
                signal, prob, catalyst, sentiment, options_play = ("BEARISH (MOMENTUM)", 65, "VWAP Breakdown", "Bearish Flow", "Debit Put Spreads")
            
            if current_vol > avg_vol * 1.5:
                prob += 5
                catalyst += " + High Vol"

            results.append({
                "ticker": ticker,
                "price": round(float(current_price), 2),
                "signal": signal,
                "probability": prob,
                "vwap": round(float(current_vwap), 2),
                "catalyst": catalyst,
                "sentiment": sentiment,
                "options_play": options_play
            })
        except: continue
    
    return sorted(results, key=lambda x: x['probability'], reverse=True)

# --- CORRECTED AI FUNCTION ---
def get_ai_rationale(ticker_data):
    if not GENAI_API_KEY:
        print("DEBUG: GENAI_API_KEY is missing!")
        return "AI Offline (Key Missing). Trade based on technicals."
    
    # UPDATED MODEL LIST: Using stable models with better limits
    models = ['gemini-1.5-flash', 'gemini-1.5-flash-latest']
    
    prompt = (
        f"Analyze {ticker_data['ticker']} (${ticker_data['price']}). "
        f"Signal: {ticker_data['signal']}. Catalyst: {ticker_data['catalyst']}. "
        f"Sentiment: {ticker_data['sentiment']}. "
        f"Suggest a concise options trade setup and explain why in one sentence."
    )

    for m in models:
        try:
            print(f"DEBUG: Trying model {m}...")
            model = genai.GenerativeModel(m)
            response = model.generate_content(prompt)
            if response.text:
                return response.text
        except Exception as e:
            print(f"DEBUG: Model {m} failed: {e}")
            continue
            
    return "AI Limit Reached (Free Tier). Rely on Signal/Probability."

@app.route('/')
def home():
    return render_template('index.html', market_status=get_market_status(), vix=get_vix_data(), indices=get_indices())

@app.route('/scan')
def scan():
    results = analyze_market_data(STOCK_TICKERS)
    chosen_one = None
    if results:
        chosen_one = results[0]
        chosen_one['rationale'] = get_ai_rationale(chosen_one)

    return render_template(
        'index.html',
        market_status=get_market_status(),
        vix=get_vix_data(),
        indices=get_indices(),
        results=results,
        chosen_one=chosen_one,
        scanned=True
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)