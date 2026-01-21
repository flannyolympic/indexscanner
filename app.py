import os
import time
import requests
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request
from datetime import datetime
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

# ASSET UNIVERSES
STOCK_TICKERS = [
    "TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX",
    "COIN", "MARA", "RIOT", "PLTR", "SOFI", "HOOD", "GME", "AMC", "SPY", "QQQ"
]
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "SHIB-USD",
    "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD", "UNI-USD"
]

def get_market_status():
    now = datetime.now()
    if 9 <= now.hour < 16 and now.weekday() < 5:
        return "MARKET OPEN"
    return "MARKET CLOSED"

# --- NEW: VIX GAUGE LOGIC ---
def get_vix_data():
    try:
        # Fetch live VIX data
        vix = yf.Ticker("^VIX")
        # fast_info is faster/stable for single points
        price = vix.fast_info.last_price
        
        # Fallback if fast_info fails
        if not price:
            hist = vix.history(period="1d")
            price = hist['Close'].iloc[-1]
            
        val = round(price, 2)
        
        # Color & Label Logic (Based on your Screenshot)
        if val < 12:
            return {"val": val, "label": "COMPLACENCY", "color": "#00E5FF", "desc": "Overly Chill"} # Cyan/Teal
        elif val < 15:
            return {"val": val, "label": "CALM / HEALTHY", "color": "#2ECC71", "desc": "Goldilocks Zone"} # Green
        elif val < 20:
            return {"val": val, "label": "MILD CONCERN", "color": "#F1C40F", "desc": "Cautious Vibes"} # Yellow
        elif val < 30:
            return {"val": val, "label": "ELEVATED FEAR", "color": "#E67E22", "desc": "Volatility Spikes"} # Orange
        elif val < 40:
            return {"val": val, "label": "HIGH ANXIETY", "color": "#D35400", "desc": "Serious Stress"} # Dark Orange
        elif val < 50:
            return {"val": val, "label": "CRISIS MODE", "color": "#C0392B", "desc": "Full-Blown Panic"} # Red
        else:
            return {"val": val, "label": "SYSTEM SHOCK", "color": "#8B0000", "desc": "Total Meltdown"} # Dark Red

    except Exception as e:
        print(f"VIX Error: {e}")
        return {"val": 0, "label": "OFFLINE", "color": "#555555", "desc": "No Connection"}

def analyze_market_data(ticker_list):
    results = []
    try:
        data = yf.download(
            tickers=" ".join(ticker_list), 
            period="5d", 
            interval="15m", 
            group_by='ticker', 
            auto_adjust=True, 
            prepost=True, 
            threads=True
        )
    except Exception as e:
        print(f"Download Error: {e}")
        return []

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
            
            signal = "NEUTRAL"
            prob = 50
            setup = "Consolidation"
            
            if current_rsi < 30:
                signal = "BULLISH (OVERSOLD)"
                prob = 75
                setup = "Mean Reversion"
            elif current_rsi > 70:
                signal = "BEARISH (OVERBOUGHT)"
                prob = 70
                setup = "Top Reversal"
            elif current_price > current_vwap * 1.01:
                signal = "BULLISH (TREND)"
                prob = 60
                setup = "VWAP Support"
            elif current_price < current_vwap * 0.99:
                signal = "BEARISH (TREND)"
                prob = 60
                setup = "VWAP Resistance"

            stop = current_price * 0.98 if "BULLISH" in signal else current_price * 1.02
            target = current_price * 1.05 if "BULLISH" in signal else current_price * 0.95

            results.append({
                "ticker": ticker,
                "price": round(float(current_price), 2),
                "signal": signal,
                "probability": prob,
                "vwap": round(float(current_vwap), 2),
                "setup": {
                    "type": setup,
                    "entry": f"${round(float(current_price), 2)}",
                    "stop": f"${round(float(stop), 2)}",
                    "target": f"${round(float(target), 2)}"
                }
            })
        except:
            continue

    return sorted(results, key=lambda x: x['probability'], reverse=True)

def get_ai_rationale(ticker_data):
    if not GENAI_API_KEY: return "AI Module Offline."
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        prompt = (
            f"Act as a quantitative analyst. Analyze {ticker_data['ticker']}. "
            f"Price: {ticker_data['price']}, Signal: {ticker_data['signal']}, Setup: {ticker_data['setup']['type']}. "
            f"Provide a 2-sentence institutional rationale. Be concise and professional. No fluff."
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- ROUTES ---
@app.route('/')
def home():
    # Pass VIX data to home page too
    return render_template(
        'index.html', 
        market_status=get_market_status(), 
        vix=get_vix_data(),
        current_mode="stock"
    )

@app.route('/scan')
def scan():
    mode = request.args.get('mode', 'stock')
    tickers = CRYPTO_TICKERS if mode == 'crypto' else STOCK_TICKERS
    
    results = analyze_market_data(tickers)
    chosen_one = None
    if results:
        chosen_one = results[0]
        chosen_one['rationale'] = get_ai_rationale(chosen_one)

    return render_template(
        'index.html',
        market_status=get_market_status(),
        vix=get_vix_data(), # Live VIX on scan result page too
        current_mode=mode,
        results=results,
        chosen_one=chosen_one
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)