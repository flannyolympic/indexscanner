import os
import time
import requests
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request
from datetime import datetime
from google import genai

app = Flask(__name__)

# --- CONFIGURATION ---
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")

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

            # Technicals
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
            
            # Logic
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
    """
    Updated for google-genai SDK v0.3+ and gemini-1.5-flash
    """
    if not GENAI_API_KEY:
        return "AI Module Offline: Check API Key."
    
    try:
        client = genai.Client(api_key=GENAI_API_KEY)
        
        prompt = (
            f"As a hedge fund analyst, give a 1-sentence risk assessment for {ticker_data['ticker']}. "
            f"Technical Context: Price ${ticker_data['price']}, Signal {ticker_data['signal']}, "
            f"Setup {ticker_data['setup']['type']}. Be direct and professional."
        )
        
        # Using 'gemini-1.5-flash' for maximum speed and lower latency
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Connection Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html', market_status=get_market_status(), current_mode="stock")

@app.route('/scan')
def scan():
    mode = request.args.get('mode', 'stock')
    tickers = CRYPTO_TICKERS if mode == 'crypto' else STOCK_TICKERS
    
    results = analyze_market_data(tickers)
    
    chosen_one = None
    if results:
        chosen_one = results[0]
        # Only run AI on the top pick to save time
        chosen_one['rationale'] = get_ai_rationale(chosen_one)

    return render_template(
        'index.html',
        market_status=get_market_status(),
        current_mode=mode,
        results=results,
        chosen_one=chosen_one
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)