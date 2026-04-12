import os
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

from src.DBManager import DBManager
from src.DataDownloader import DataDownloader
from src.DataProcessor import SwingProcessor
from src.FeatureEngineer import FeatureEngineer
from src.DatasetBuilder import DatasetBuilder
from src.LSTMModel import SwingLabLSTM
from src.Visualizer import PlotlyOracle


def get_ticker_metadata(symbol, db):
    """
    Fetches Sector, Beta, and Market Cap for a ticker.
    Prioritizes DB, falls back to yfinance.
    """
    with db._get_connection() as conn:
        df = pd.read_sql(f"SELECT sector, beta, market_cap_category FROM stocks_meta_data WHERE symbol = '{symbol}'", conn)
    
    if not df.empty and df.iloc[0]['sector']:
        return df.iloc[0].to_dict()
    
    # Fallback to yfinance
    print(f"Ticker {symbol} metadata missing. Fetching from yfinance...")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', 'Technology')
        beta = info.get('beta', 1.0)
        mcap_val = info.get('marketCap', 1e9)
    except:
        sector, beta, mcap_val = 'Technology', 1.0, 1e9

    if mcap_val > 2e11: mcap = 'Mega'
    elif mcap_val > 1e10: mcap = 'Large'
    elif mcap_val > 2e9: mcap = 'Mid'
    else: mcap = 'Small'
    
    return {'sector': sector, 'beta': beta, 'market_cap_category': mcap}

def run_prediction():
    print("\n" + "="*60)
    print("  SWINGLAB THE ORACLE: Live Ticker Predictor")

    print("="*60)
    
    symbol = input("Enter Ticker Symbol (e.g. NVDA): ").upper().strip()
    if not symbol: return

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # --- CONFIG AUDIT: Load important constants once ---
    SEQ_LEN = config.get('model2', {}).get('sequence_length', 5)
    EXTREMA_ORDER = config['processing']['extrema_order']
    DB_PATH = config['database']['path']
    POLY_KEY = config['api']['polygon_key']

    db = DBManager(DB_PATH)
    downloader = DataDownloader(POLY_KEY)
    processor = SwingProcessor(extrema_order=EXTREMA_ORDER)
    engineer = FeatureEngineer(mode='live', indicator_windows=config['indicator_windows_months'])
    
    # 1. Fetch Metadata
    meta = get_ticker_metadata(symbol, db)
    
    # 2. Fetch Ticker Data (Polygon)
    print(f"Downloading last 60 days of Hourly data for {symbol} (Polygon)...")
    ticker_df = downloader.fetch_stock_data(symbol, weeks=8)
    
    # 2b. Incremental Macro Sync
    print(f"Syncing Market Indicators (Incremental)...")
    with db._get_connection() as conn:
        indicators_df = pd.read_sql("SELECT * FROM market_indicators ORDER BY timestamp ASC", conn)
    
    latest_macro = indicators_df['timestamp'].max() if not indicators_df.empty else None
    
    if not latest_macro or (datetime.now() - pd.to_datetime(latest_macro)).total_seconds() > 12 * 3600:
        print(f"  Fetching fresh delta for macro refresh...")
        delta_df = downloader.fetch_market_indicators(years_back=0.02)
        if not delta_df.empty:
            indicators_df['timestamp'] = pd.to_datetime(indicators_df['timestamp'])
            delta_df['timestamp'] = pd.to_datetime(delta_df['timestamp'])
            indicators_df = pd.concat([indicators_df, delta_df]).drop_duplicates('timestamp').sort_values('timestamp')
            db.save_market_indicators(indicators_df)
            print(f"  ✓ Cache updated. Context: {len(indicators_df)} days.")
    else:
        print(f"  ✓ Indicators up to date.")

    # 3. Feature Engineering
    indicators_df = engineer.transform_market_indicators(indicators_df)
    
    print("Generating swing segments...")
    segments_df = processor.generate_segments(ticker_df)
    
    if len(segments_df) < 2:
        print("\n[!] ERROR: Not enough volatility found to generate swings.")
        print("This stock might be too flat for the 'Swing Processor' order setting.")
        return

    # 4. Global Join
    builder = DatasetBuilder(db)
    segments_df = builder._aggregate_indicators_per_segment(segments_df, indicators_df)
    segments_df['sector'] = meta['sector']
    segments_df['beta'] = meta['beta']
    segments_df['market_cap_category'] = meta['market_cap_category']
    
    print("Engineered features for live inference...")
    # --- CRITICAL FIX: Pull SEQ_LEN from processing section ---
    X_live = engineer.prepare_live_tensor(segments_df, sequence_length=SEQ_LEN)
    
    # 5. Load Model & Predict
    print("Loading Neural Network Brain...")
    model = SwingLabLSTM(
        sequence_length=SEQ_LEN,
        num_features=X_live.shape[2],
        model_config=config.get('model2', {})
    )
    model.load_weights('models/swinglab_lstm.weights.h5')
    
    prediction = model.predict(X_live)[0]
    pred_return = prediction[0]
    
    # Reverse scale the duration prediction
    pred_duration_scaled = np.array([[prediction[1]]])
    pred_duration_hours = engineer.duration_scaler.inverse_transform(pred_duration_scaled)[0][0]
    
    # Get current price
    current_price = segments_df.iloc[-1]['price_end']
    target_price = current_price * (1 + pred_return)
    
    # 6. GENERATE VISUAL REPORT
    oracle = PlotlyOracle()
    html_report, png_report = oracle.generate_report(symbol, ticker_df, segments_df, pred_return, pred_duration_hours)

    
    # 7. THE ORACLE: Output
    print("\n" + "*"*60)
    print(f"  PREDICTION FOR {symbol}")
    print("*"*60)
    print(f"Current Price:           ${current_price:.2f}")
    print(f"Predicted Swing Return:  {pred_return:+.2%}")
    print(f"Predicted Duration:      {pred_duration_hours:.1f} Hours")
    print(f"Expected Target Price:   ${target_price:.2f} in {pred_duration_hours:.1f} hours")
    print(f"Interactive Report:      {html_report}")
    
    if pred_return > 0.04: msg = "STRONG BULLISH: Model detects high-velocity breakout."
    elif pred_return > 0.01: msg = "MILD BULLISH: Positive drift expected."
    elif pred_return < -0.04: msg = "STRONG BEARISH: Model warns of deep exhaustion."
    elif pred_return < -0.01: msg = "MILD BEARISH: Expect some cooling off."
    else: msg = "NEUTRAL: Indecisive sideways action."
        
    print(f"Summary: {msg}")
    print("*"*60)
    print("(Note: Prediction targets the NEXT major swing extrema point.)\n")

if __name__ == "__main__":
    run_prediction()
