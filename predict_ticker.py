import os
import sys
# Fix Windows terminal unicode encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

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
        df = pd.read_sql(
            f"SELECT sector, beta, market_cap_category FROM stocks_meta_data WHERE symbol = '{symbol}'",
            conn
        )

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
        if not sector:
            sector = 'Technology'
        if not beta:
            beta = 1.0
        if not mcap_val:
            mcap_val = 1e9
    except Exception:
        sector, beta, mcap_val = 'Technology', 1.0, 1e9

    if mcap_val > 2e11:
        mcap = 'Mega'
    elif mcap_val > 1e10:
        mcap = 'Large'
    elif mcap_val > 2e9:
        mcap = 'Mid'
    else:
        mcap = 'Small'

    return {'sector': sector, 'beta': beta, 'market_cap_category': mcap}


def run_prediction():
    print("\n" + "=" * 60)
    print("  SWINGLAB THE ORACLE: Live Ticker Predictor")
    print("=" * 60)

    symbol = input("Enter Ticker Symbol (e.g. NVDA): ").upper().strip()
    if not symbol:
        return

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- Config constants ---
    # Model2 is the current dual-output model (return + duration).
    # SEQ_LEN must match what Model2 was trained on (from model2 config or processing fallback).
    model2_cfg = config.get('model2', {})
    SEQ_LEN = model2_cfg.get('sequence_length', config.get('processing', {}).get('sequence_length', 5))
    EXTREMA_ORDER = config['processing']['extrema_order']
    DB_PATH = config['database']['path']
    POLY_KEY = config['api']['polygon_key']

    WEIGHTS_PATH = 'models/swinglab_lstm2.weights.h5'

    db = DBManager(DB_PATH)
    downloader = DataDownloader(POLY_KEY)
    processor = SwingProcessor(extrema_order=EXTREMA_ORDER)
    engineer = FeatureEngineer(mode='live', indicator_windows=config['indicator_windows_months'])

    # ── Step 1: Fetch stock metadata ────────────────────────────────
    print(f"\n[1/6] Fetching metadata for {symbol}...")
    meta = get_ticker_metadata(symbol, db)
    print(f"      Sector: {meta['sector']} | Beta: {meta['beta']:.2f} | Cap: {meta['market_cap_category']}")

    # ── Step 2: Download ticker OHLCV (Polygon, 8 weeks) ────────────
    print(f"\n[2/6] Downloading 8 weeks of hourly data for {symbol} (Polygon)...")
    try:
        ticker_df = downloader.fetch_stock_data(symbol, weeks=8)
    except Exception as e:
        print(f"      [ERROR] Polygon fetch failed: {e}")
        print("      Falling back to yfinance...")
        try:
            ticker_df = downloader.fetch_stock_data_fast(symbol, weeks=8)
        except Exception as e2:
            print(f"      [ERROR] yfinance fallback also failed: {e2}")
            return
    print(f"      Downloaded {len(ticker_df)} hourly candles.")

    # ── Step 3: Incremental macro indicator sync ─────────────────────
    print(f"\n[3/6] Syncing macro indicators (SPY, VXX, USO, UUP, IEF)...")
    with db._get_connection() as conn:
        indicators_df = pd.read_sql(
            "SELECT * FROM market_indicators ORDER BY timestamp ASC", conn,
            parse_dates=['timestamp']
        )

    latest_macro = indicators_df['timestamp'].max() if not indicators_df.empty else None

    if latest_macro is not None:
        latest_macro = pd.to_datetime(latest_macro)
        if latest_macro.tzinfo is not None:
            latest_macro = latest_macro.tz_localize(None)

    needs_refresh = (
        latest_macro is None or
        (datetime.now() - latest_macro).total_seconds() > 12 * 3600
    )

    if needs_refresh:
        print("      Cache is stale. Fetching fresh delta from yfinance...")
        try:
            if latest_macro is not None:
                # Only fetch the missing delta window
                delta_df = downloader.fetch_market_indicators(start_date=latest_macro)
            else:
                # First-run: fetch full 4-year history
                delta_df = downloader.fetch_market_indicators(years_back=4)
            if not delta_df.empty:
                indicators_df['timestamp'] = pd.to_datetime(indicators_df['timestamp'])
                delta_df['timestamp'] = pd.to_datetime(delta_df['timestamp'])
                indicators_df = (
                    pd.concat([indicators_df, delta_df])
                    .drop_duplicates('timestamp')
                    .sort_values('timestamp')
                    .reset_index(drop=True)
                )
                db.save_market_indicators(indicators_df)
                print(f"      [OK] Cache updated. {len(indicators_df)} days of macro context.")
        except Exception as e:
            print(f"      [WARN] Delta fetch failed ({e}). Using cached data.")
    else:
        print(f"      [OK] Indicators up to date ({len(indicators_df)} days cached).")

    # ── Step 4: Feature engineering ──────────────────────────────────
    print(f"\n[4/6] Engineering features...")

    # Transform raw ETF closes -> rolling Z-score macro regimes
    indicators_engineered = engineer.transform_market_indicators(indicators_df)

    # Generate swing segments from raw price data
    segments_df = processor.generate_segments(ticker_df)
    if len(segments_df) < SEQ_LEN + 1:
        print(f"\n[ERROR] Only {len(segments_df)} swing segments detected (need >{SEQ_LEN}).")
        print("  Try a more volatile stock, or reduce extrema_order in config.yaml.")
        return
    print(f"      Detected {len(segments_df)} swing segments.")

    # Join macro context + stock metadata
    builder = DatasetBuilder(db)
    segments_df = builder._aggregate_indicators_per_segment(segments_df, indicators_engineered)
    segments_df['sector'] = meta['sector']
    segments_df['beta'] = meta['beta']
    segments_df['market_cap_category'] = meta['market_cap_category']

    # Build the live inference tensor [1, seq_len, num_features]
    X_live = engineer.prepare_live_tensor(segments_df, sequence_length=SEQ_LEN)
    num_features = X_live.shape[2]
    print(f"      Tensor shape: {X_live.shape}  (1 x {SEQ_LEN} steps x {num_features} features)")

    # ── Step 5: Load model and predict ───────────────────────────────
    print(f"\n[5/6] Loading neural network ({WEIGHTS_PATH})...")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"      [ERROR] Model weights not found at '{WEIGHTS_PATH}'.")
        print("      Run run_pipeline.py Step 5 to train the model first.")
        return

    model = SwingLabLSTM(
        sequence_length=SEQ_LEN,
        num_features=num_features,
        model_config=model2_cfg
    )
    model.load_weights(WEIGHTS_PATH)

    raw_pred = model.model.predict(X_live, verbose=0)[0]  # shape: (2,)
    pred_return = float(raw_pred[0])

    # Reverse-scale the duration prediction back to hours
    pred_duration_scaled = np.array([[raw_pred[1]]])
    pred_duration_hours = float(
        engineer.duration_scaler.inverse_transform(pred_duration_scaled)[0][0]
    )
    # Clamp to sensible range (1h – 500h)
    pred_duration_hours = max(1.0, min(pred_duration_hours, 500.0))

    current_price = float(segments_df.iloc[-1]['price_end'])
    target_price = current_price * (1 + pred_return)

    # ── Step 6: Generate visual report ───────────────────────────────
    print(f"\n[6/6] Generating interactive visual report...")
    try:
        oracle = PlotlyOracle()
        html_report, png_report = oracle.generate_report(
            symbol, ticker_df, segments_df, pred_return, pred_duration_hours
        )
    except Exception as e:
        print(f"      [WARN] Report generation failed: {e}")
        html_report = "N/A"

    # ── Final output ─────────────────────────────────────────────────
    direction = "UP" if pred_return > 0 else "DOWN"
    if abs(pred_return) > 0.04:
        strength = "STRONG"
    elif abs(pred_return) > 0.01:
        strength = "MILD"
    else:
        strength = "NEUTRAL"

    if strength == "NEUTRAL":
        msg = "NEUTRAL: Indecisive sideways action expected."
    elif pred_return > 0:
        msg = f"{strength} BULLISH: Model detects upward swing." if strength == "STRONG" else "MILD BULLISH: Positive drift expected."
    else:
        msg = f"{strength} BEARISH: Model warns of downward swing." if strength == "STRONG" else "MILD BEARISH: Expect some cooling off."

    print("\n" + "*" * 60)
    print(f"  ORACLE PREDICTION FOR: {symbol}")
    print("*" * 60)
    print(f"  Current Price      : ${current_price:.2f}")
    print(f"  Predicted Return   : {pred_return:+.2%}  ({direction})")
    print(f"  Predicted Duration : {pred_duration_hours:.1f} hours")
    print(f"  Target Price       : ${target_price:.2f} in ~{pred_duration_hours:.0f} hours")
    print(f"  Summary            : {msg}")
    if html_report != "N/A":
        print(f"  Visual Report      : {html_report}")
    print("*" * 60)
    print("  (Prediction targets the NEXT major swing extrema point.)\n")


if __name__ == "__main__":
    run_prediction()
