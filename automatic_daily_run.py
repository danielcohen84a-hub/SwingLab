"""
SwingLab — Automatic Daily Runner
===================================
Runs every business day at market close (via Windows Task Scheduler).
Incrementally syncs data, grades closed predictions, and issues new ones.

Flow (after first run):
  Step 1 — Sync new hourly candles    → raw_stock_data
  Step 2 — Sync macro indicators      → market_indicators
  Step 3 — Grade open predictions     → prediction_results
  Step 4 — Issue new predictions      → predictions

First run only:
  Step 0 — Bootstrap: pick 100 liquid stocks, download history,
            back-predict last closed segment (is_bootstrap=1),
            immediately grade it, then issue live prediction.

Usage:
    python automatic_daily_run.py                # Normal daily run
    python automatic_daily_run.py --setup-scheduler  # Register Windows Task Scheduler
"""

import os
import sys
import yaml
import time
import random
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix Windows terminal unicode encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.DBManager import DBManager
from src.DataDownloader import DataDownloader
from src.DataProcessor import SwingProcessor
from src.FeatureEngineer import FeatureEngineer
from src.DatasetBuilder import DatasetBuilder
from src.LSTMModel import SwingLabLSTM


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

WEIGHTS_PATH    = 'models/swinglab_lstm.weights.h5'
FEATURE_NAMES   = 'Data/feature_names.txt'
# 13 seconds = ~4.6 req/min, comfortably under the Polygon 5 req/min free-tier cap.
POLYGON_SLEEP   = 13.0
MIN_VOLUME      = 1_000_000   # min 90-day average volume for universe candidates
UNIVERSE_SIZE   = 100         # number of stocks to track
BOOTSTRAP_DAYS  = 60          # days of candle history to download on first run


# ─────────────────────────────────────────────────────────────────
# Setup Helpers
# ─────────────────────────────────────────────────────────────────

def setup_logging():
    """Writes to both stdout and a dated log file in outputs/."""
    os.makedirs('outputs', exist_ok=True)
    log_path = f"outputs/daily_run_{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding='utf-8'),
        ]
    )
    return logging.getLogger('SwingLabDaily')


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def setup_scheduler():
    """
    Registers automatic_daily_run.py as a Windows Task Scheduler task.
    Fires every day at 4:30 PM (30 min after US market close).
    The task will run for up to 2 hours to accommodate 100 stocks at Polygon's rate limit.
    Run once with: python automatic_daily_run.py --setup-scheduler
    """
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    cmd = (
        f'schtasks /create /tn "SwingLab Daily Close" '
        f'/tr "\\"{python_path}\\" \\"{script_path}\\"" '
        f'/sc daily /st 16:30 /du 0002:00 /f'
    )
    print("Registering Windows Task Scheduler task...")
    print(f"Command: {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Task registered successfully.")
        print("  Script will now run automatically every day at 4:30 PM.")
        print('  Verify with: schtasks /query /tn "SwingLab Daily Close"')
    else:
        print(f"✗ Failed: {result.stderr.strip()}")
        print("  Try running this script as Administrator and retry.")


# ─────────────────────────────────────────────────────────────────
# Model + Engineer Loader
# ─────────────────────────────────────────────────────────────────

def load_model_and_engineer(config, log):
    """
    Loads the LSTM model (with trained weights) and FeatureEngineer (with pre-fit scalers).
    Called once per daily run — not once per ticker.
    """
    if not os.path.exists(WEIGHTS_PATH):
        log.error(f"Model weights not found at '{WEIGHTS_PATH}'. Run run_pipeline.py Step 5 first.")
        sys.exit(1)
    if not os.path.exists(FEATURE_NAMES):
        log.error(f"Feature names file not found at '{FEATURE_NAMES}'. Run run_pipeline.py Step 4 first.")
        sys.exit(1)

    # Derive num_features from saved feature names list
    with open(FEATURE_NAMES, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    num_features = len(feature_names)

    model2_cfg = config.get('model2', {})
    SEQ_LEN = model2_cfg.get('sequence_length', config.get('processing', {}).get('sequence_length', 11))

    model = SwingLabLSTM(sequence_length=SEQ_LEN, num_features=num_features, model_config=model2_cfg)
    model.load_weights(WEIGHTS_PATH)
    log.info(f"  Model loaded: seq_len={SEQ_LEN}, num_features={num_features}")

    engineer = FeatureEngineer(mode='live', indicator_windows=config.get('indicator_windows_months', {}))
    log.info("  FeatureEngineer ready (live mode, pre-fit scalers).")

    return model, engineer


# ─────────────────────────────────────────────────────────────────
# Metadata Helper
# ─────────────────────────────────────────────────────────────────

def get_ticker_meta(ticker, db):
    """
    Returns ticker metadata (sector, beta, market_cap_category) from the DB.
    Returns None if the ticker has no usable metadata.
    """
    with db._get_connection() as conn:
        df = pd.read_sql(
            "SELECT sector, beta, market_cap_category FROM stocks_meta_data WHERE symbol = ?",
            conn, params=[ticker]
        )
    if df.empty or pd.isna(df.iloc[0]['sector']):
        return None
    row = df.iloc[0]
    return {
        'sector':              row['sector'] or 'Technology',
        'beta':                float(row['beta']) if pd.notna(row['beta']) else 1.0,
        'market_cap_category': row['market_cap_category'] or 'Large',
    }


# ─────────────────────────────────────────────────────────────────
# Core Prediction Logic
# ─────────────────────────────────────────────────────────────────

def _make_single_prediction(ticker, segments_df, meta, engineer, model, db, config, is_bootstrap, log):
    """
    Given a confirmed segment history (up to the last closed extremum), predicts
    the next swing segment and inserts the result into the `predictions` table.

    LEAK-FREE GUARANTEES:
    - segments_df contains only confirmed, closed segments (nothing from the future).
    - Macro indicators are joined via merge_asof(direction='backward'):
      each segment only sees indicators dated <= its own t_start.
    - price_at_prediction = open of the FIRST bar after the last confirmed extremum.
      This is the earliest realistic entry price; no intra-segment data is used.

    Returns: prediction_id (int) on success, or None on failure.
    """
    SEQ_LEN = config['model2']['sequence_length']
    if len(segments_df) < SEQ_LEN:
        log.warning(f"  {ticker}: Only {len(segments_df)} segments (need {SEQ_LEN}). Skipping.")
        return None

    # Load macro indicators — filtered per segment via merge_asof below
    indicators_df = db.get_market_indicators()
    if indicators_df.empty:
        log.warning(f"  {ticker}: No macro indicators in DB. Skipping.")
        return None

    indicators_engineered = engineer.transform_market_indicators(indicators_df)

    # Aggregate macro context using backward merge (no future leakage)
    builder = DatasetBuilder(db)
    segs = segments_df.copy()
    segs_enriched = builder._aggregate_indicators_per_segment(segs, indicators_engineered)
    segs_enriched['sector']              = meta['sector']
    segs_enriched['beta']                = meta['beta']
    segs_enriched['market_cap_category'] = meta['market_cap_category']

    # Build LSTM inference tensor [1, SEQ_LEN, num_features]
    try:
        X_live = engineer.prepare_live_tensor(segs_enriched, sequence_length=SEQ_LEN)
    except Exception as e:
        log.warning(f"  {ticker}: Tensor preparation failed: {e}")
        return None

    # Run inference
    raw_pred = model.model.predict(X_live, verbose=0)[0]   # shape: (2,)
    pred_return = float(raw_pred[0])

    # Reverse-scale duration from model output → bar count
    pred_duration_bars = int(max(1, round(
        float(engineer.duration_scaler.inverse_transform([[raw_pred[1]]])[0][0])
    )))

    # The entry extremum is the t_end of the LAST confirmed segment
    entry_extremum_time = pd.to_datetime(segments_df.iloc[-1]['t_end'])

    # segment_start_time = first trading bar AFTER the confirmed extremum (realistic entry)
    with db._get_connection() as conn:
        next_bar = pd.read_sql(
            "SELECT datetime, open FROM raw_stock_data "
            "WHERE symbol = ? AND datetime > ? ORDER BY datetime ASC LIMIT 1",
            conn, params=[ticker, str(entry_extremum_time)]
        )
    if next_bar.empty:
        log.warning(f"  {ticker}: No bar found after extremum {entry_extremum_time}. Skipping.")
        return None

    segment_start_time  = pd.to_datetime(next_bar.iloc[0]['datetime'])
    price_at_prediction = float(next_bar.iloc[0]['open'])
    price_target        = round(price_at_prediction * (1 + pred_return), 4)

    prediction_id = db.insert_prediction(
        ticker=ticker,
        predicted_at=datetime.now(),
        entry_extremum_time=entry_extremum_time,
        segment_start_time=segment_start_time,
        price_at_prediction=price_at_prediction,
        predicted_return=pred_return,
        predicted_duration_bars=pred_duration_bars,
        price_target=price_target,
        is_bootstrap=is_bootstrap,
    )

    direction = "▲" if pred_return > 0 else "▼"
    tag = " [bootstrap]" if is_bootstrap else ""
    log.info(
        f"  [{ticker}]{tag} id={prediction_id} | {direction}{abs(pred_return):.2%} in "
        f"{pred_duration_bars} bars | target ${price_target:.2f}"
    )
    return prediction_id


def _grade_single_prediction(pred_row, db, processor, log):
    """
    Attempts to grade one open prediction using confirmed candle data.

    Grading trigger: trading bars elapsed since segment_start_time >= predicted_duration_bars.
    Once triggered, SwingProcessor is run on the full candle history to detect whether
    a new confirmed extremum has formed at entry_extremum_time (= the start of the predicted
    segment). If confirmed, the segment is graded. If not yet confirmed, it waits.

    Returns True if successfully graded, False if not yet ready.
    """
    ticker               = pred_row['ticker']
    prediction_id        = int(pred_row['prediction_id'])
    predicted_duration   = int(pred_row['predicted_duration_bars'])
    entry_extremum_time  = pd.to_datetime(pred_row['entry_extremum_time'])
    segment_start_time   = pd.to_datetime(pred_row['segment_start_time'])
    predicted_return     = float(pred_row['predicted_return'])
    price_target         = float(pred_row['price_target'])

    # Count trading bars elapsed since trade entry
    with db._get_connection() as conn:
        bars_elapsed = pd.read_sql(
            "SELECT COUNT(*) as cnt FROM raw_stock_data WHERE symbol = ? AND datetime > ?",
            conn, params=[ticker, str(segment_start_time)]
        ).iloc[0]['cnt']

    if bars_elapsed < predicted_duration:
        log.debug(f"  {ticker} id={prediction_id}: {bars_elapsed}/{predicted_duration} bars. Not ready yet.")
        return False

    # Load full candle history and regenerate segments
    raw_df = db.get_raw_stock_data(ticker)
    if raw_df.empty:
        log.warning(f"  {ticker} id={prediction_id}: No raw data in DB.")
        return False
        
    if 'Datetime' not in raw_df.columns and raw_df.index.name == 'Datetime':
        raw_df = raw_df.reset_index()

    segments_df = processor.generate_segments(raw_df)
    if segments_df.empty:
        log.warning(f"  {ticker} id={prediction_id}: SwingProcessor returned no segments.")
        return False

    # The predicted segment has t_start ≈ entry_extremum_time.
    # (t_end of segment N == t_start of segment N+1 in SwingProcessor output)
    tol = pd.Timedelta(hours=2)
    graded_candidates = segments_df[
        (segments_df['t_start'] >= entry_extremum_time - tol) &
        (segments_df['t_start'] <= entry_extremum_time + tol)
    ]

    if graded_candidates.empty:
        log.debug(
            f"  {ticker} id={prediction_id}: {bars_elapsed} bars elapsed but extremum at "
            f"{entry_extremum_time} not yet confirmed by SwingProcessor. Will retry tomorrow."
        )
        return False

    if len(graded_candidates) > 1:
        log.warning(
            f"  {ticker} id={prediction_id}: {len(graded_candidates)} candidates near extremum "
            f"— using the first one."
        )

    graded = graded_candidates.iloc[0]
    actual_return        = float(graded['swing_return'])
    actual_duration_bars = int(graded['duration_hours'])
    direction_correct    = 1 if np.sign(actual_return) == np.sign(predicted_return) else 0
    return_error         = float(predicted_return - actual_return)
    duration_error       = int(predicted_duration - actual_duration_bars)

    # Check if price_target was hit at any bar within the actual realized segment
    seg_end = pd.to_datetime(graded['t_end'])
    with db._get_connection() as conn:
        bars_in_seg = pd.read_sql(
            "SELECT high, low FROM raw_stock_data "
            "WHERE symbol = ? AND datetime >= ? AND datetime <= ?",
            conn, params=[ticker, str(segment_start_time), str(seg_end)]
        )
    if bars_in_seg.empty:
        target_was_hit = 0
    elif predicted_return > 0:
        target_was_hit = 1 if bars_in_seg['high'].max() >= price_target else 0
    else:
        target_was_hit = 1 if bars_in_seg['low'].min() <= price_target else 0

    db.insert_prediction_result(
        prediction_id=prediction_id,
        ticker=ticker,
        actual_return=actual_return,
        actual_duration_bars=actual_duration_bars,
        direction_correct=direction_correct,
        return_error=return_error,
        duration_error=duration_error,
        target_was_hit=target_was_hit,
    )

    dir_str = "✓" if direction_correct else "✗"
    hit_str = "HIT" if target_was_hit else "MISS"
    log.info(
        f"  [{ticker}] GRADED id={prediction_id} | "
        f"Pred {predicted_return:+.2%} → Actual {actual_return:+.2%} | "
        f"Dir {dir_str} | Target {hit_str}"
    )
    return True


# ─────────────────────────────────────────────────────────────────
# Step 1 — Sync Candle Data
# ─────────────────────────────────────────────────────────────────

def step1_sync_candles(db, downloader, tracked_tickers, log):
    """
    Incrementally downloads new hourly candles for all tracked tickers.
    Only fetches candles since the last saved bar — never re-downloads history.
    """
    log.info(f"\n{'─'*55}")
    log.info(f"STEP 1: Syncing candle data ({len(tracked_tickers)} tickers)")
    log.info(f"{'─'*55}")

    synced, skipped, errors = 0, 0, 0

    for i, ticker in enumerate(tracked_tickers, 1):
        try:
            # Find the last bar we have in the DB for this ticker
            with db._get_connection() as conn:
                last_dt_row = pd.read_sql(
                    "SELECT MAX(datetime) as last_dt FROM raw_stock_data WHERE symbol = ?",
                    conn, params=[ticker]
                )
            last_dt_str = last_dt_row.iloc[0]['last_dt']

            if last_dt_str:
                last_dt = pd.to_datetime(last_dt_str)
                # Already fully up-to-date
                if last_dt.date() >= datetime.now().date():
                    log.debug(f"  [{i}/{len(tracked_tickers)}] {ticker}: Already up-to-date.")
                    skipped += 1
                    continue
                start_date = last_dt + timedelta(hours=1)
            else:
                # No existing data — download full bootstrap window
                start_date = datetime.now() - timedelta(days=BOOTSTRAP_DAYS)

            log.info(f"  [{i}/{len(tracked_tickers)}] {ticker}: Fetching from {start_date.strftime('%Y-%m-%d')}...")
            new_df = downloader.fetch_stock_data(
                ticker,
                start_date=start_date,
                end_date=datetime.now(),
                sleep_time=POLYGON_SLEEP,
            )

            if not new_df.empty:
                db.insert_raw_data(ticker, new_df)
                db.insert_load_tracking(ticker, start_date, datetime.now())
                log.info(f"    ✓ {len(new_df)} new bars saved.")
                synced += 1
            else:
                log.warning(f"    No new bars returned for {ticker}.")
                skipped += 1

            # Sleep between tickers to respect Polygon's 5 req/min free-tier limit
            time.sleep(POLYGON_SLEEP)

        except Exception as e:
            log.error(f"  [{i}] {ticker}: Download failed — {e}")
            errors += 1

    log.info(f"\n  Candle sync done: {synced} updated | {skipped} skipped | {errors} errors")


# ─────────────────────────────────────────────────────────────────
# Step 2 — Sync Macro Indicators
# ─────────────────────────────────────────────────────────────────

def step2_sync_indicators(db, downloader, log):
    """
    Incrementally fetches new daily closes for SPY, VXX, USO, UUP, IEF.
    Only downloads the delta since the last saved row.
    """
    log.info(f"\n{'─'*55}")
    log.info(f"STEP 2: Syncing macro indicators")
    log.info(f"{'─'*55}")

    indicators_df = db.get_market_indicators()
    latest = indicators_df['timestamp'].max() if not indicators_df.empty else None

    if latest is not None:
        latest = pd.to_datetime(latest)
        if latest.tzinfo is not None:
            latest = latest.tz_localize(None)

    # Skip if already updated today
    if latest is not None and (datetime.now() - latest).days < 1:
        log.info(f"  Indicators already up-to-date (last: {latest.date()}). Skipping.")
        return

    try:
        if latest is not None:
            log.info(f"  Fetching delta from {latest.date()} → today...")
            delta_df = downloader.fetch_market_indicators(start_date=latest)
        else:
            log.info("  First-time indicator fetch — downloading 4 years of history...")
            delta_df = downloader.fetch_market_indicators(years_back=4)

        if not delta_df.empty:
            if not indicators_df.empty:
                indicators_df['timestamp'] = pd.to_datetime(indicators_df['timestamp'])
                delta_df['timestamp']      = pd.to_datetime(delta_df['timestamp'])
                merged = (
                    pd.concat([indicators_df, delta_df])
                    .drop_duplicates('timestamp')
                    .sort_values('timestamp')
                    .reset_index(drop=True)
                )
            else:
                merged = delta_df

            db.save_market_indicators(merged)
            log.info(f"  ✓ Indicators updated: {len(merged)} total days in DB.")
        else:
            log.warning("  No new indicator data returned.")

    except Exception as e:
        log.error(f"  Indicator sync failed: {e}")


# ─────────────────────────────────────────────────────────────────
# Step 3 — Grade Open Predictions
# ─────────────────────────────────────────────────────────────────

def step3_grade_predictions(db, processor, log):
    """
    Reviews all open predictions and grades any where enough trading bars
    have elapsed AND SwingProcessor has confirmed the next extremum.
    """
    log.info(f"\n{'─'*55}")
    log.info(f"STEP 3: Grading open predictions")
    log.info(f"{'─'*55}")

    open_preds = db.get_open_predictions()
    if open_preds.empty:
        log.info("  No open predictions to grade.")
        return

    log.info(f"  {len(open_preds)} open prediction(s) found.")
    graded_count = 0

    for _, pred_row in open_preds.iterrows():
        try:
            if _grade_single_prediction(pred_row, db, processor, log):
                graded_count += 1
        except Exception as e:
            log.error(
                f"  {pred_row['ticker']} id={pred_row['prediction_id']}: Grade error — {e}",
                exc_info=True
            )

    log.info(f"\n  Graded {graded_count} / {len(open_preds)} prediction(s) this run.")


# ─────────────────────────────────────────────────────────────────
# Step 4 — Issue New Predictions
# ─────────────────────────────────────────────────────────────────

def step4_issue_predictions(db, config, model, engineer, processor, log):
    """
    For every tracked ticker whose last prediction has been graded (no open prediction),
    check if a new swing segment has formed and issue the next prediction.

    If 2+ new segments have formed since the last prediction (script was late, or the
    stock moved unusually fast), the missed segments are logged and skipped.
    The model always predicts from the MOST RECENT confirmed segment.
    """
    log.info(f"\n{'─'*55}")
    log.info(f"STEP 4: Issuing new predictions")
    log.info(f"{'─'*55}")

    tracked      = db.get_tracked_tickers()
    open_tickers = set(db.get_open_prediction_tickers())
    ready        = [t for t in tracked if t not in open_tickers]

    log.info(f"  {len(ready)} ticker(s) ready | {len(open_tickers)} still have open predictions.")

    SEQ_LEN = config['model2']['sequence_length']
    issued, skipped, errors = 0, 0, 0

    for ticker in ready:
        try:
            meta = get_ticker_meta(ticker, db)
            if meta is None:
                log.warning(f"  {ticker}: No metadata in DB. Skipping.")
                skipped += 1
                continue

            raw_df = db.get_raw_stock_data(ticker)
            if raw_df.empty:
                log.warning(f"  {ticker}: No raw candle data. Skipping.")
                skipped += 1
                continue
                
            if 'Datetime' not in raw_df.columns and raw_df.index.name == 'Datetime':
                raw_df = raw_df.reset_index()

            segments_df = processor.generate_segments(raw_df)
            if len(segments_df) < SEQ_LEN:
                log.warning(f"  {ticker}: Only {len(segments_df)} segments (need {SEQ_LEN}). Skipping.")
                skipped += 1
                continue

            # Check if new segments have appeared since the last prediction
            last_pred = db.get_latest_prediction_for_ticker(ticker)
            if last_pred is not None:
                last_extremum = pd.to_datetime(last_pred['entry_extremum_time'])
                new_segs = segments_df[segments_df['t_start'] > last_extremum]

                if new_segs.empty:
                    log.debug(f"  {ticker}: No new segment since last prediction.")
                    skipped += 1
                    continue

                if len(new_segs) > 1:
                    log.warning(
                        f"  {ticker}: {len(new_segs)} new segments detected since last prediction. "
                        f"Skipping {len(new_segs) - 1} missed segment(s) — predicting from latest."
                    )

            # Predict the NEXT segment using ALL confirmed segments as context
            pid = _make_single_prediction(
                ticker, segments_df, meta, engineer, model, db, config, is_bootstrap=0, log=log
            )
            if pid is not None:
                issued += 1
            else:
                skipped += 1

        except Exception as e:
            log.error(f"  {ticker}: Prediction failed — {e}", exc_info=True)
            errors += 1

    log.info(f"\n  New predictions: {issued} issued | {skipped} skipped | {errors} errors")


# ─────────────────────────────────────────────────────────────────
# Step 0 — First Run Bootstrap
# ─────────────────────────────────────────────────────────────────

def step0_bootstrap(db, config, model, engineer, processor, downloader, log):
    """
    First-run bootstrap — runs only once when the predictions table is empty.

    1. Selects UNIVERSE_SIZE liquid stocks (avg_volume_90d > MIN_VOLUME) at random.
    2. Downloads BOOTSTRAP_DAYS of hourly candles for each via Polygon.
    3. Syncs macro indicators (full history if needed).
    4. For each ticker:
       a. Back-predicts the most recently CLOSED segment (is_bootstrap=1):
          - Context window = segments[:-1]
          - Predicting what would have been segments[-1]
          - Immediately grades it (the actual result is already known)
       b. Issues a live prediction for the NEXT (future) segment (is_bootstrap=0):
          - Context window = all confirmed segments
    """
    log.info(f"\n{'='*55}")
    log.info("  FIRST RUN — Running Bootstrap")
    log.info(f"{'='*55}")

    # Select liquid universe
    with db._get_connection() as conn:
        candidates = pd.read_sql("""
            SELECT symbol FROM stocks_meta_data
            WHERE average_volume_90_days > :min_vol
              AND sector IS NOT NULL
              AND market_cap_category IS NOT NULL
        """, conn, params={'min_vol': MIN_VOLUME})

    if candidates.empty:
        log.error("No liquid candidates found in stocks_meta_data. Populate metadata first (run_pipeline.py).")
        sys.exit(1)

    all_candidates = candidates['symbol'].tolist()
    random.seed(42)   # Reproducible universe selection
    universe = random.sample(all_candidates, min(UNIVERSE_SIZE, len(all_candidates)))
    log.info(f"  Selected {len(universe)} stocks from {len(all_candidates)} liquid candidates.")

    # Sync indicators first — needed for all feature-engineering calls below
    step2_sync_indicators(db, downloader, log)

    SEQ_LEN = config['model2']['sequence_length']
    boot_issued, boot_graded, skipped, errors = 0, 0, 0, 0

    for i, ticker in enumerate(universe, 1):
        log.info(f"\n[{i}/{len(universe)}] Bootstrap: {ticker}")
        try:
            # ─── Download candle history ──────────────────────────────────
            start_date = datetime.now() - timedelta(days=BOOTSTRAP_DAYS)
            log.info(f"  Fetching {BOOTSTRAP_DAYS} days of hourly candles...")
            ticker_df = downloader.fetch_stock_data(
                ticker,
                start_date=start_date,
                end_date=datetime.now(),
                sleep_time=POLYGON_SLEEP,
            )

            if ticker_df.empty:
                log.warning("  No data returned. Skipping.")
                skipped += 1
                time.sleep(POLYGON_SLEEP)
                continue

            db.insert_raw_data(ticker, ticker_df)
            db.insert_load_tracking(ticker, start_date, datetime.now())
            log.info(f"  {len(ticker_df)} candles saved.")

            # ─── Generate swing segments ──────────────────────────────────
            raw_df = db.get_raw_stock_data(ticker)
            if 'Datetime' not in raw_df.columns and raw_df.index.name == 'Datetime':
                raw_df = raw_df.reset_index()
            segments_df = processor.generate_segments(raw_df)

            # Need SEQ_LEN for the context window + 1 to predict + 1 extra to have something to grade
            if len(segments_df) < SEQ_LEN + 2:
                log.warning(f"  Only {len(segments_df)} segments (need ≥{SEQ_LEN + 2}). Skipping.")
                skipped += 1
                continue

            meta = get_ticker_meta(ticker, db)
            if meta is None:
                log.warning("  No metadata found. Skipping.")
                skipped += 1
                continue

            # ─── Back-prediction: predict the LAST closed segment ─────────
            # Context = all confirmed segments EXCEPT the last one.
            # Target  = the last segment (already historically realized).
            context_segs = segments_df.iloc[:-1].copy()
            last_seg     = segments_df.iloc[-1]

            boot_pred_id = _make_single_prediction(
                ticker, context_segs, meta, engineer, model, db, config, is_bootstrap=1, log=log
            )

            if boot_pred_id is not None:
                # Retrieve what was inserted to compute grade fields
                boot_pred = db.get_prediction_by_id(boot_pred_id)
                predicted_return = float(boot_pred['predicted_return'])
                price_target     = float(boot_pred['price_target'])
                pred_dur_bars    = int(boot_pred['predicted_duration_bars'])
                seg_start_time   = pd.to_datetime(boot_pred['segment_start_time'])

                actual_return        = float(last_seg['swing_return'])
                actual_duration_bars = int(last_seg['duration_hours'])
                direction_correct    = 1 if np.sign(actual_return) == np.sign(predicted_return) else 0
                return_error         = float(predicted_return - actual_return)
                duration_error       = int(pred_dur_bars - actual_duration_bars)

                # target_was_hit: check highs/lows within the actual realized segment
                seg_end_time = pd.to_datetime(last_seg['t_end'])
                with db._get_connection() as conn:
                    bars_in_seg = pd.read_sql(
                        "SELECT high, low FROM raw_stock_data "
                        "WHERE symbol = ? AND datetime >= ? AND datetime <= ?",
                        conn, params=[ticker, str(seg_start_time), str(seg_end_time)]
                    )
                if bars_in_seg.empty:
                    target_was_hit = 0
                elif predicted_return > 0:
                    target_was_hit = 1 if bars_in_seg['high'].max() >= price_target else 0
                else:
                    target_was_hit = 1 if bars_in_seg['low'].min() <= price_target else 0

                db.insert_prediction_result(
                    prediction_id=boot_pred_id,
                    ticker=ticker,
                    actual_return=actual_return,
                    actual_duration_bars=actual_duration_bars,
                    direction_correct=direction_correct,
                    return_error=return_error,
                    duration_error=duration_error,
                    target_was_hit=target_was_hit,
                )

                dir_str = "✓" if direction_correct else "✗"
                hit_str = "HIT" if target_was_hit else "MISS"
                log.info(
                    f"  Bootstrap grade: Pred {predicted_return:+.2%} → Actual {actual_return:+.2%} "
                    f"| Dir {dir_str} | Target {hit_str}"
                )
                boot_graded += 1

            # ─── Live prediction: predict the NEXT (future) segment ───────
            # Context = ALL confirmed segments. (The one after this is unknown.)
            live_pred_id = _make_single_prediction(
                ticker, segments_df, meta, engineer, model, db, config, is_bootstrap=0, log=log
            )
            if live_pred_id is not None:
                boot_issued += 1

            # Rate limit: sleep between tickers
            time.sleep(POLYGON_SLEEP)

        except Exception as e:
            log.error(f"  {ticker}: Bootstrap error — {e}", exc_info=True)
            errors += 1

    log.info(f"\n{'='*55}")
    log.info(
        f"  Bootstrap complete: {boot_issued} live predictions | "
        f"{boot_graded} bootstrap grades | {skipped} skipped | {errors} errors"
    )
    log.info(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SwingLab Automatic Daily Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python automatic_daily_run.py                 # Run normally\n'
            '  python automatic_daily_run.py --setup-scheduler  # Schedule daily 4:30 PM task\n'
        )
    )
    parser.add_argument(
        '--setup-scheduler', action='store_true',
        help='Register this script as a Windows Task Scheduler task (runs at 4:30 PM daily)'
    )
    args = parser.parse_args()

    if args.setup_scheduler:
        setup_scheduler()
        return

    log = setup_logging()
    log.info("=" * 55)
    log.info("  SwingLab — Automatic Daily Runner")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    config     = load_config()
    db         = DBManager(config['database']['path'])
    downloader = DataDownloader(config['api']['polygon_key'])
    processor  = SwingProcessor(config['processing']['extrema_order'])

    # Load model and scalers once — expensive, don't repeat per ticker
    model, engineer = load_model_and_engineer(config, log)

    if db.is_predictions_empty():
        # ── First ever run ──────────────────────────────────────────
        step0_bootstrap(db, config, model, engineer, processor, downloader, log)
    else:
        # ── Normal daily run ────────────────────────────────────────
        tracked = db.get_tracked_tickers()
        log.info(f"\nCurrently tracking {len(tracked)} stock(s).")

        step1_sync_candles(db, downloader, tracked, log)
        step2_sync_indicators(db, downloader, log)
        step3_grade_predictions(db, processor, log)
        step4_issue_predictions(db, config, model, engineer, processor, log)

    log.info(f"\n✓ Run complete at {datetime.now().strftime('%H:%M:%S')}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
