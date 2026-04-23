"""
SwingLab Setup Pipeline
=======================
Interactive orchestrator for the full project setup sequence.
Run this file to prepare data and train the final model from scratch.

Usage:
    python run_pipeline.py

Steps:
    1. Download market indicator data (SPY, VXX, USO, UUP, IEF)
    2. Download raw OHLCV data for all stocks in the universe
    3. Generate swing segments from the raw data
    4. Build the training dataset (feature engineering + multi-target tensor creation)
    5. Train the final multi-target LSTM model
"""

import os
import sys
import yaml
import numpy as np

from src.DBManager import DBManager
from src.DataDownloader import DataDownloader
from src.DataProcessor import SwingProcessor
from src.DatasetBuilder import DatasetBuilder
from src.LSTMModel import SwingLabLSTM


# ─────────────────────────────────────────────
# Status Checks
# ─────────────────────────────────────────────

def _status_indicators(db):
    """Returns (done: bool, detail: str)"""
    count = db.count_market_indicators()
    if count > 0:
        return True, f"{count:,} hourly rows across 5 ETFs"
    return False, "Not started"


def _status_stock_download(db):
    downloaded = len(db.get_loaded_symbols())
    total      = db.count_stocks_in_universe()
    if downloaded == 0:
        return False, "Not started"
    if downloaded < total:
        return False, f"{downloaded}/{total} stocks downloaded (in progress or incomplete)"
    return True, f"{downloaded}/{total} stocks fully downloaded"


def _status_segments(db):
    seg_count, sym_count = db.count_segments()
    if seg_count == 0:
        return False, "Not started"
    total_downloaded = len(db.get_loaded_symbols())
    if sym_count < total_downloaded:
        return False, f"{sym_count}/{total_downloaded} stocks segmented (incomplete)"
    return True, f"{seg_count:,} segments across {sym_count} stocks"


def _status_dataset(db):
    x_path = "Data/X_train.npy"
    y_path = "Data/y_train.npy"
    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        return True, f"Saved — shape {X.shape} (samples × sequence × features)"
    return False, "Not started"


def _status_model(db):
    path = "models/swinglab_lstm.weights.h5"
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        return True, f"Weights saved at {path} ({size_mb:.1f} MB)"
    return False, "Not trained yet"


# ─────────────────────────────────────────────
# Step Runners
# ─────────────────────────────────────────────

def run_indicators(db, config):
    print("\n" + "─" * 50)
    print("STEP 1: Downloading Market Indicators (4 Years)")
    print("─" * 50)
    api_key = config['api']['polygon_key']
    settings = config['historical_indicator_run']
    downloader = DataDownloader(api_key)
    downloader.run_indicators_pipeline(settings, db_manager=db)
    print("\n✓ Indicator download complete.")


def run_stock_download(db, config):
    print("\n" + "─" * 50)
    print("STEP 2: Downloading Raw Stock Data (2 Years)")
    print("─" * 50)
    api_key = config['api']['polygon_key']
    settings = config['historical_stock_run']
    downloader = DataDownloader(api_key)
    downloader.run_data_pipeline(settings, db_manager=db)
    print("\n✓ Stock download complete.")


def run_segment_generation(db, config):
    print("\n" + "─" * 50)
    print("STEP 3: Generating Swing Segments")
    print("─" * 50)

    loaded_symbols    = db.get_loaded_symbols()
    segmented_symbols = db.get_segmented_symbols()
    pending           = sorted(loaded_symbols - segmented_symbols)

    if not pending:
        print("All symbols already segmented. Nothing to do.")
        return

    total   = len(pending)
    done_so_far = len(segmented_symbols)
    print(f"Processing {total} symbols (skipping {done_so_far} already done)...\n")

    proc_settings = config.get('processing', {})
    order         = proc_settings.get('extrema_order', 5)
    
    processor     = SwingProcessor(extrema_order=order)
    success_count = 0
    skip_count    = 0
    error_count   = 0

    for idx, symbol in enumerate(pending, start=1):
        print(f"[{idx}/{total}] {symbol}...", end=" ", flush=True)
        try:
            raw_df = db.get_raw_stock_data(symbol)
            if raw_df.empty or len(raw_df) < 20:
                print("SKIP (too few rows)")
                skip_count += 1
                continue

            segments_df = processor.generate_segments(raw_df)
            if segments_df.empty:
                print("SKIP (no extrema detected)")
                skip_count += 1
                continue

            db.insert_segments(symbol, segments_df)
            print(f"OK ({len(segments_df)} segments)")
            success_count += 1

        except Exception as e:
            print(f"ERROR: {e}")
            error_count += 1

    print(f"\n✓ Segment generation complete.")
    print(f"  Succeeded: {success_count} | Skipped: {skip_count} | Errors: {error_count}")


def run_build_dataset(db, config):
    print("\n" + "─" * 50)
    print("STEP 4: Building Training Dataset (Multi-Target)")
    print("─" * 50)

    import glob
    print("  [Auto-Cleanup] Removing old dataset tensors and tuning caches...")
    for f in glob.glob("Data/*train*.npy") + glob.glob("Data/tuning_cache*.pkl"):
        try: os.remove(f)
        except OSError: pass

    proc_settings = config.get('processing', {})
    seq_len = config.get('model2', {}).get('sequence_length', 5)
    windows = config.get('indicator_windows_months', None)
    
    builder = DatasetBuilder(db_manager=db)
    X_train, y_train = builder.build_training_dataset(
        sequence_length=seq_len, 
        indicator_windows=windows
    )

    os.makedirs("Data", exist_ok=True)
    np.save("Data/X_train.npy", X_train)
    np.save("Data/y_train.npy", y_train)

    print(f"\n✓ Dataset saved.")
    print(f"  X_train: {X_train.shape}  (samples × sequence_length × features)")
    print(f"  y_train: {y_train.shape}")


def run_train_model(config):
    print("\n" + "-" * 50)
    print("STEP 5: Training Multi-Target LSTM Model")
    print("-" * 50)

    import glob
    print("  [Auto-Cleanup] Removing previous model weights...")
    for f in glob.glob("models/*.h5"):
        try: os.remove(f)
        except OSError: pass

    x_path, y_path = "Data/X_train.npy", "Data/y_train.npy"
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("ERROR: Dataset not found. Run Step 4 first.")
        return

    print("Loading datasets...")
    X = np.load(x_path)
    y = np.load(y_path)

    # 1. SANITY CHECK: Verify Data Integrity
    print("Performing final data sanity sweep...")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        print("CRITICAL ERROR: NaN or Infinite values detected in Training Tensors!")
        print("Check your feature scaling for zero-volatility stocks.")
        return
    print("[OK] Data Integrity: No NaNs or Infinite values detected.")

    # 2. CHRONOLOGICAL SPLIT (Manual)
    split_idx = int(len(X) * 0.80)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    _, sequence_length, num_features = X_train.shape
    print(f"Dataset Split: {len(X_train):,} Training samples | "
          f"{len(X_val):,} Validation (Future) samples")
    print(f"Model: {sequence_length} sequence length | {num_features} technical features")

    model = SwingLabLSTM(
        sequence_length=sequence_length,
        num_features=num_features,
        model_config=config.get('model2', {})
    )
    model.model.summary()

    train_cfg = config.get('model2', {})
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val, 
        epochs=train_cfg.get('epochs', 50), 
        batch_size=train_cfg.get('batch_size', 256),
    )
    
    model.save_weights("models/swinglab_lstm.weights.h5")
    model.plot_training_history(history)

    print("\n✓ Training complete. Weights saved to models/swinglab_lstm.weights.h5")

    # --- SUCCESS METRICS ---
    print("\nRunning test sweep against the unseen future dataset (X_val) to calculate Success %...")
    # Model outputs a 2D array [samples, 2]
    y_pred = model.model.predict(X_val, verbose=0)
    
    y_true_ret = y_val[:, 0]
    y_pred_ret = y_pred[:, 0]
    y_true_dur = y_val[:, 1]
    y_pred_dur = y_pred[:, 1]
    
    mae_ret = np.mean(np.abs(y_true_ret - y_pred_ret))
    correct_direction = np.sum(np.sign(y_true_ret) == np.sign(y_pred_ret))
    accuracy = correct_direction / len(y_true_ret) if len(y_true_ret) > 0 else 0
    
    mae_dur = np.mean(np.abs(y_true_dur - y_pred_dur))
    
    print("\n" + "="*50)
    print("  MULTI-TARGET SUCCESS METRICS (On Unseen Future Data)")
    print("="*50)
    print(f"  [TARGET 1 - PRICE RETURN]")
    print(f"  Mean Absolute Error:  {mae_ret:.4f}  (e.g., off by {mae_ret*100:.2f}%)")
    print(f"  Directional Accuracy: {accuracy*100:.1f}%\n")
    
    print(f"  [TARGET 2 - DURATION]")
    print(f"  Scaled Duration MAE:  {mae_dur:.4f}")
    
    if accuracy >= 0.55:
        print("\n  Status: Excellent edge! Highly profitable algorithm.")
    elif accuracy > 0.50:
        print("\n  Status: Slight edge. It statistically beats a coin flip!")
    else:
        print("\n  Status: Below 50%. Keep tuning hyperparameters.")
    print("="*50 + "\n")


# ─────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────

STEPS = [
    ("Market Indicators Download",  _status_indicators),
    ("Raw Stock Data Download",      _status_stock_download),
    ("Swing Segment Generation",     _status_segments),
    ("Build Training Dataset",       _status_dataset),
    ("Train Multi-Target LSTM Model",_status_model),
]

RUNNERS = [
    run_indicators,
    run_stock_download,
    run_segment_generation,
    run_build_dataset,
    run_train_model,
]


def print_menu(db):
    print("\n" + "=" * 55)
    print("  SwingLab -- First Run Setup")
    print("=" * 55)
    for i, (name, check_fn) in enumerate(STEPS, start=1):
        done, detail = check_fn(db)
        status_icon = "[X]" if done else "[ ]"
        print(f"  [{i}] {status_icon}  {name}")
        print(f"           {detail}")
    print("=" * 55)
    print("  [a] Run ALL remaining steps in sequence")
    print("  [q] Quit")
    print("=" * 55)


def run_step(step_idx, db, config):
    """Run a single step by 0-based index."""
    runner = RUNNERS[step_idx]
    
    if step_idx in (0, 1, 2, 3):
        runner(db, config)
    else:
        runner(config)


def main():
    config = yaml.safe_load(open("config.yaml", "r"))
    db_path = config['database']['path']
    db = DBManager(db_path=db_path)

    while True:
        print_menu(db)
        choice = input("  Enter choice: ").strip().lower()

        if choice == 'q':
            print("Exiting.")
            break

        elif choice == 'a':
            # Run all incomplete steps in order
            for i, (name, check_fn) in enumerate(STEPS):
                done, _ = check_fn(db)
                if not done:
                    print(f"\n→ Running: {name}")
                    run_step(i, db, config)
                else:
                    print(f"\n→ Skipping (already done): {name}")
            print("\n=== All steps complete. ===")
            break

        elif choice.isdigit() and 1 <= int(choice) <= len(STEPS):
            idx        = int(choice) - 1
            name, check_fn = STEPS[idx]
            done, detail   = check_fn(db)

            if done:
                print(f"\n⚠  Step {choice} already completed ({detail}).")
                override = input("   Run it again anyway? (y/n): ").strip().lower()
                if override != 'y':
                    continue

            run_step(idx, db, config)

        else:
            print("  Invalid choice. Enter a number 1-5, 'a', or 'q'.")


if __name__ == "__main__":
    main()
