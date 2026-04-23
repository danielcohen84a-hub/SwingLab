"""
SwingLab Hyperparameter Tuner
==============================
Uses Optuna (Bayesian TPE search) to find the optimal combination of:
  - sequence_length  : how many past swings the LSTM looks back
  - learning_rate    : Adam optimizer step size
  - dropout_rate     : regularization strength
  - hidden_units     : LSTM memory size

Time budget design (~30 min total):
  - processed_df is cached to Data/tuning_cache.parquet on first run
    (saves the ~10-15 min DB+feature-engineering step on every repeat run).
  - top 500 stocks (~230k segments) — full quality sample.
  - 25 epochs per trial with EarlyStopping(patience=8).
  - MedianPruner kills clearly bad trials after 4 warmup epochs.
  - 25 trials with batch_size=256 ≈ 25 min training after cache is warm.

First run:  ~10-15 min (cache build) + ~25 min (25 trials) ≈ 35-40 min
Repeat run: ~5s  (cache load)        + ~25 min (25 trials) ≈ 26 min

Usage:
    python tune_hyperparameters.py              # 25 trials (default, ~30 min)
    python tune_hyperparameters.py --trials 35  # more thorough, ~40 min
    python tune_hyperparameters.py --rebuild    # force fresh cache rebuild
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import optuna
import argparse
import time

from src.DBManager import DBManager
from src.DatasetBuilder import DatasetBuilder
from src.LSTMModel import SwingLabLSTM

# Suppress Optuna verbosity (we print our own per-trial summary)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants  (these are the quality/speed knobs)
# ─────────────────────────────────────────────────────────────────────────────
TOP_N_STOCKS = 500    # top stocks by segment count — full quality sample
MAX_EPOCHS   = 25     # same as final training; important for dual-target convergence
BATCH_SIZE   = 256    # small batches → more gradient steps → better signal per trial
PATIENCE     = 8      # EarlyStopping patience (allows late-blooming models to converge)
CACHE_PATH   = "Data/tuning_cache.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_processed_df(config, force_rebuild=False):
    """
    Returns the 2D engineered DataFrame.

    First run  : builds from DB (~10-15 min), saves to Parquet cache.
    Repeat runs: loads cache in ~5 seconds.
    Auto-invalidates if the SQLite DB file is newer than the cache
    (i.e. the pipeline was re-run and new segments were added).
    """
    db_path = config['database']['path']

    if not force_rebuild and os.path.exists(CACHE_PATH):  # noqa
        cache_mtime = os.path.getmtime(CACHE_PATH)
        db_mtime    = os.path.getmtime(db_path)

        if db_mtime < cache_mtime:
            print(f"  Loading cached features from {CACHE_PATH}...")
            t0 = time.time()
            df = pd.read_pickle(CACHE_PATH)
            print(f"  Loaded {len(df):,} segments in {time.time()-t0:.1f}s.")
            return df

        print("  DB has changed since last cache — rebuilding cache...")

    print("  Building features from DB (first run or forced rebuild).")
    print("  This takes ~10-15 minutes but only happens once per DB version.")
    t0 = time.time()

    db      = DBManager(db_path)
    builder = DatasetBuilder(db)
    windows = config.get('indicator_windows_months', {})
    df      = builder.prepare_processed_df(indicator_windows=windows)

    os.makedirs("Data", exist_ok=True)
    df.to_pickle(CACHE_PATH)
    elapsed = (time.time() - t0) / 60
    print(f"  Built & cached {len(df):,} segments in {elapsed:.1f} min.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial, tuning_df, config):
    """
    Called by Optuna once per trial.
    Suggests hyperparameters → builds fresh 3D tensor → trains → returns val_loss.
    """
    # 1. Suggest hyperparameters
    #    sequence_length: how many past swings the LSTM sees.
    #    Range 3-15 covers all practical lookbacks (>15 swings ≈ several months of history,
    #    which dilutes the signal and grows the model input — rarely helpful for swing trading).
    sequence_length = trial.suggest_int("sequence_length", 3, 15)
    learning_rate   = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    dropout_rate    = trial.suggest_float("dropout_rate", 0.10, 0.45)
    hidden_units    = trial.suggest_categorical("hidden_units", [32, 64, 96, 128])

    trial_model_cfg = {
        'learning_rate': learning_rate,
        'dropout_rate':  dropout_rate,
        'hidden_units':  hidden_units,
    }

    # 2. Build 3D sliding-window tensor for this specific sequence_length.
    #    Fast: only numpy slicing on the already-cached 2D DataFrame.
    builder = DatasetBuilder(None)
    X, y, _ = builder._create_sliding_windows(tuning_df, sequence_length)

    if len(X) < 500:
        return 1e9  # degenerate — skip

    # 3. Chronological 80/20 split (NOT random — the future must be held out)
    split_idx = int(len(X) * 0.80)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 4. Fresh randomly-initialized model for this trial's hyperparameters
    model = SwingLabLSTM(
        sequence_length=sequence_length,
        num_features=X_train.shape[2],
        model_config=trial_model_cfg
    )

    # 5. Pruning callback — Optuna kills trials that are clearly worse than
    #    the median of all previous trials after the warmup window.
    from optuna.integration import TFKerasPruningCallback
    pruning_cb = TFKerasPruningCallback(trial, 'val_loss')

    try:
        history = model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.0,
            callbacks=[pruning_cb]      # EarlyStopping is added inside model.train()
        )
        return min(history.history['val_loss'])

    except optuna.exceptions.TrialPruned:
        raise   # let Optuna handle it
    except Exception as e:
        print(f"\n  [Trial {trial.number}] ERROR: {e}")
        return 1e9


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_tuning(n_trials=25, force_rebuild=False):
    print("\n" + "=" * 60)
    print("  SWINGLAB MULTI-TARGET AUTO-TUNER")
    print("=" * 60)
    print(f"  Trials       : {n_trials}")
    print(f"  Stocks       : Top {TOP_N_STOCKS} by segment count")
    print(f"  Epochs/trial : {MAX_EPOCHS} (EarlyStopping patience={PATIENCE})")
    print(f"  Batch size   : {BATCH_SIZE}")
    print("=" * 60)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    import glob
    print("\n  [Auto-Cleanup] Removing old dataset and model files because tuning will invalidate them...")
    for f in glob.glob("Data/*train*.npy") + glob.glob("models/*.h5"):
        try: os.remove(f)
        except OSError: pass

    # ── Step 1: Get processed data (cache aware) ──────────────────────────
    print("\n[1/3] Preparing 2D feature data...")
    processed_df = get_processed_df(config, force_rebuild=force_rebuild)

    # ── Step 2: Sample top N stocks ───────────────────────────────────────
    print(f"\n[2/3] Selecting top {TOP_N_STOCKS} stocks...")
    symbol_counts = processed_df['symbol'].value_counts()
    top_symbols   = symbol_counts.head(TOP_N_STOCKS).index
    tuning_df     = processed_df[processed_df['symbol'].isin(top_symbols)].copy()
    n_segs        = len(tuning_df)
    steps_per_ep  = int(n_segs * 0.80) // BATCH_SIZE
    est_full_s    = steps_per_ep * MAX_EPOCHS * 0.005
    est_total_min = est_full_s * n_trials * 0.65 / 60  # ~35% pruned early
    print(f"  {n_segs:,} segments | ~{steps_per_ep} steps/epoch")
    print(f"  Estimated training time: ~{est_total_min:.0f} min "
          f"({est_full_s:.0f}s/trial, ~35% pruned early)")

    # ── Step 3: Optuna study ──────────────────────────────────────────────
    print(f"\n[3/3] Running {n_trials} Optuna trials...\n")
    t_start = time.time()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,   # explore freely for the first 5 trials
            n_warmup_steps=4,     # give each trial 4 epochs before pruning
            interval_steps=1
        )
    )

    completed_count = [0]

    def trial_callback(study, trial):
        completed_count[0] += 1
        elapsed   = time.time() - t_start
        avg_s     = elapsed / completed_count[0]
        remaining = max(0, (n_trials - completed_count[0]) * avg_s)

        if trial.state == optuna.trial.TrialState.PRUNED:
            status = "PRUNED"
            val_str = ""
        else:
            status  = "OK"
            val_str = f"loss={trial.value:.5f}  "

        try:
            best_seq = study.best_params.get('sequence_length', '?')
            best_val = f"{study.best_value:.5f}"
            best_str = f"best={best_val} @ seq={best_seq}"
        except Exception:
            best_str = "no best yet"

        print(f"  Trial {completed_count[0]:>3}/{n_trials}  [{status:<6}]  "
              f"{val_str:<20}  {best_str}  ETA {remaining/60:.1f}min")

    study.optimize(
        lambda trial: objective(trial, tuning_df, config),
        n_trials=n_trials,
        callbacks=[trial_callback]
    )

    # ── Print results ─────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned   = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.PRUNED])

    best = study.best_params

    print("\n" + "=" * 60)
    print("  TUNING COMPLETE!")
    print("=" * 60)
    print(f"  Total time  : {elapsed_total/60:.1f} minutes")
    print(f"  Completed   : {n_complete} trials  |  Pruned early: {n_pruned}")
    print(f"  Best loss   : {study.best_value:.6f}")
    print()
    print("  WINNER CONFIGURATION")
    print("  ─────────────────────────────────────────────")
    print(f"    sequence_length : {best['sequence_length']}")
    print(f"    learning_rate   : {best['learning_rate']:.6f}")
    print(f"    dropout_rate    : {best['dropout_rate']:.2f}")
    print(f"    hidden_units    : {best['hidden_units']}")
    print()
    print("  Paste this into config.yaml -> model2 block:")
    print("  ─────────────────────────────────────────────")
    print(f"  model2:")
    print(f"    sequence_length: {best['sequence_length']}")
    print(f"    learning_rate: {best['learning_rate']:.6f}")
    print(f"    dropout_rate: {best['dropout_rate']:.2f}")
    print(f"    hidden_units: {best['hidden_units']}")
    print()
    print("  Next steps to apply these results:")
    print("    1. Update model2 block in config.yaml")
    print("    2. python run_pipeline.py  ->  Step 4  (rebuild dataset)")
    print("    3. python run_pipeline.py  ->  Step 5  (retrain model)")
    print("=" * 60)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwingLab Hyperparameter Tuner")
    parser.add_argument(
        "--trials", type=int, default=25,
        help="Number of Optuna trials (default: 25, ~30 min)"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force feature cache rebuild (use after re-running the pipeline)"
    )
    args = parser.parse_args()

    run_tuning(n_trials=args.trials, force_rebuild=args.rebuild)
