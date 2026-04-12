import os
import yaml
import numpy as np
import pandas as pd
import optuna
from datetime import datetime

from src.DBManager import DBManager
from src.DatasetBuilder import DatasetBuilder
from src.LSTMModel import SwingLabLSTM


# Redirect Optuna logs to be cleaner
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, processed_df, config):
    # 1. Suggest Hyperparameters
    # We check a range of sequence lengths as requested
    sequence_length = trial.suggest_int("sequence_length", 3, 20)
    
    # Model Architecture & Training
    learning_rate   = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    dropout_rate    = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hidden_units    = trial.suggest_categorical("hidden_units", [32, 64, 96, 128])
    
    trial_config = config.copy()
    trial_config['model2'] = {
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'hidden_units': hidden_units
    }
    
    # 2. Build 3D Tensor for this sequence length (Fast because 2D is pre-processed)
    builder = DatasetBuilder(None)
    X, y, _ = builder._create_sliding_windows(processed_df, sequence_length)

    
    if len(X) < 1000:
        return 1e9 # Safety

    # 3. Chronological Split (Manual)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 4. Initialize and Train
    model = SwingLabLSTM(sequence_length, X_train.shape[2], trial_config.get('model2', {}))

    
    # Optuna Pruning: Stops the trial early if it's statistically hopeless
    from optuna.integration import TFKerasPruningCallback
    pruning_cb = TFKerasPruningCallback(trial, 'val_loss')
    
    try:
        history = model.train(
            X_train, y_train, 
            X_val=X_val, y_val=y_val,
            epochs=25, # 25 epochs is enough to see the dual-target trend
            batch_size=256,
            validation_split=0.0,
            callbacks=[pruning_cb]
        )
        val_loss = min(history.history['val_loss'])
        return val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return 1e9

def run_tuning(n_trials=20):
    print("\n" + "="*60)
    print("  SWINGLAB MULTI-TARGET AUTO-TUNER")

    print("="*60)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    db = DBManager(config['database']['path'])
    builder = DatasetBuilder(db)

    
    # 1. Prepare Data Once (Step 1-5: Heavy lifting)
    print("Preparing 2D features for all segments...")
    windows = config.get('indicator_windows_months', {})
    processed_df = builder.prepare_processed_df(indicator_windows=windows)
    
    # 2. Top-500 Stock Strategy (Speed-Accuracy Sweet Spot)
    print(f"Sampling top 500 stocks for 10x faster tuning...")
    symbol_counts = processed_df['symbol'].value_counts()
    top_symbols = symbol_counts.head(500).index
    tuning_df = processed_df[processed_df['symbol'].isin(top_symbols)].copy()
    
    print(f"Tuning Dataset: {len(tuning_df):,} segments across 500 stocks.")
    print(f"Starting {n_trials} optimization trials...")

    # 3. Create Study
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # Progress Bar
    from tqdm import tqdm
    with tqdm(total=n_trials) as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_description(f"Best Loss: {study.best_value:.6f}")
        
        study.optimize(lambda trial: objective(trial, tuning_df, config), n_trials=n_trials, callbacks=[callback])

    print("\n" + "="*60)
    print("  TUNING COMPLETE!")
    print("="*60)
    print(f"Best Huber Loss: {study.best_value:.6f}")
    print("\nWINNER CONFIGURATION (Put these in config.yaml):")
    print(f"model2:")
    print(f"  sequence_length: {study.best_params['sequence_length']}")
    print(f"  learning_rate: {study.best_params['learning_rate']:.6f}")
    print(f"  dropout_rate: {study.best_params['dropout_rate']:.2f}")
    print(f"  hidden_units: {study.best_params['hidden_units']}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    
    run_tuning(n_trials=args.trials)
