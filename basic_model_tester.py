import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.DBManager import DBManager
from src.DataDownloader import DataDownloader
from src.DataProcessor import SwingProcessor
from src.FeatureEngineer import FeatureEngineer
from src.LSTMModel import SwingLabLSTM


def run_tester():
    print("\n" + "="*60)
    print("  SWINGLAB: BASIC MODEL TESTER (Asynchronous Walk-Forward)")
    print("="*60)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    SEQ_LEN = config.get('model2', {}).get('sequence_length', 5)
    EXTREMA_ORDER = config['processing']['extrema_order']
    DB_PATH = config['database']['path']
    POLY_KEY = config['api']['polygon_key']

    db = DBManager(DB_PATH)
    downloader = DataDownloader(POLY_KEY)
    processor = SwingProcessor(extrema_order=EXTREMA_ORDER)
    engineer = FeatureEngineer(mode='live', indicator_windows=config['indicator_windows_months'])
    
    # Update Macro Indicators using Smart Caching
    print(f"Syncing Market Indicators...")
    with db._get_connection() as conn:
        indicators_df = pd.read_sql("SELECT * FROM market_indicators ORDER BY timestamp ASC", conn, parse_dates=['timestamp'])
    latest_macro = indicators_df['timestamp'].max() if not indicators_df.empty else None
    
    # If stale by 12 hours, patch the delta
    if not latest_macro or (datetime.now() - pd.to_datetime(latest_macro)).total_seconds() > 12 * 3600:
        if latest_macro:
            start_date = pd.to_datetime(latest_macro)
            delta_df = downloader.fetch_market_indicators(start_date=start_date)
        else:
            delta_df = downloader.fetch_market_indicators(years_back=4)
            
        if not delta_df.empty:
            db.save_market_indicators(delta_df)
            
        # Reload fully patched DB
        with db._get_connection() as conn:
            indicators_df = pd.read_sql("SELECT * FROM market_indicators ORDER BY timestamp ASC", conn, parse_dates=['timestamp'])
            
    indicators_df = engineer.transform_market_indicators(indicators_df)

    # Dynamic Watchlist (Top 50 Tech, Top 20 all other sectors by Volume)
    with db._get_connection() as conn:
        all_meta_df = pd.read_sql("SELECT symbol, sector, average_volume_90_days FROM stocks_meta_data", conn)
    
    all_meta_df = all_meta_df.dropna(subset=['sector', 'average_volume_90_days']).sort_values('average_volume_90_days', ascending=False)
    
    symbols = []
    # Tech
    tech_df = all_meta_df[all_meta_df['sector'] == 'Technology'].head(50)
    symbols.extend(tech_df['symbol'].tolist())
    # Other sectors
    for sector in all_meta_df['sector'].unique():
        if sector == 'Technology': continue
        sec_df = all_meta_df[all_meta_df['sector'] == sector].head(20)
        symbols.extend(sec_df['symbol'].tolist())
        
    print(f"Tracking {len(symbols)} High-Liquidity symbols dynamically selected by sector...\n")
    
    model = None # Lazy load when we know num_features
    total_newly_graded = 0
    total_new_live = 0

    for symbol in symbols:
        # Fetch meta
        with db._get_connection() as conn:
            meta_df = pd.read_sql(f"SELECT sector, beta, market_cap_category FROM stocks_meta_data WHERE symbol = '{symbol}'", conn)
        if meta_df.empty: continue
        meta = meta_df.iloc[0].to_dict()

        # --- SMART CACHING ARCHITECTURE ---
        try:
            ticker_df_db = db.get_raw_stock_data(symbol)
            
            if ticker_df_db is None or ticker_df_db.empty:
                # Bootloader: Download 24 weeks to ensure enough swings exist
                print(f"  [{symbol}] New symbol. Bootloading 6 months of data...")
                delta_df = downloader.fetch_stock_data_fast(symbol, weeks=24)
                db.insert_raw_data(symbol, delta_df)
            else:
                max_date = ticker_df_db.index.max()
                if max_date.tzinfo is not None:
                     max_date = max_date.tz_localize(None)
                     
                # If older than 4 hours, fetch delta
                if (datetime.now() - max_date).total_seconds() > 4 * 3600:
                    print(f"  [{symbol}] Fetching delta since {max_date}...")
                    delta_df = downloader.fetch_stock_data_fast(symbol, start_date=max_date)
                    if not delta_df.empty:
                        print(f"  [{symbol}] Saved {len(delta_df)} new rows via Smart Cache.")
                        db.insert_raw_data(symbol, delta_df)
                    else:
                        print(f"  [{symbol}] Yfinance returned no new delta data.")
                        
            # Now safely read the full stitched history natively from DB
            ticker_df = db.get_raw_stock_data(symbol).reset_index()
            if ticker_df.empty: continue
            
            # Prevent In-Sample Leakage: Only pass the most recent 16 weeks max!
            cutoff_date = datetime.now() - timedelta(weeks=16)
            ticker_df = ticker_df[ticker_df['Datetime'] >= cutoff_date].reset_index(drop=True)
            
        except Exception as e:
            print(f"  [X] Skipping {symbol} due to download error: {e}")
            continue
            
        segments_df = processor.generate_segments(ticker_df)
        
        if len(segments_df) <= SEQ_LEN:
            continue
            
        # Join globals
        from src.DatasetBuilder import DatasetBuilder
        segments_df = DatasetBuilder(db)._aggregate_indicators_per_segment(segments_df, indicators_df)

        segments_df['sector'] = meta['sector']
        segments_df['beta'] = meta['beta']
        segments_df['market_cap_category'] = meta['market_cap_category']
        
        # Walk Forward
        OOS_CUTOFF = pd.to_datetime('2026-04-08 00:00:00')
        if segments_df['t_end'].max() < OOS_CUTOFF:
            continue # Skip stock if it hasn't formed any segments recently
            
        tests_df = db.get_basic_model_tests(symbol)
        
        for i in range(SEQ_LEN, len(segments_df) + 1):
            current_t_end = segments_df.iloc[i-1]['t_end']
            
            if current_t_end < OOS_CUTOFF:
                continue # Skip processing strictly in-sample historical data!
            
            existing_row = tests_df[tests_df['base_t_end'] == current_t_end]
            status = existing_row.iloc[0]['status'] if not existing_row.empty else None
            
            if status == 'GRADED':
                continue # Already fully evaluated in a previous run
                
            is_pending = (status == 'PENDING')
            
            if not is_pending:
                # Simulate the model at this exact point in time
                sub_df = segments_df.iloc[:i].copy()
                X_live = engineer.prepare_live_tensor(sub_df, sequence_length=SEQ_LEN)
                
                if model is None:
                    model = SwingLabLSTM(sequence_length=SEQ_LEN, num_features=X_live.shape[2], model_config=config.get('model2', {}))
                    model.load_weights('models/swinglab_lstm.weights.h5')

                    
                pred = model.model.predict(X_live, verbose=0)[0]
                pred_ret = float(pred[0])
                pred_dur = float(engineer.duration_scaler.inverse_transform([[pred[1]]])[0][0])
            else:
                pred_ret = float(existing_row.iloc[0]['predicted_return'])
                pred_dur = float(existing_row.iloc[0]['predicted_duration'])
                
            # --- DELAYED EXECUTION MATH (Common for Graded & Pending) ---
            delayed_entry_price = None
            delayed_actual_exit_price = None
            delayed_predicted_exit_price = None
            was_target_hit_early = 0
            
            # Use relative slicing instead of absolute index arithmetic for robustness
            after_anchor_df = ticker_df[ticker_df['Datetime'] >= current_t_end]
            
            if len(after_anchor_df) > EXTREMA_ORDER:
                # 1. Target Hit Early Check
                lag_window = after_anchor_df.iloc[:EXTREMA_ORDER+1]
                anchor_price = float(lag_window.iloc[0]['Close'])
                target_price = anchor_price * (1 + pred_ret)
                
                # Check if target was hit during lag (Entry Confirming phase)
                if pred_ret > 0: # Long
                    if lag_window['High'].max() >= target_price:
                        was_target_hit_early = 1
                else: # Short
                    if lag_window['Low'].min() <= target_price:
                        was_target_hit_early = 1

                # 2. Entry Price
                entry_row = after_anchor_df.iloc[EXTREMA_ORDER]
                delayed_entry_price = float(entry_row['Close'])
                delayed_entry_time = entry_row['Datetime']
                
                # If target was hit early, we theoretically skip this trade (Buy Price = Sell Price in simulator)
                if was_target_hit_early:
                    delayed_actual_exit_price = delayed_entry_price
                    delayed_predicted_exit_price = delayed_entry_price
                else:
                    # Actual Exit (Only for Graded)
                    if i < len(segments_df):
                        next_t_end = segments_df.iloc[i]['t_end']
                        exit_match = ticker_df[ticker_df['Datetime'] == next_t_end]
                        if not exit_match.empty:
                            delayed_actual_exit_price = float(exit_match.iloc[0]['Close'])
                            
                    # Predicted Exit: Anchor Time + Scaled Duration hours
                    pred_exit_time = current_t_end + timedelta(hours=pred_dur)
                    
                    # Only execute if the predicted exit occurs AFTER our delayed entry!
                    if pred_exit_time > delayed_entry_time:
                        # Find nearest price at or after target exit time
                        future_df = ticker_df[ticker_df['Datetime'] >= pred_exit_time]
                        if not future_df.empty:
                            delayed_predicted_exit_price = float(future_df.iloc[0]['Close'])
                    else:
                        # Aborted trade: Prediction lived shorter than confirmation lag.
                        delayed_predicted_exit_price = delayed_entry_price

            # Upsert into DB
            if i < len(segments_df):
                actual_ret = float(segments_df.iloc[i]['swing_return'])
                actual_dur = float(segments_df.iloc[i]['duration_hours'])
                
                db.upsert_basic_model_test(
                    symbol, current_t_end, pred_ret, pred_dur, actual_ret, actual_dur, 
                    delayed_entry_price, delayed_actual_exit_price, delayed_predicted_exit_price, 
                    was_target_hit_early, 'GRADED'
                )
                
                if is_pending or True: 
                    entry_str = f"${delayed_entry_price:.2f}" if delayed_entry_price else "LAGGING"
                    hit_early_tag = " [TARGET HIT EARLY]" if was_target_hit_early else ""
                    print(f"  [GRADED] {symbol}: Entry {entry_str} | Pred {pred_ret:+.2%}{hit_early_tag}")
                    total_newly_graded += 1
            else:
                if not is_pending:
                    db.upsert_basic_model_test(
                        symbol, current_t_end, pred_ret, pred_dur, 
                        None, None,
                        delayed_entry_price, None, delayed_predicted_exit_price, 
                        was_target_hit_early, 'PENDING'
                    )
                    entry_str = f"${delayed_entry_price:.2f}" if delayed_entry_price else "CONFIRMING"
                    hit_early_tag = " [TARGET HIT EARLY]" if was_target_hit_early else ""
                    print(f"[NEW LIVE] {symbol}: Pred {pred_ret:+.2%}. Entry: {entry_str}{hit_early_tag}")
                    total_new_live += 1

    # --- DASHBOARD SUMMARY ---
    print("\n" + "="*60)
    print("  BASIC MODEL TESTER DASHBOARD")
    print("="*60)
    
    with db._get_connection() as conn:
        all_graded = pd.read_sql("SELECT * FROM basic_model_tests WHERE status='GRADED'", conn)
        
    if all_graded.empty:
        print("Not enough completed historical segments in DB to generate metrics yet.")
        print("Come back later when pending targets finish!")
        return
        
    correct_dir = np.sum(np.sign(all_graded['predicted_return']) == np.sign(all_graded['actual_return']))
    win_rate = correct_dir / len(all_graded)
    mae_ret = np.mean(np.abs(all_graded['predicted_return'] - all_graded['actual_return']))
    
    # Profiling
    winners = all_graded[np.sign(all_graded['predicted_return']) == np.sign(all_graded['actual_return'])]
    losers = all_graded[np.sign(all_graded['predicted_return']) != np.sign(all_graded['actual_return'])]
    
    avg_win = winners['actual_return'].mean() if not winners.empty else 0.0
    avg_loss = losers['actual_return'].mean() if not losers.empty else 0.0
    
    # Theoretical Portfolio Tracker
    # Assume $100 starting capital per trade whenever model predicted a Positive return
    long_trades = all_graded[all_graded['predicted_return'] > 0]
    if len(long_trades) > 0:
        theoretical_profit = long_trades['actual_return'].sum() * 100
    else:
        theoretical_profit = 0

    print(f"Total Graded Predictions : {len(all_graded)}")
    print(f"Directional Accuracy     : {win_rate:.1%}")
    print(f"Avg Prediction Error     : {mae_ret:.2%}")
    print(f"")
    print(f"Metrics When Buying (Predicted > 0):")
    print(f"  Avg Winning Trade      : {avg_win:+.2%}")
    print(f"  Avg Losing Trade       : {avg_loss:+.2%}")
    print(f"  Theoretical P&L        : ${theoretical_profit:+.2f} (from $100 base stakes)")
    
    # Get total active pendings
    with db._get_connection() as conn:
        pending_count = conn.execute("SELECT COUNT(*) FROM basic_model_tests WHERE status='PENDING'").fetchone()[0]
    print(f"\n{pending_count} Active Predictions currently live in the market.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_tester()
