import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import yaml
import os
import glob
from src.DBManager import DBManager
from src.DataProcessor import SwingProcessor

def tune_extrema():
    # 1. Setup
    config = yaml.safe_load(open("config.yaml", "r"))
    db = DBManager(db_path=config['database']['path'])
    
    # 2. Results Folder Management
    output_dir = "TuningResults"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear old files inside to avoid confusion
    for f in glob.glob(os.path.join(output_dir, "*")):
        try:
            os.remove(f)
        except Exception:
            pass # Keep going if a file is locked
    
    # 3. Pick ONE random symbol that has data
    all_symbols = list(db.get_loaded_symbols())
    if not all_symbols:
        print("ERROR: No stock data found in DB. Run Step 2 first!")
        return
    
    symbol = random.choice(all_symbols)
    orders_to_test = [3, 5, 8, 12]
    
    print(f"Creating High-Res analysis for {symbol}...")
    
    # Load the last 300 hours
    raw_df = db.get_raw_stock_data(symbol).tail(300)
    if raw_df.empty:
        print(f"No data found for {symbol}.")
        return

    # HELPER: Plot a single view for shared use
    def plot_on_ax(ax, order=None, is_raw=False):
        # Dark Slate Grey for maximum visibility
        ax.plot(raw_df.index, raw_df['Close'], color='#434a54', alpha=0.9, label='Raw Price', linewidth=1.2)
        
        if not is_raw and order is not None:
            processor = SwingProcessor(extrema_order=order)
            segments_df = processor.generate_segments(raw_df)
            
            if not segments_df.empty:
                for _, row in segments_df.iterrows():
                    t0, t1 = row['t_start'], row['t_end']
                    p0 = raw_df.loc[t0, 'Close']
                    p1 = raw_df.loc[t1, 'Close']
                    color = 'forestgreen' if p1 > p0 else 'firebrick'
                    ax.plot([t0, t1], [p0, p1], color=color, linewidth=2.0, marker='o', markersize=5)
                ax.set_title(f"Sensitivity: {order} | {len(segments_df)} Swings", fontsize=20, fontweight='bold')
            else:
                ax.set_title(f"Sensitivity: {order} | NO SWINGS", fontsize=20, color='red')
        else:
            ax.set_title(f"RAW PRICE ONLY: {symbol}", fontsize=20, fontweight='bold')
            
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    # 4. Save the RAW ONLY file
    fig_raw, ax_raw = plt.subplots(figsize=(20, 10))
    plot_on_ax(ax_raw, is_raw=True)
    fig_raw.savefig(os.path.join(output_dir, f"0_Raw_Price_{symbol}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_raw)

    # 5. Save INDIVIDUAL sensitivity files
    for order in orders_to_test:
        fig_ind, ax_ind = plt.subplots(figsize=(20, 10))
        plot_on_ax(ax_ind, order=order)
        fig_ind.savefig(os.path.join(output_dir, f"Order_{order}_{symbol}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig_ind)

    # 6. Save the FULL COMPARISON
    fig_main, axes = plt.subplots(1, len(orders_to_test), figsize=(28, 12), sharey=True)
    fig_main.suptitle(f"SwingLab Sensitivity Analysis: {symbol}", fontsize=28, fontweight='bold', y=0.98)
    for idx, order in enumerate(orders_to_test):
        plot_on_ax(axes[idx], order=order)

    fig_main.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig_main.savefig(os.path.join(output_dir, f"Full_Comparison_{symbol}.png"), dpi=120, bbox_inches='tight')
    
    print(f"\n✓ ALL PHOTOS SAVED in {output_dir}/")
    print(f"✓ Stock analyzed: {symbol}")
    print("\nDisplaying comparison window...")
    plt.show()

if __name__ == "__main__":
    tune_extrema()
