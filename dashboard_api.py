import os
import sqlite3
import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import yaml

app = Flask(__name__, static_folder='dashboard', static_url_path='')
CORS(app)

# Load configuration for DB Path
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
DB_PATH = config['database']['path']

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def serve_index():
    return send_from_directory('dashboard', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('dashboard', path)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM basic_model_tests", conn)
    conn.close()

    if df.empty:
        return jsonify({"error": "No data available"}), 404

    graded = df[df['status'] == 'GRADED']
    pending_count = len(df[df['status'] == 'PENDING'])

    if graded.empty:
        return jsonify({"message": "No graded predictions yet", "pending_count": pending_count}), 200

    correct_dir = np.sum(np.sign(graded['predicted_return']) == np.sign(graded['actual_return']))
    win_rate = int(correct_dir) / len(graded)
    mae_ret = np.mean(np.abs(graded['predicted_return'] - graded['actual_return']))

    winners = graded[np.sign(graded['predicted_return']) == np.sign(graded['actual_return'])]
    losers = graded[np.sign(graded['predicted_return']) != np.sign(graded['actual_return'])]

    avg_win = winners['actual_return'].mean() if not winners.empty else 0.0
    avg_loss = losers['actual_return'].mean() if not losers.empty else 0.0

    long_trades = graded[graded['predicted_return'] > 0]
    short_trades = graded[graded['predicted_return'] < 0]
    
    theoretical_profit_long = long_trades['actual_return'].sum() * 100 if not long_trades.empty else 0
    theoretical_profit_short = (short_trades['actual_return'] * -1).sum() * 100 if not short_trades.empty else 0

    # For chart mapping: Cumulative profit over time
    graded = graded.sort_values(by='base_t_end')
    chart_data = []
    
    cum_ideal_long = 0
    cum_ideal_short = 0
    cum_execution = 0 # Combined realistic lagged P&L
    
    for _, row in graded.iterrows():
        # Ideal Oracle Streams
        if row['predicted_return'] > 0:
            cum_ideal_long += float(row['actual_return']) * 100
        elif row['predicted_return'] < 0:
            # Profit on short is -1 * actual return
            cum_ideal_short += float(row['actual_return']) * -100
            
        # Execution Stream (Combined realistically if not skipped)
        if pd.notnull(row['delayed_entry_price']) and pd.notnull(row['delayed_actual_exit_price']):
            # If target hit early, profit is 0 (skipped)
            if row.get('was_target_hit_early', 0) == 1:
                pass 
            else:
                # Calculate return based on direction
                if row['predicted_return'] > 0: # Long
                    trade_ret = (row['delayed_actual_exit_price'] - row['delayed_entry_price']) / row['delayed_entry_price']
                else: # Short
                    trade_ret = (row['delayed_entry_price'] - row['delayed_actual_exit_price']) / row['delayed_entry_price']
                cum_execution += float(trade_ret) * 100
                
        chart_data.append({
            'date': str(row['base_t_end']),
            'cum_long': cum_ideal_long,
            'cum_short': cum_ideal_short,
            'cum_execution': cum_execution
        })

    # Timing Accuracy: How many predictions had a duration error < 50% of the prediction?
    duration_errors = np.abs(graded['predicted_duration'] - graded['actual_duration'])
    time_accurate = np.sum(duration_errors < (graded['predicted_duration'] * 0.5))
    time_acc_rate = int(time_accurate) / len(graded) if len(graded) > 0 else 0

    stats = {
        "total_graded": len(graded),
        "win_rate": float(win_rate),
        "time_acc": float(time_acc_rate),
        "mae": float(mae_ret),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_long": float(theoretical_profit_long),
        "profit_short": float(theoretical_profit_short),
        "pending_count": pending_count,
        "chart_data": chart_data
    }
    return jsonify(stats)

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM basic_model_tests ORDER BY base_t_end DESC", conn)
    conn.close()
    
    # Replace NaN with None so JSON serialization works
    df = df.replace({np.nan: None})
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  SWINGLAB DASHBOARD ENGINE INITIALIZED")
    print("  -> http://localhost:5000")
    print("="*60 + "\n")
    # Turn off reloader so it doesn't double-initialize in terminal
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
