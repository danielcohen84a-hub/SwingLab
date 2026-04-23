from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yaml
import pandas as pd
from datetime import timedelta
from typing import Dict, Any

from src.DBManager import DBManager

app = FastAPI(title="SwingLab Dashboard API")

# Allow Vite frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_manager() -> DBManager:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return DBManager(config['database']['path'])

@app.get("/api/predictions")
def get_predictions() -> Dict[str, Any]:
    """Returns a joined table of predictions, their results, and metadata."""
    db = get_db_manager()
    with db._get_connection() as conn:
        # Join predictions with their results (LEFT JOIN since open predictions have no results yet)
        # Also join stocks_meta_data to get the sector for filtering
        query = '''
            SELECT 
                p.prediction_id,
                p.ticker,
                m.sector,
                p.predicted_at,
                p.segment_start_time,
                p.price_at_prediction,
                p.predicted_return,
                p.predicted_duration_bars,
                p.price_target,
                p.is_bootstrap,
                r.actual_return,
                r.actual_duration_bars,
                r.direction_correct,
                r.return_error,
                r.target_was_hit,
                r.graded_at
            FROM predictions p
            LEFT JOIN prediction_results r ON p.prediction_id = r.prediction_id
            LEFT JOIN stocks_meta_data m ON p.ticker = m.symbol
            ORDER BY p.predicted_at DESC
        '''
        df = pd.read_sql_query(query, conn)
    
    # Handle NaNs from LEFT JOIN so JSON serialization works
    df = df.fillna("")
    
    # Return as list of dictionaries
    return {"data": df.to_dict(orient="records")}

@app.get("/api/summary")
def get_summary() -> Dict[str, Any]:
    """Returns top-level metrics for the dashboard (Win Rate)."""
    db = get_db_manager()
    with db._get_connection() as conn:
        df = pd.read_sql_query('SELECT direction_correct FROM prediction_results', conn)
    
    if df.empty:
        return {"win_rate": 0.0}
        
    wins = (df['direction_correct'] == 1).sum()
    win_rate = float(wins / len(df))
    
    return {"win_rate": win_rate}

@app.get("/api/chart/{prediction_id}")
def get_chart_data(prediction_id: int) -> Dict[str, Any]:
    """
    Returns all data needed to render a PlotlyOracle-style chart for a single prediction:
      - candles: windowed OHLCV around the entry time
      - zigzag: swing segment zig-zag points
      - prediction: the raw prediction fields (entry price, return, duration, target)
    """
    db = get_db_manager()

    # 1. Fetch the prediction row
    pred = db.get_prediction_by_id(prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    ticker = pred['ticker']
    # Use entry_extremum_time as chart anchor; show 6 weeks before and predicted_duration_bars after
    try:
        anchor = pd.to_datetime(pred['entry_extremum_time'])
    except Exception:
        anchor = pd.to_datetime(pred['segment_start_time'])

    window_start = anchor - timedelta(weeks=6)
    window_end   = anchor + timedelta(hours=pred['predicted_duration_bars'] * 1.5 + 100)

    with db._get_connection() as conn:
        # 2. Fetch windowed OHLCV candles
        candles_df = pd.read_sql_query(
            '''SELECT datetime, open, high, low, close, volume
               FROM raw_stock_data
               WHERE symbol = ?
                 AND datetime >= ?
                 AND datetime <= ?
               ORDER BY datetime ASC''',
            conn,
            params=[ticker, str(window_start), str(window_end)],
        )

        # 3. Fetch swing segments in the same window
        segs_df = pd.read_sql_query(
            '''SELECT t_start, t_end, swing_return
               FROM segments
               WHERE symbol = ?
                 AND t_start >= ?
                 AND t_end   <= ?
               ORDER BY t_start ASC''',
            conn,
            params=[ticker, str(window_start), str(window_end)],
        )

    # 4. Build zig-zag price list – we need price_start / price_end.
    # They aren't stored in segments, but we can reconstruct them from candles.
    candles_df['datetime'] = pd.to_datetime(candles_df['datetime'])
    if not candles_df.empty and not segs_df.empty:
        # Map each segment endpoint to the closest candle close price
        segs_df['t_start'] = pd.to_datetime(segs_df['t_start'])
        segs_df['t_end']   = pd.to_datetime(segs_df['t_end'])

        def nearest_close(ts):
            idx = (candles_df['datetime'] - ts).abs().idxmin()
            return float(candles_df.loc[idx, 'close'])

        zigzag_points = []
        for _, seg in segs_df.iterrows():
            if not zigzag_points:
                zigzag_points.append({
                    "t": seg['t_start'].isoformat(),
                    "price": nearest_close(seg['t_start'])
                })
            zigzag_points.append({
                "t": seg['t_end'].isoformat(),
                "price": nearest_close(seg['t_end'])
            })
    else:
        zigzag_points = []

    # 5. Serialize candles
    candles_out = [
        {
            "t": row['datetime'].isoformat(),
            "o": row['open'],
            "h": row['high'],
            "l": row['low'],
            "c": row['close'],
        }
        for _, row in candles_df.iterrows()
    ]

    return {
        "ticker": ticker,
        "candles": candles_out,
        "zigzag": zigzag_points,
        "prediction": {
            "entry_time":        str(pred.get('segment_start_time', pred.get('entry_extremum_time'))),
            "entry_price":       pred['price_at_prediction'],
            "predicted_return":  pred['predicted_return'],
            "predicted_duration_bars": pred['predicted_duration_bars'],
            "price_target":      pred['price_target'],
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard_api:app", host="0.0.0.0", port=8000, reload=True)
