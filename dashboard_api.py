from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yaml
import pandas as pd
from datetime import timedelta
from typing import Dict, Any

from src.DBManager import DBManager
from src.DataProcessor import SwingProcessor

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
      - context_zigzag: the last 11 swing segments used for prediction
      - other_zigzag: the rest of the recent swing segments
      - prediction: the raw prediction fields
    """
    db = get_db_manager()

    # 1. Fetch the prediction row
    pred = db.get_prediction_by_id(prediction_id)
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    ticker = pred['ticker']
    try:
        anchor = pd.to_datetime(pred['entry_extremum_time'])
    except Exception:
        anchor = pd.to_datetime(pred['segment_start_time'])

    # 2. Fetch full raw data to calculate accurate segments
    raw_df = db.get_raw_stock_data(ticker)
    if raw_df.empty:
        raise HTTPException(status_code=404, detail="No raw data found for ticker")
        
    if 'Datetime' not in raw_df.columns and raw_df.index.name == 'Datetime':
        raw_df = raw_df.reset_index()

    # 3. Generate segments
    processor = SwingProcessor()
    segs_df = processor.generate_segments(raw_df)
    
    if not segs_df.empty:
        segs_df['t_start'] = pd.to_datetime(segs_df['t_start'])
        segs_df['t_end'] = pd.to_datetime(segs_df['t_end'])

    # 4. Find the context segments (last 11 ending at or before anchor)
    if not segs_df.empty:
        context_segs = segs_df[segs_df['t_end'] <= anchor].tail(11)
    else:
        context_segs = pd.DataFrame()

    if not context_segs.empty:
        first_context_t = pd.to_datetime(context_segs.iloc[0]['t_start'])
        window_start = first_context_t - timedelta(days=5)
    else:
        window_start = anchor - timedelta(weeks=6)

    window_end = anchor + timedelta(hours=pred['predicted_duration_bars'] * 1.5 + 100)

    # 5. Filter candles to window
    raw_df['Datetime'] = pd.to_datetime(raw_df['Datetime'])
    candles_df = raw_df[(raw_df['Datetime'] >= window_start) & (raw_df['Datetime'] <= window_end)]

    # 6. Build zig-zag points
    context_zigzag = []
    for _, seg in context_segs.iterrows():
        if not context_zigzag:
            context_zigzag.append({
                "t": seg['t_start'].isoformat(),
                "price": float(seg['price_start'])
            })
        context_zigzag.append({
            "t": seg['t_end'].isoformat(),
            "price": float(seg['price_end'])
        })

    other_zigzag = []
    if not segs_df.empty:
        # Future segments (after anchor)
        future_segs = segs_df[(segs_df['t_start'] >= anchor) & (segs_df['t_start'] <= window_end)]
        for _, seg in future_segs.iterrows():
            if not other_zigzag:
                other_zigzag.append({
                    "t": seg['t_start'].isoformat(),
                    "price": float(seg['price_start'])
                })
            other_zigzag.append({
                "t": seg['t_end'].isoformat(),
                "price": float(seg['price_end'])
            })

    # 7. Serialize candles
    candles_out = [
        {
            "t": row['Datetime'].isoformat(),
            "o": float(row['Open']),
            "h": float(row['High']),
            "l": float(row['Low']),
            "c": float(row['Close']),
        }
        for _, row in candles_df.iterrows()
    ]

    return {
        "ticker": ticker,
        "candles": candles_out,
        "context_zigzag": context_zigzag,
        "other_zigzag": other_zigzag,
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
