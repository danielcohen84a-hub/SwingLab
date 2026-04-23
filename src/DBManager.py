import sqlite3
import pandas as pd
from datetime import datetime
import os

class DBManager:
    def __init__(self, db_path="Data/SwingLabDB.sqlite"):
        """
        Initializes the DB connection and ensures tables exist.
        """
        # Ensure directory exists if there is one in the path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        """Returns a new connection to the SQLite database with WAL and Timeout to prevent locks."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _create_tables(self):
        """Creates the necessary tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. stocks_meta_data Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stocks_meta_data (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT,
                    industry TEXT,
                    exchange TEXT,
                    market_cap_category TEXT,
                    beta REAL,
                    average_volume_90_days REAL,
                    has_dividend TEXT,
                    ipo_year TEXT
                )
            ''')
            
            # 2. market_indicators Table
            # Stores raw close prices for all 5 macro ETF proxies.
            # Transformations (log returns, z-scores) are computed in FeatureEngineer.py.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME UNIQUE,
                    spy_close  REAL,
                    vxx_close  REAL,
                    uso_close  REAL,
                    uup_close  REAL,
                    ief_close  REAL
                )
            ''')
            
            # Migrate old schema if columns are missing (safe for existing DBs)
            existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(market_indicators)").fetchall()]
            for col, typ in [('spy_close','REAL'),('vxx_close','REAL'),('uso_close','REAL'),('uup_close','REAL'),('ief_close','REAL')]:
                if col not in existing_cols:
                    cursor.execute(f'ALTER TABLE market_indicators ADD COLUMN {col} {typ}')
            # Remove old columns from new inserts (SQLite can't DROP columns easily, just leave them)
            
            # 3. load_tracking Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS load_tracking (
                    load_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    window_start DATETIME,
                    window_end DATETIME,
                    loaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 4. segments Table
            # Stores the mathematical shape features of each local coordinate price swing.
            # CHECK: If swing_return is missing, we drop/recreate to migrate to Universal Geometry schema.
            seg_cols = [row[1] for row in cursor.execute('PRAGMA table_info(segments)').fetchall()]
            if seg_cols and 'swing_return' not in seg_cols:
                print("MIGRATION: Dropping old segments table (legacy schema detected)...")
                cursor.execute('DROP TABLE segments')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    t_start DATETIME,
                    t_end DATETIME,
                    -- Geometric
                    swing_return REAL,
                    duration_hours INTEGER,
                    slope_pct_per_hour REAL,
                    -- Quality
                    residual_volatility_pct REAL,
                    -- Volume
                    avg_hourly_volume REAL,
                    -- Momentum
                    rsi_start REAL,
                    rsi_end REAL,
                    -- Temporal
                    hour_of_day INTEGER,
                    day_of_week INTEGER
                )
            ''')
            
            # 5. raw_stock_data Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_stock_data (
                    symbol TEXT,
                    datetime DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, datetime)
                )
            ''')
            
            # (Removed basic_model_tests table creation)

            # 6. predictions Table
            # One row per live prediction issued by the daily runner.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker                 TEXT    NOT NULL,
                    -- When the script actually ran and made this prediction (real clock time)
                    predicted_at           DATETIME NOT NULL,
                    -- The confirmed extremum that ends the context window (= t_end of last segment)
                    entry_extremum_time    DATETIME NOT NULL,
                    -- First trading bar AFTER entry_extremum_time: trade entry time
                    segment_start_time     DATETIME NOT NULL,
                    -- Open price of the segment_start_time bar: realistic entry price
                    price_at_prediction    REAL    NOT NULL,
                    predicted_return       REAL    NOT NULL,
                    predicted_duration_bars INTEGER NOT NULL,
                    price_target           REAL    NOT NULL,
                    -- 1 = first-run back-prediction (bootstrap), 0 = live daily prediction
                    is_bootstrap           INTEGER DEFAULT 0
                )
            ''')

            # 7. prediction_results Table
            # One row per graded prediction; joined to predictions via prediction_id.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_results (
                    result_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id         INTEGER NOT NULL,
                    ticker                TEXT    NOT NULL,
                    actual_return         REAL    NOT NULL,
                    actual_duration_bars  INTEGER NOT NULL,
                    -- 1 if sign(actual_return) == sign(predicted_return)
                    direction_correct     INTEGER NOT NULL,
                    -- predicted_return - actual_return
                    return_error          REAL    NOT NULL,
                    -- predicted_duration_bars - actual_duration_bars
                    duration_error        INTEGER NOT NULL,
                    -- 1 if max(high) >= price_target (bull) or min(low) <= price_target (bear)
                    target_was_hit        INTEGER NOT NULL,
                    graded_at             DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                )
            ''')

            conn.commit()

    def upsert_stock_metadata(self, symbol, company_name=None, sector=None, industry=None, exchange=None, 
                              market_cap_category=None, beta=None, average_volume_90_days=None, 
                              has_dividend=None, ipo_year=None):
        """
        Inserts new metadata or updates it if the stock already exists.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO stocks_meta_data (
                    symbol, company_name, sector, industry, exchange, market_cap_category, 
                    beta, average_volume_90_days, has_dividend, ipo_year
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    company_name=excluded.company_name,
                    sector=excluded.sector,
                    industry=excluded.industry,
                    exchange=excluded.exchange,
                    market_cap_category=excluded.market_cap_category,
                    beta=excluded.beta,
                    average_volume_90_days=excluded.average_volume_90_days,
                    has_dividend=excluded.has_dividend,
                    ipo_year=excluded.ipo_year
            ''', (symbol, company_name, sector, industry, exchange, market_cap_category, 
                  beta, average_volume_90_days, has_dividend, ipo_year))
            conn.commit()

    def insert_market_indicators(self, timestamp, spy_close, vxx_close, uso_close, uup_close, ief_close):
        """
        Inserts a single market indicator row with raw ETF close prices.
        """
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR IGNORE INTO market_indicators
                    (timestamp, spy_close, vxx_close, uso_close, uup_close, ief_close)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, spy_close, vxx_close, uso_close, uup_close, ief_close))
            conn.commit()

    def save_market_indicators(self, df):
        """
        Bulk-saves market indicators from a DataFrame into the database.
        Uses a temp table to handle deduplication and schema alignment.
        """
        if df is None or df.empty:
            return
            
        # Ensure we only have the columns the DB expects and that actually exist in the df
        db_cols = ['timestamp', 'spy_close', 'vxx_close', 'uso_close', 'uup_close', 'ief_close']
        df_cols = [c for c in db_cols if c in df.columns]
        df_save = df[df_cols].copy()
        
        with self._get_connection() as conn:
            # Save to temporary table first
            df_save.to_sql('market_indicators_temp', conn, if_exists='replace', index=False)
            
            # Upsert into final table using only columns we have
            conn.execute(f'''
                INSERT OR REPLACE INTO market_indicators ({", ".join(df_cols)})
                SELECT {", ".join(df_cols)} FROM market_indicators_temp
            ''')
            conn.commit()

    def insert_raw_data(self, symbol, df):
        """
        Inserts raw OHLCV DataFrame into the DB, ignoring duplicates automatically.
        """
        if df is None or df.empty:
            return
            
        df_insert = df.copy()
        df_insert['symbol'] = symbol
        
        rename_map = {
            'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        df_insert = df_insert.rename(columns=rename_map)
        
        db_columns = ['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        df_insert = df_insert[db_columns]
        
        with self._get_connection() as conn:
            df_insert.to_sql('raw_stock_data_temp', conn, if_exists='replace', index=False)
            
            conn.execute('''
                INSERT INTO raw_stock_data (symbol, datetime, open, high, low, close, volume)
                SELECT symbol, datetime, open, high, low, close, volume 
                FROM raw_stock_data_temp
                WHERE True
                ON CONFLICT(symbol, datetime) DO NOTHING
            ''')
            conn.commit()

    def insert_load_tracking(self, symbol, window_start, window_end):
        """Records a successful data fetch window."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO load_tracking (symbol, window_start, window_end)
                VALUES (?, ?, ?)
            ''', (symbol, window_start, window_end))
            conn.commit()

    def insert_segments(self, symbol, segments_df):
        """
        Bulk-inserts all segments for a single stock into the DB.
        """
        if segments_df is None or segments_df.empty:
            return

        df_to_insert = segments_df.copy()
        df_to_insert['symbol'] = symbol

        # Ensure we only insert columns that exist in the database
        db_columns = [
            'symbol', 't_start', 't_end', 
            'swing_return', 'duration_hours', 'slope_pct_per_hour',
            'residual_volatility_pct', 'avg_hourly_volume', 
            'rsi_start', 'rsi_end', 'hour_of_day', 'day_of_week'
        ]
        
        # Clean up column list to only what's available
        df_to_insert = df_to_insert[[c for c in db_columns if c in df_to_insert.columns]]

        with self._get_connection() as conn:
            df_to_insert.to_sql('segments', conn, if_exists='append', index=False)
            
    def get_segments(self, symbol=None):
        """Returns all segments, optionally filtered by symbol. No JOIN needed."""
        query = 'SELECT * FROM segments'
        params = []
        if symbol:
            query += ' WHERE symbol = ?'
            params.append(symbol)
        query += ' ORDER BY symbol, t_start ASC'
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params,
                                     parse_dates=['t_start', 't_end'])

    def get_market_indicators(self):
        """Returns all raw ETF close prices from market_indicators."""
        with self._get_connection() as conn:
            return pd.read_sql_query(
                'SELECT * FROM market_indicators ORDER BY timestamp ASC',
                conn, parse_dates=['timestamp']
            )

    def get_stock_metadata(self, columns='symbol, sector, beta, market_cap_category'):
        """Returns stock metadata columns most useful for feature engineering."""
        with self._get_connection() as conn:
            return pd.read_sql_query(f'SELECT {columns} FROM stocks_meta_data', conn)

    def get_raw_stock_data(self, symbol):
        """
        Returns the full OHLCV history for a single symbol as a DataFrame
        with a DatetimeIndex and capitalised column names (Open, High, Low, Close, Volume)
        matching DataProcessor expectations.
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                'SELECT datetime, open, high, low, close, volume '
                'FROM raw_stock_data WHERE symbol = ? ORDER BY datetime ASC',
                conn, params=[symbol], parse_dates=['datetime']
            )
        df = df.rename(columns={
            'datetime': 'Datetime', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        })
        return df.set_index('Datetime')

    def get_loaded_symbols(self):
        """Returns the set of all symbols with completed raw data downloads."""
        with self._get_connection() as conn:
            df = pd.read_sql_query('SELECT DISTINCT symbol FROM load_tracking', conn)
        return set(df['symbol'].tolist())

    def get_segmented_symbols(self):
        """Returns the set of symbols that already have segments (used for resume logic)."""
        with self._get_connection() as conn:
            df = pd.read_sql_query('SELECT DISTINCT symbol FROM segments', conn)
        return set(df['symbol'].tolist())

    def count_market_indicators(self):
        """Returns the total number of rows in market_indicators."""
        with self._get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM market_indicators').fetchone()[0]

    def count_stocks_in_universe(self):
        """Returns the total number of stocks in stocks_meta_data."""
        with self._get_connection() as conn:
            return conn.execute('SELECT COUNT(*) FROM stocks_meta_data').fetchone()[0]

    def count_segments(self):
        """Returns (total_segment_rows, distinct_symbol_count) from the segments table."""
        with self._get_connection() as conn:
            seg_count = conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]
            sym_count = conn.execute('SELECT COUNT(DISTINCT symbol) FROM segments').fetchone()[0]
        return seg_count, sym_count

    def get_universe_symbols(self):
        """Returns an ordered list of all symbols in the stock universe."""
        with self._get_connection() as conn:
            df = pd.read_sql_query('SELECT symbol FROM stocks_meta_data ORDER BY symbol', conn)
        return df['symbol'].tolist()

    # ─────────────────────────────────────────────────────────────────
    # Predictions & Results
    # ─────────────────────────────────────────────────────────────────

    def insert_prediction(self, ticker, predicted_at, entry_extremum_time, segment_start_time,
                          price_at_prediction, predicted_return, predicted_duration_bars,
                          price_target, is_bootstrap=0):
        """
        Inserts a new prediction row and returns the auto-generated prediction_id.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions
                    (ticker, predicted_at, entry_extremum_time, segment_start_time,
                     price_at_prediction, predicted_return, predicted_duration_bars,
                     price_target, is_bootstrap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                str(predicted_at),
                str(entry_extremum_time),
                str(segment_start_time),
                float(price_at_prediction),
                float(predicted_return),
                int(predicted_duration_bars),
                float(price_target),
                int(is_bootstrap),
            ))
            conn.commit()
            return cursor.lastrowid

    def insert_prediction_result(self, prediction_id, ticker, actual_return, actual_duration_bars,
                                 direction_correct, return_error, duration_error, target_was_hit):
        """Inserts a graded result row for a completed prediction."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO prediction_results
                    (prediction_id, ticker, actual_return, actual_duration_bars,
                     direction_correct, return_error, duration_error, target_was_hit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(prediction_id),
                ticker,
                float(actual_return),
                int(actual_duration_bars),
                int(direction_correct),
                float(return_error),
                int(duration_error),
                int(target_was_hit),
            ))
            conn.commit()

    def get_open_predictions(self):
        """Returns all predictions that have not yet been graded (no matching result row)."""
        with self._get_connection() as conn:
            return pd.read_sql_query('''
                SELECT * FROM predictions
                WHERE prediction_id NOT IN (SELECT prediction_id FROM prediction_results)
                ORDER BY predicted_at ASC
            ''', conn)

    def get_tracked_tickers(self):
        """Returns the distinct list of tickers currently being tracked (from predictions table)."""
        with self._get_connection() as conn:
            df = pd.read_sql_query('SELECT DISTINCT ticker FROM predictions ORDER BY ticker', conn)
        return df['ticker'].tolist()

    def get_open_prediction_tickers(self):
        """Returns tickers that currently have at least one ungraded open prediction."""
        with self._get_connection() as conn:
            df = pd.read_sql_query('''
                SELECT DISTINCT ticker FROM predictions
                WHERE prediction_id NOT IN (SELECT prediction_id FROM prediction_results)
            ''', conn)
        return df['ticker'].tolist()

    def is_predictions_empty(self):
        """Returns True if no predictions have been issued yet (signals first-run bootstrap)."""
        with self._get_connection() as conn:
            count = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        return count == 0

    def get_latest_prediction_for_ticker(self, ticker):
        """
        Returns the most recently issued prediction for a ticker as a dict.
        Returns None if the ticker has no predictions.
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query('''
                SELECT * FROM predictions WHERE ticker = ?
                ORDER BY predicted_at DESC LIMIT 1
            ''', conn, params=[ticker])
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_prediction_by_id(self, prediction_id):
        """Returns a single prediction row as a dict, or None if not found."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                'SELECT * FROM predictions WHERE prediction_id = ?',
                conn, params=[int(prediction_id)]
            )
        if df.empty:
            return None
        return df.iloc[0].to_dict()
