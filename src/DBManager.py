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
            
            # 6. basic_model_tests Table
            # Migrating to tracking execution delay logic + Hit Early check
            db_cols = [row[1] for row in cursor.execute('PRAGMA table_info(basic_model_tests)').fetchall()]
            if db_cols and 'was_target_hit_early' not in db_cols:
                print("MIGRATION: Dropping old basic_model_tests table to add hit_early schemas...")
                cursor.execute('DROP TABLE basic_model_tests')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS basic_model_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    base_t_end DATETIME,
                    predicted_return REAL,
                    predicted_duration REAL,
                    actual_return REAL,
                    actual_duration REAL,
                    delayed_entry_price REAL,
                    delayed_actual_exit_price REAL,
                    delayed_predicted_exit_price REAL,
                    was_target_hit_early INTEGER,
                    status TEXT,
                    UNIQUE(symbol, base_t_end)
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

    def upsert_basic_model_test(self, symbol, base_t_end, predicted_return, predicted_duration, 
                                actual_return=None, actual_duration=None, 
                                delayed_entry_price=None, delayed_actual_exit_price=None, delayed_predicted_exit_price=None,
                                was_target_hit_early=0, status='PENDING'):
        """Inserts a new prediction or updates an existing one when graded."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO basic_model_tests (
                    symbol, base_t_end, predicted_return, predicted_duration, actual_return, actual_duration, 
                    delayed_entry_price, delayed_actual_exit_price, delayed_predicted_exit_price, was_target_hit_early, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, base_t_end) DO UPDATE SET
                    actual_return=excluded.actual_return,
                    actual_duration=excluded.actual_duration,
                    delayed_entry_price=excluded.delayed_entry_price,
                    delayed_actual_exit_price=excluded.delayed_actual_exit_price,
                    delayed_predicted_exit_price=excluded.delayed_predicted_exit_price,
                    was_target_hit_early=excluded.was_target_hit_early,
                    status=excluded.status
            ''', (str(symbol), str(base_t_end), float(predicted_return), float(predicted_duration), 
                  float(actual_return) if actual_return is not None else None, 
                  float(actual_duration) if actual_duration is not None else None, 
                  float(delayed_entry_price) if delayed_entry_price is not None else None,
                  float(delayed_actual_exit_price) if delayed_actual_exit_price is not None else None,
                  float(delayed_predicted_exit_price) if delayed_predicted_exit_price is not None else None,
                  int(was_target_hit_early),
                  str(status)))
            conn.commit()

    def get_basic_model_tests(self, symbol=None):
        """Fetches all past predictions/grades."""
        query = 'SELECT * FROM basic_model_tests'
        params = []
        if symbol:
            query += ' WHERE symbol = ?'
            params.append(symbol)
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params, parse_dates=['base_t_end'])