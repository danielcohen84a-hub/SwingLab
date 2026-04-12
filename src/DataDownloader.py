import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

class DataDownloader:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"

    def fetch_stock_data(self, symbol, multiplier=1, timeframe="hour", weeks=8, start_date=None, end_date=None, sleep_time=15.0):
        """Downloads raw stock data from Polygon API"""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(weeks=weeks)

        str_end = end_date.strftime("%Y-%m-%d")
        str_start = start_date.strftime("%Y-%m-%d")

        url = f"{self.base_url}/{symbol}/range/{multiplier}/{timeframe}/{str_start}/{str_end}?adjusted=false&sort=asc&limit=50000&apiKey={self.api_key}"

        all_results = []
        while url:
            if all_results:
                time.sleep(sleep_time) 
                
            response = requests.get(url)
            if response.status_code == 429:
                print(f"Rate limit reached for {symbol}. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
                
            data = response.json()
            if "status" in data and data["status"] == "ERROR":
                raise ValueError(f"Error fetching data for {symbol}: {data.get('error', 'Unknown error')}")

            if "results" in data:
                all_results.extend(data["results"])

            next_url = data.get("next_url")
            if next_url:
                url = f"{next_url}&apiKey={self.api_key}"
            else:
                break

        if not all_results:
            raise ValueError(f"No data returned for {symbol}.")

        df = pd.DataFrame(all_results)
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "Datetime"})
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='ms')
        df["idx"] = range(len(df))
        df.set_index('idx', inplace=True)
        return df

    def fetch_stock_data_fast(self, symbol, weeks=8, start_date=None):
        """Downloads short-term hourly data very fast bypassing Polygon limits using yfinance."""
        import yfinance as yf
        try:
            if start_date:
                if start_date.tzinfo is not None:
                    start_date = start_date.tz_localize(None)
                # Ensure we only fetch up to max 730d (limitation of yf 1h)
                df = yf.download(symbol, start=start_date, interval="1h", progress=False)
            else:
                df = yf.download(symbol, period=f"{weeks*7}d", interval="1h", progress=False)
            if df is None or df.empty:
                raise ValueError(f"No data returned for {symbol} from yfinance.")
            
            # yfinance returns multiindex if multiple columns, but we passed 1 ticker
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten the MultiIndex by dropping the 'Ticker' level
                df.columns = df.columns.droplevel('Ticker')
            
            df = df.reset_index()
            # Rename column names properly
            rename_map = {}
            for col in df.columns:
                lower_col = str(col).lower()
                if 'date' in lower_col or 'time' in lower_col: rename_map[col] = 'Datetime'
                elif 'open' in lower_col: rename_map[col] = 'Open'
                elif 'high' in lower_col: rename_map[col] = 'High'
                elif 'low' in lower_col: rename_map[col] = 'Low'
                elif 'close' in lower_col: rename_map[col] = 'Close'
                elif 'volume' in lower_col: rename_map[col] = 'Volume'
                
            df = df.rename(columns=rename_map)
            df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            
            df["idx"] = range(len(df))
            df.set_index('idx', inplace=True)
            return df
            
        except Exception as e:
            raise ValueError(f"yfinance fast-fetch failed for {symbol}: {e}")

    def fetch_market_indicators(self, years_back=4, start_date=None):
        """
        Fetches DAILY close prices for 5 macro ETF proxies from yfinance.
        Using DAILY data is the ONLY way to bypass the 2-year intraday limit!
        """
        import yfinance as yf
        
        end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * years_back)

        INDICATORS = {
            "SPY": "spy_close",   "VXX": "vxx_close", 
            "USO": "uso_close",   "UUP": "uup_close", "IEF": "ief_close",
        }

        merged = None
        for ticker, col_name in INDICATORS.items():
            print(f"  Fetching {ticker} from yfinance (Daily)...", end=" ", flush=True)
            try:
                # yf.download(ticker) often returns MultiIndex columns (Price, Ticker)
                # We specifically want 'Close' for the requested ticker
                df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
                if df is None or df.empty:
                    print(f"X (Failed to fetch {ticker})")
                    continue
                
                # Handle MultiIndex Columns (typical in newer yfinance)
                if isinstance(df.columns, pd.MultiIndex):
                    # Flatten it down to one column: 'Close'
                    try:
                        df = df['Close'][[ticker]] # Subset to our ticker
                    except:
                        df = df['Close']
                else:
                    # In some versions it's just 'Close'
                    if 'Close' in df.columns:
                        df = df[['Close']]
                    else:
                        print(f"X (Close column missing for {ticker})")
                        continue

                # Reset index to pull 'Date' out as a column
                df = df.reset_index()
                # Ensure the date column is present
                date_col = [c for c in df.columns if 'date' in str(c).lower()][0]
                df = df[[date_col, df.columns[-1]]]
                df.columns = ["timestamp", col_name]
                
                # FORCE DTYPE CONSISTENCY
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
                
                print(f"OK ({len(df)} rows)")
                time.sleep(1) # Faster delay for delta fetches

                if merged is None:
                    merged = df
                else:
                    merged = pd.merge(merged, df, on="timestamp", how="outer")
            except Exception as e:
                print(f"X (Error for {ticker}: {e})")
                continue

        if merged is None or merged.empty:
            raise ValueError("Failed to fetch any market indicator data from Yahoo Finance.")

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        return merged.ffill().bfill()

    def run_indicators_pipeline(self, settings, db_manager):
        """
        Highest level orchestrator for Step [1].
        1. Downloads 4-year DAILY history from yfinance.
        2. Saves raw data to the 'market_indicators' table.
        """
        years_back = settings.get('years_back', 4)
        print(f"Executing Deep Macro Sync ({years_back} years via YFinance)...")
        
        # 1. Download
        df = self.fetch_market_indicators(years_back=years_back)
        
        # 2. Save to DB
        # Re-mapping to DB schema column 'timestamp'
        df = df.rename(columns={"Datetime": "timestamp"})
        
        with db_manager._get_connection() as conn:
            # We must force clean column names to avoid SQLite tuple-names
            df.to_sql('market_indicators', conn, if_exists='replace', index=False)
            
        print(f"Successfully saved {len(df)} days of macro context to database.")