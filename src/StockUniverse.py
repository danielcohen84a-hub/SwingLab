import pandas as pd
import ssl
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

# Bypass potential SSL certificate issues when scraping Wikipedia
ssl._create_default_https_context = ssl._create_unverified_context

class UniverseManager:
    def __init__(self):
        self.russell3000_url = 'https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund'
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    def fetch_russell3000(self):
        print("Fetching Russell 3000 Constituents (IWV ETF Proxy) from BlackRock...")
        response = requests.get(self.russell3000_url, headers=self.headers)
        if response.status_code != 200:
            raise ValueError("Failed to fetch Russell 3000 holdings.")
            
        import io
        # BlackRock CSV starts with 9 lines of warning text
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data, skiprows=9)
        
        # Filter to only actual Equity holdings (remove Cash or Derivatives)
        df = df[df['Asset Class'] == 'Equity'].copy()
        df = df[['Ticker', 'Name', 'Sector']].copy()
        
        # Drop weird tickers early (dashes, dots, empty)
        df = df[~df['Ticker'].str.contains(r'[\.\-\ ]', na=True)]
        
        df.rename(columns={
            'Ticker': 'symbol', 
            'Name': 'company_name',
            'Sector': 'sector'
        }, inplace=True)
        
        # We'll determine industry from yfinance since BlackRock only provides Sector
        df['industry'] = None
        
        return df

    def fetch_sp500(self):
        print("Fetching S&P 500 (Large Cap)...")
        table = pd.read_html(self.sp500_url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
        df = table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].copy()
        df['Market Cap Category'] = 'Large'
        return df

    def fetch_sp400(self):
        print("Fetching S&P 400 (Mid Cap)...")
        table = pd.read_html(self.sp400_url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
        df = table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].copy()
        df['Market Cap Category'] = 'Mid'
        return df

    def fetch_sp600(self):
        print("Fetching S&P 600 (Small Cap)...")
        table = pd.read_html(self.sp600_url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
        sec_col = 'Company' if 'Company' in table.columns else 'Security'
        df = table[['Symbol', sec_col, 'GICS Sector', 'GICS Sub-Industry']].copy()
        df.rename(columns={sec_col: 'Security'}, inplace=True)
        df['Market Cap Category'] = 'Small'
        return df

    def _fetch_yf_metadata(self, symbol):
        """Helper to fetch single ticker metadata and strictly filter it"""
        time.sleep(0.2)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return symbol, None

            # --- APPLY STRICT LSTM FILTERS ---
            avg_vol = info.get('averageVolume', 0)
            if avg_vol < 500000:
                return symbol, None # Rejected: Low Liquidity

            price = info.get('regularMarketPrice') or info.get('currentPrice', 0)
            if price < 2.00:
                return symbol, None # Rejected: Penny stock math skew

            mcap = info.get('marketCap', 0)
            if mcap < 200_000_000:
                return symbol, None # Rejected: Nano-cap manipulation vulnerability

            # --- DYNAMIC MARKET CAP CATEGORIZATION ---
            category = 'Micro'
            if mcap >= 10_000_000_000:
                category = 'Large'
            elif mcap >= 2_000_000_000:
                category = 'Mid'
            elif mcap >= 300_000_000:
                category = 'Small'

            has_div = False
            if info.get('dividendYield') or info.get('trailingAnnualDividendYield') or info.get('dividendRate'):
                has_div = True

            ipo_year = None
            if info.get('firstTradeDateEpoch'):
                ipo_year = pd.to_datetime(info['firstTradeDateEpoch'], unit='s').year

            # If industry isn't defined by BlackRock, grab it here
            industry = info.get('industry', 'Unknown')
            sector = info.get('sector', 'Unknown')

            return symbol, {
                'exchange': info.get('exchange'),
                'beta': info.get('beta'),
                'average_volume_90_days': avg_vol,
                'has_dividend': 'Yes' if has_div else 'No',
                'ipo_year': str(ipo_year) if ipo_year else None,
                'market_cap_category': category,
                'industry': industry,
                'sector': sector  # Override ETF sector with YF sector for precision
            }
        except Exception:
            # Silent fail for individual tickers (e.g. delisted)
            return symbol, None

    def get_russell_3000_with_metadata(self, limit=None):
        """
        Fetches the Russell 3000 and heavily filters it using Yahoo Finance metadata.
        """
        df = self.fetch_russell3000()

        if limit:
            print(f"LIMIT ACTIVE: Only fetching metadata for the first {limit} stocks.")
            df = df.head(limit).copy()

        print(f"Applying strict filters and extracting metadata for {len(df)} initial stocks...")
        
        # Initialize new columns
        for col in ['exchange', 'beta', 'average_volume_90_days', 'has_dividend', 'ipo_year', 'market_cap_category']:
            df[col] = None

        results = {}
        total_fetched = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(self._fetch_yf_metadata, row['symbol']): row['symbol'] for _, row in df.iterrows()}
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                results[symbol] = data
                total_fetched += 1
                if total_fetched % 100 == 0:
                    print(f"  ... {total_fetched}/{len(df)} processed ...")

        # Map results and drop rejected stocks
        valid_indices = []
        for idx, row in df.iterrows():
            sym = row['symbol']
            if sym in results and results[sym]:
                valid_indices.append(idx)
                df.at[idx, 'exchange'] = results[sym].get('exchange')
                df.at[idx, 'beta'] = results[sym].get('beta')
                df.at[idx, 'average_volume_90_days'] = results[sym].get('average_volume_90_days')
                df.at[idx, 'has_dividend'] = results[sym].get('has_dividend')
                df.at[idx, 'ipo_year'] = results[sym].get('ipo_year')
                df.at[idx, 'market_cap_category'] = results[sym].get('market_cap_category')
                
                # Better precision from YF
                if results[sym].get('industry') != 'Unknown':
                    df.at[idx, 'industry'] = results[sym].get('industry')
                    df.at[idx, 'sector'] = results[sym].get('sector')
                    
        filtered_df = df.loc[valid_indices].copy()
        filtered_df = filtered_df.reset_index(drop=True)
        
        print(f"Filtering complete: Kept {len(filtered_df)} premium LSTM-ready stocks (Rejected {len(df) - len(filtered_df)}).")
        return filtered_df

if __name__ == "__main__":
    manager = UniverseManager()
    
    print("Building highly-optimized Russell 3000 universe... (This takes ~15 minutes)")
    df = manager.get_russell_3000_with_metadata()
    
    # Save a CSV backup
    import os
    os.makedirs('Data', exist_ok=True)
    df.to_csv('Data/russell3000_filtered.csv', index=False)
    print(f"\nSaved {len(df)} pristine stocks to CSV backup: Data/russell3000_filtered.csv")

    # Insert into the SQLite database
    from DBManager import DBManager
    db = DBManager()
    print("Inserting data into SQLite Database `stocks_meta_data` table...")
    
    for idx, row in df.iterrows():
        db.upsert_stock_metadata(
            symbol=row['symbol'],
            company_name=row['company_name'],
            sector=row['sector'],
            industry=row['industry'],
            exchange=row['exchange'],
            market_cap_category=row['market_cap_category'],
            beta=row['beta'],
            average_volume_90_days=row['average_volume_90_days'],
            has_dividend=row['has_dividend'],
            ipo_year=row['ipo_year']
        )
        
    print("Successfully populated your database!")

