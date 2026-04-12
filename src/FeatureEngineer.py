import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

class FeatureEngineer:
    def __init__(self, mode='training', scaler_dir='models/scalers/', indicator_windows=None):
        """
        Initializes the Feature Engineer.
        :param mode: 'training' or 'live'. 'training' fits new scalers and saves them. 'live' loads existing scalers.
        :param scaler_dir: Directory where the scalers (.pkl files) are stored.
        :param indicator_windows: Dictionary of {ticker: months} for Z-score calculation.
        """
        self.mode = mode
        self.scaler_dir = scaler_dir
        
        # Default windows if none provided (Fallbacks)
        self.indicator_windows = indicator_windows or {
            'spy': 8, 'vxx': 3, 'uso': 6, 'uup': 18, 'ief': 24
        }

        # Ensure directory exists for saving scalers during training
        if self.mode == 'training' and not os.path.exists(self.scaler_dir):
            os.makedirs(self.scaler_dir)

        # Initialize or load scalers
        if self.mode == 'training':
            self.return_scaler     = StandardScaler()
            self.slope_scaler      = StandardScaler()
            self.duration_scaler   = MinMaxScaler(feature_range=(0, 1))
            self.volatility_scaler = RobustScaler()
            self.volume_scaler     = RobustScaler()
            self.beta_scaler       = StandardScaler()
        else:
            self._load_scalers()

    def _load_scalers(self):
        """Helper to load all saved scalers during live inference."""
        self.return_scaler     = joblib.load(os.path.join(self.scaler_dir, 'return_scaler.pkl'))
        self.slope_scaler      = joblib.load(os.path.join(self.scaler_dir, 'slope_scaler.pkl'))
        self.duration_scaler   = joblib.load(os.path.join(self.scaler_dir, 'duration_scaler.pkl'))
        self.volatility_scaler = joblib.load(os.path.join(self.scaler_dir, 'volatility_scaler.pkl'))
        self.volume_scaler     = joblib.load(os.path.join(self.scaler_dir, 'volume_scaler.pkl'))
        self.beta_scaler       = joblib.load(os.path.join(self.scaler_dir, 'beta_scaler.pkl'))

    def _save_scaler(self, scaler_obj, filename):
        """Helper to save a scaler object."""
        if self.mode == 'training':
            joblib.dump(scaler_obj, os.path.join(self.scaler_dir, filename))

    def _apply_scaling(self, df, column_name, scaler_obj, filename):
        """
        Applies fit_transform (training) or transform (live) and saves the scaler.
        """
        data_2d = df[[column_name]].values
        if self.mode == 'training':
            df[f'{column_name}_scaled'] = scaler_obj.fit_transform(data_2d)
            self._save_scaler(scaler_obj, filename)
        else:
            df[f'{column_name}_scaled'] = scaler_obj.transform(data_2d)
        return df

    def _scale_fixed_bounds(self, df):
        """Scales features that have hard mathematical bounds manually."""
        if 'rsi_start' in df.columns:
            df['rsi_start_scaled'] = df['rsi_start'] / 100.0
        if 'rsi_end' in df.columns:
            df['rsi_end_scaled'] = df['rsi_end'] / 100.0
        if 'hour_of_day' in df.columns:
            df['hour_scaled'] = df['hour_of_day'] / 23.0
        if 'day_of_week' in df.columns:
            df['day_scaled'] = df['day_of_week'] / 4.0
        return df

    def _scale_market_cap(self, df):
        """No log-scaling for categories; they are handled in _encode_categories."""
        return df

    def _encode_categories(self, df):
        """One-Hot Encodes Categorical Data (Sector and Market Cap Category)."""
        # We force dtype=int so they are recognized by numeric filters later
        if 'sector' in df.columns:
            df = pd.get_dummies(df, columns=['sector'], prefix='sector', dtype=int)
        if 'market_cap_category' in df.columns:
            df = pd.get_dummies(df, columns=['market_cap_category'], prefix='mcap', dtype=int)
        return df

    def transform_market_indicators(self, df):
        """
        Transforms raw ETF close prices into Macro Regime (Z-Score) features.

        Instead of hourly returns (noise), we use 1-year rolling Z-Scores of the level.
        This captures the MARKET ENVIRONMENT (e.g., 'Strong Dollar Regime' or 'Cheap Oil Regime').
        
        Z-score = (Current Close - Rolling 252-day Mean) / Rolling 252-day Std
        
        The result is a stationary value typically between -4.0 and +4.0.
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        for ticker in ['spy', 'vxx', 'uso', 'uup', 'ief']:
            raw_col = f'{ticker}_close'
            out_col = f'{ticker}_zscore'
            
            if raw_col in df.columns:
                # Z-Score window: months * 21 trading days (Daily frequency)
                months = self.indicator_windows.get(ticker, 12)
                window_size = int(months * 21)
                
                rolling_mean = df[raw_col].rolling(window=window_size, min_periods=min(window_size, 20)).mean()
                rolling_std  = df[raw_col].rolling(window=window_size, min_periods=min(window_size, 20)).std()
                
                # Formula: (x - mean) / std
                df[out_col] = (df[raw_col] - rolling_mean) / rolling_std.replace(0, np.nan)
                
                # Fill early rows (regime is neutral 0.0)
                df[out_col] = df[out_col].fillna(0.0)

        # Drop raw closes and return only the Z-Scores
        raw_cols = [f'{t}_close' for t in ['spy', 'vxx', 'uso', 'uup', 'ief']]
        df.drop(columns=[c for c in raw_cols if c in df.columns], inplace=True)

        return df

    def process_data(self, df):
        """
        Master Feature Engineering function.
        Takes the raw joined DataFrame from DatasetBuilder and runs all engineering steps.
        """
        processed_df = df.copy()

        # 1. Fixed-bound scaling (RSI, Temporal)
        processed_df = self._scale_fixed_bounds(processed_df)

        # 2. Fitted scalers for stock-level numerical features
        if 'swing_return' in processed_df.columns:
            processed_df = self._apply_scaling(processed_df, 'swing_return', self.return_scaler, 'return_scaler.pkl')
        
        if 'slope_pct_per_hour' in processed_df.columns:
            processed_df = self._apply_scaling(processed_df, 'slope_pct_per_hour', self.slope_scaler, 'slope_scaler.pkl')

        if 'duration_hours' in processed_df.columns:
            processed_df = self._apply_scaling(processed_df, 'duration_hours', self.duration_scaler, 'duration_scaler.pkl')

        if 'residual_volatility_pct' in processed_df.columns:
            processed_df = self._apply_scaling(processed_df, 'residual_volatility_pct', self.volatility_scaler, 'volatility_scaler.pkl')

        if 'avg_hourly_volume' in processed_df.columns:
            processed_df = self._apply_scaling(processed_df, 'avg_hourly_volume', self.volume_scaler, 'volume_scaler.pkl')

        if 'beta' in processed_df.columns:
            # Fill NaN beta (common in OTC/Small caps) with 1.0 (Average Risk)
            processed_df['beta'] = processed_df['beta'].fillna(1.0)
            processed_df = self._apply_scaling(processed_df, 'beta', self.beta_scaler, 'beta_scaler.pkl')

        # 3. Log transform + MinMax for market cap (extreme skew)
        processed_df = self._scale_market_cap(processed_df)

        # 4. One-hot encoding for categorical features
        processed_df = self._encode_categories(processed_df)

        # 5. Drop raw unscaled columns (keep only engineered versions)
        raw_columns_to_drop = [
            'rsi_start', 'rsi_end', 'hour_of_day', 'day_of_week',
            'swing_return', 'slope_pct_per_hour', 'duration_hours',
            'residual_volatility_pct', 'avg_hourly_volume',
            'beta', 'market_cap', 'market_cap_category', 'sector'
        ]
        
        existing_to_drop = [c for c in raw_columns_to_drop if c in processed_df.columns and c != 'swing_return']
        processed_df.drop(columns=existing_to_drop, inplace=True)

        return processed_df

    def prepare_live_tensor(self, live_df, sequence_length=8):
        """
        Transforms a live stock's recent segments into a 3D Tensor [1, seq, 35].
        Ensures the column order matches 'feature_names.txt' exactly.
        """
        # 1. Prepare Features (Scaling)
        engineered_df = self.process_data(live_df)
        
        # 2. Get the official list of features (to ensure order & presence)
        if not os.path.exists('Data/feature_names.txt'):
            raise FileNotFoundError("Missing Data/feature_names.txt. Run Step 4 in pipeline first to generate it.")
            
        with open('Data/feature_names.txt', 'r') as f:
            official_features = [line.strip() for line in f if line.strip()]
            
        # 3. Handle One-Hot Encoding for 'Live' Ticker
        # We need all 35 columns, filled with zeros where inapplicable.
        final_df = pd.DataFrame(0.0, index=engineered_df.index, columns=official_features)
        
        # 4. Fill in the columns we HAVE
        for col in official_features:
            if col in engineered_df.columns:
                final_df[col] = engineered_df[col]
                
        # 5. Extract the last N segments and Reshape
        if len(final_df) < sequence_length:
            print(f"WARNING: Only {len(final_df)} segments found. Prediction might be low quality.")
            pad_len = sequence_length - len(final_df)
            padding = pd.DataFrame(0.0, index=range(pad_len), columns=official_features)
            final_df = pd.concat([padding, final_df])
            
        last_window = final_df.tail(sequence_length).values
        # Reshape to [Batch, Sequence, Features]
        return np.expand_dims(last_window, axis=0)
