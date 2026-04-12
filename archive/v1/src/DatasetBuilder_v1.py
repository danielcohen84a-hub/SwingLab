import pandas as pd
import numpy as np
import os
from src.FeatureEngineer import FeatureEngineer
from src.DBManager import DBManager


class DatasetBuilder:
    def __init__(self, db_manager: DBManager):
        self.db = db_manager

    def _load_raw_tables(self):
        """Loads segments, market indicators, and stock metadata via DBManager."""
        segments_df   = self.db.get_segments()
        indicators_df = self.db.get_market_indicators()
        metadata_df   = self.db.get_stock_metadata()
        return segments_df, indicators_df, metadata_df

    def _aggregate_indicators_per_segment(self, segments_df, indicators_df):
        """
        Hyper-Speed Vectorized Aggregation:
        Uses pandas merge_asof to map each hourly swing segment to 
        the corresponding Daily Macro Z-Score using binary search.
        
        This replacement for the 700k-row loop reduces preparation 
        time from 20 minutes to < 1 second.
        """
        # Ensure data is sorted by time (required for merge_asof)
        segments_df = segments_df.sort_values('t_start')
        indicators_df = indicators_df.sort_values('timestamp')
        
        # Rename indicator columns to the final 'avg_zscore' names
        indicator_cols = [c for c in indicators_df.columns if '_zscore' in c]
        rename_map = {c: f"{c.replace('_zscore', '')}_avg_zscore" for c in indicator_cols}
        indicators_prep = indicators_df[['timestamp'] + indicator_cols].rename(columns=rename_map)
        
        # Perform the vectorized 'Backwards' search (map each swing to its day's daily close)
        merged = pd.merge_asof(
            segments_df, 
            indicators_prep, 
            left_on='t_start', 
            right_on='timestamp', 
            direction='backward'
        )
        
        # Cleanup the extra timestamp column from the join
        if 'timestamp_y' in merged.columns:
            merged.drop(columns=['timestamp_y'], inplace=True)
            merged.rename(columns={'timestamp_x': 'timestamp'}, inplace=True)
        elif 'timestamp' in merged.columns and 'timestamp' in indicators_prep.columns:
            # If names collided, pandas might have renamed them. 
            pass

        return merged

    def _join_metadata(self, segments_df, metadata_df):
        """Merges static stock metadata (sector, beta, market cap category) onto segments."""
        return pd.merge(segments_df, metadata_df, on='symbol', how='left')

    def build_training_dataset(self, sequence_length=5, indicator_windows=None):
        """
        Full pipeline:
          1. Load raw tables from DB.
          2. Transform raw ETF closes -> custom rolling Z-scores.
          3. Aggregate macro indicators per segment via range-join.
          4. Join stock metadata.
          5. Run FeatureEngineer.process_data() for scaling + encoding.
          6. Slice into sliding window 3D tensors for the LSTM.
        """
    def build_training_dataset(self, sequence_length=10, indicator_windows=None):
        """
        Standard entry point for run_pipeline.py Step [4].
        """
        print("Building Training Dataset...")
        
        # 1. Prepare the 2D Engineered DataFrame (Slow part)
        processed_df = self.prepare_processed_df(indicator_windows=indicator_windows)

        # 2. Build 3D tensors (Fast part)
        X_train, y_train, feature_names = self._create_sliding_windows(processed_df, sequence_length)

        # 3. Save feature names for transparency/labels
        os.makedirs("Data", exist_ok=True)
        with open("Data/feature_names.txt", "w") as f:
            f.write("\n".join(feature_names))

        print(f"Dataset Built! X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, y_train

    def prepare_processed_df(self, indicator_windows=None):
        """
        Orchestrates the entire 2D data pipeline:
        DB Loading -> Indicator Z-Scores -> Macro Aggregation -> Metadata -> Scaling.
        Returns a clean 2D DataFrame ready for windowing.
        """
        # 1. Load raw data
        segments_df, indicators_df, metadata_df = self._load_raw_tables()
        if segments_df.empty or indicators_df.empty:
             raise ValueError("Missing database data for Step [4].")

        # 2. Macro Engineering
        print("  Transforming market indicators (Custom Regime Z-Scores)...")
        engineer = FeatureEngineer(mode='training', indicator_windows=indicator_windows)
        indicators_engineered = engineer.transform_market_indicators(indicators_df)

        # 3. Join Macro + Metadata
        print(f"  Aggregating macro context for {len(segments_df):,} segments...")
        segments_df = self._aggregate_indicators_per_segment(segments_df, indicators_engineered)
        segments_df = self._join_metadata(segments_df, metadata_df)

        # 4. Feature engineering (scaling, encoding)
        print("  Performing feature scaling and one-hot encoding...")
        processed_df = engineer.process_data(segments_df)
        
        return processed_df

    def _create_sliding_windows(self, df, sequence_length):
        """
        Groups by symbol and creates overlapping lookback windows.
        Converts 2D DataFrame into 3D Numpy Arrays: (samples, sequence_length, features).

        Target (y): the swing_return of the segment IMMEDIATELY AFTER the window.
        """
        df = df.sort_values(by=['symbol', 't_start'])

        X, y, t_ends = [], [], []
        grouped = df.groupby('symbol')

        for symbol, group in grouped:
            # The target is the NEXT swing's return
            target_col = 'swing_return'
            if target_col not in group.columns:
                continue

            # Need the outcome times for sorting
            group_t_ends = group['t_end'].values
            
            # Keep only numeric columns for the final LSTM matrix.
            drop_cols = ['symbol', 't_start', 't_end', 'segment_id', 'load_id', 'company_name']
            math_df = group.drop(columns=[c for c in drop_cols if c in group.columns], errors='ignore')
            numeric_df = math_df.select_dtypes(include=[np.number])
            
            data_matrix = numeric_df.values
            if len(data_matrix) <= sequence_length:
                continue

            numerical_cols = numeric_df.columns.tolist()
            if target_col not in numerical_cols:
                continue

            target_idx = numerical_cols.index(target_col)

            # Identification of feature columns (everything EXCEPT the target)
            x_cols_indices = [idx for idx, col in enumerate(numerical_cols) if col != target_col]

            for i in range(len(data_matrix) - sequence_length):
                window_features = data_matrix[i: i + sequence_length, x_cols_indices]
                target = data_matrix[i + sequence_length, target_idx]
                outcome_time = group_t_ends[i + sequence_length]
                
                X.append(window_features)
                y.append(target)
                t_ends.append(outcome_time)

        # FINAL CHRONOLOGICAL SORT
        # This ensures that X_train[0] is the oldest Move in the market
        # and X_train[-1] is the most recent Move.
        sort_idx = np.argsort(t_ends)
        X = np.array(X)[sort_idx]
        y = np.array(y)[sort_idx]

        # Capture final feature names
        final_feature_names = [col for col in numerical_cols if col != target_col]
        return X, y, final_feature_names
