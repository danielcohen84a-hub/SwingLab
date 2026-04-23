import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


class SwingProcessor:
    def __init__(self, extrema_order=5):
        """
        :param extrema_order: A point must be greater/less than this many candles on
                              each side to qualify as a local extremum. Higher = fewer,
                              larger swings detected.
        """
        self.order = extrema_order

    def _add_rsi(self, df):
        """Calculates standard 14-period EMA RSI on Close prices (range: 0-100)."""
        delta    = df["Close"].diff()
        gain     = delta.clip(lower=0)
        loss     = -1 * delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs       = ema_gain / ema_loss
        df["RSI"] = (100 - (100 / (1 + rs))).fillna(50)
        return df

    def _residual_volatility_pct(self, interior_close, price_start, price_end):
        """
        Measures how much the ACTUAL price path deviated from the ideal straight line.

        The 'ideal' path is the straight line from price_start to price_end.
        We compute residuals ONLY for interior candles (not the endpoints, which by
        definition sit exactly on the line), then normalize by price_start so the
        result is comparable across stocks at any price level.

        Returns 0.0 if no interior points exist (segment is only 1-2 candles long).
        """
        values = interior_close.values
        n      = len(values)
        if n < 1:
            return 0.0

        # Linear interpolation: interior point i is at fraction (i+1)/(n+1) along the line
        ideal    = price_start + (price_end - price_start) * np.arange(1, n + 1) / (n + 1)
        residuals = values - ideal

        return float(np.std(residuals) / price_start) if price_start > 0 else 0.0

    def generate_segments(self, df):
        """
        Decomposes a stock's OHLCV history into a sequence of linear swing segments.

        Each segment represents the straight-line move y = mx + b between two consecutive
        local extrema (peaks and troughs). All features are computed LOCALLY per segment
        — no reference to the total length of history — ensuring they are universally
        comparable across:
          - Stocks with different numbers of candles (4000 vs 7000)
          - Training data (2 years) vs live inference data (4-8 weeks)
          - Any absolute price level

        Returns a DataFrame where each row is one swing segment.

        NOTE ON DURATION:
        'duration_hours' is calculated as (idx1 - idx0). This is the number of 
        hourly candles (bars) between extrema, NOT the wall-clock elapsed time. 
        The model predicts this candle count directly.
        """
        df = df.copy()
        df = self._add_rsi(df)

        # ── Extrema Detection ────────────────────────────────────────────────────
        # Run on RAW close prices. argrelextrema only compares relative magnitudes,
        # so no scaling is needed here.
        close_vals = df["Close"].values
        max_idx    = argrelextrema(close_vals, np.greater, order=self.order)[0]
        min_idx    = argrelextrema(close_vals, np.less,    order=self.order)[0]
        ext_idx    = np.sort(np.concatenate([max_idx, min_idx]))

        if len(ext_idx) < 2:
            return pd.DataFrame()

        # ── Build One Row Per Consecutive Extremum Pair ──────────────────────────
        segments = []
        for i in range(len(ext_idx) - 1):
            idx0 = int(ext_idx[i])
            idx1 = int(ext_idx[i + 1])

            price_start = float(df["Close"].iloc[idx0])
            price_end   = float(df["Close"].iloc[idx1])

            if price_start <= 0:
                continue

            duration_hours = idx1 - idx0
            if duration_hours == 0:
                continue

            # --- GEOMETRIC: the line itself ---
            swing_return       = (price_end - price_start) / price_start
            slope_pct_per_hour = swing_return / duration_hours

            # --- QUALITY: how clean was the path (residual from ideal line) ---
            interior_close      = df["Close"].iloc[idx0 + 1: idx1]  # strictly between endpoints
            residual_vol_pct    = self._residual_volatility_pct(interior_close, price_start, price_end)

            # --- VOLUME: average per hour (removes duration bias) ---
            volume_window       = df["Volume"].iloc[idx0: idx1 + 1]
            avg_hourly_volume   = float(volume_window.sum() / duration_hours)

            # --- MOMENTUM: RSI at both extrema ---
            rsi_start = float(df["RSI"].iloc[idx0])
            rsi_end   = float(df["RSI"].iloc[idx1])

            # --- TEMPORAL: when in the trading week did this swing begin ---
            t_start     = df['Datetime'].iloc[idx0]
            t_end       = df['Datetime'].iloc[idx1]
            hour_of_day = int(t_start.hour)
            day_of_week = int(t_start.dayofweek)   # 0 = Monday, 4 = Friday

            segments.append({
                "t_start":                 t_start,
                "t_end":                   t_end,
                "price_start":             float(price_start),
                "price_end":               float(price_end),
                # Geometric
                "swing_return":            float(swing_return),
                "duration_hours":          int(duration_hours),
                "slope_pct_per_hour":      float(slope_pct_per_hour),
                # Quality
                "residual_volatility_pct": float(residual_vol_pct),
                # Volume
                "avg_hourly_volume":       avg_hourly_volume,
                # Momentum
                "rsi_start":               rsi_start,
                "rsi_end":                 rsi_end,
                # Temporal
                "hour_of_day":             hour_of_day,
                "day_of_week":             day_of_week,
            })

        return pd.DataFrame(segments)