import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

class PlotlyOracle:

    def __init__(self, output_dir="outputs/predictions"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_report(self, ticker, ticker_df, segments_df, prediction_return, prediction_duration_hours):
        """
        Creates an interactive HTML dashboard and a static PNG report for dual-target model.
        """
        print(f"Creating Interactive Visual Oracle for {ticker}...")
        
        # 1. Prepare Data
        # Ensure datetimes are correct
        ticker_df['Datetime'] = pd.to_datetime(ticker_df['Datetime'])
        segments_df['t_start'] = pd.to_datetime(segments_df['t_start'])
        segments_df['t_end'] = pd.to_datetime(segments_df['t_end'])

        fig = go.Figure()

        # 2. LAYER 1: Raw Candlesticks (Standard Trading Chart)
        fig.add_trace(go.Candlestick(
            x=ticker_df['Datetime'],
            open=ticker_df['Open'],
            high=ticker_df['High'],
            low=ticker_df['Low'],
            close=ticker_df['Close'],
            name="Raw Price Data",
            opacity=0.6,
            visible=True
        ))

        # 3. LAYER 2: Swing Segment Overlay (Zig-Zag)
        # We build a continuous sequence of peaks and valleys
        zigzag_x = []
        zigzag_y = []
        for i, row in segments_df.iterrows():
            if not zigzag_x:
                zigzag_x.append(row['t_start'])
                zigzag_y.append(row['price_start'])
            zigzag_x.append(row['t_end'])
            zigzag_y.append(row['price_end'])

        fig.add_trace(go.Scatter(
            x=zigzag_x,
            y=zigzag_y,
            mode='lines+markers',
            name="Swing Segments (Zig-Zag)",
            line=dict(color='yellow', width=3),
            marker=dict(size=8, symbol='diamond'),
            visible=True
        ))

        # 4. LAYER 3: Prediction Projection (Neon Dash)
        last_time = zigzag_x[-1]
        last_price = zigzag_y[-1]
        
        # Calculate Target Price from Predicted Return
        target_price = last_price * (1 + prediction_return)
        
        # Use explicitly predicted duration from Model 2
        target_time = last_time + timedelta(hours=float(prediction_duration_hours))
        dur_msg = f" (in {prediction_duration_hours:.1f}h)"

        # Draw the 'Neon Future' dotted line
        pred_color = 'lime' if prediction_return > 0 else 'red'
        fig.add_trace(go.Scatter(
            x=[last_time, target_time],
            y=[last_price, target_price],
            mode='lines+markers+text',
            name="Model Prediction (Next Leg)",
            line=dict(color=pred_color, width=4, dash='dot'),
            marker=dict(size=12, symbol='star'),
            text=[None, f"<b>Target: {prediction_return:+.2%}</b>{dur_msg}"],
            textposition="top center",
            visible=True
        ))

        # 5. Dashboard Layout (Professional Dark Theme)
        fig.update_layout(
            title=f"SWINGLAB ORACLE: {ticker} Prediction Dashboard",
            template="plotly_dark",
            xaxis_title="Time (Hourly Candles)",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # 6. SAVE OUTPUTS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        html_path = os.path.join(self.output_dir, f"{ticker}_{timestamp}_interactive.html")
        png_path = os.path.join(self.output_dir, f"{ticker}_{timestamp}_summary.png")

        # Save HTML (Interactive)
        fig.write_html(html_path)
        
        # Save PNG (Static)
        # Note: Requires 'kaleido' package. Falling back to print if fails.
        try:
            fig.write_image(png_path)
            print(f"Static Report saved: {png_path}")
        except Exception as e:
            print(f"Note: Static PNG skipped (Install 'kaleido' for static images).")

        print(f"Interactive Dashboard saved: {html_path}")
        return html_path, png_path
