# SwingLab: Multi-Target LSTM Swing Trading Engine

SwingLab is a quantitative research framework and algorithmic trading pipeline designed to predict short-term stock market "swing" movements. It uses a custom-tuned Multi-Target LSTM (Long Short-Term Memory) neural network to forecast both the **directional magnitude (return)** and the **expected duration** of price segments.

![Interactive Dashboard Placeholder](https://via.placeholder.com/800x400.png?text=SwingLab+Dashboard+Interface+Mockup)

## 🚀 Key Features

*   **Multi-Target Learning**: Simultaneously predicts price return % and trade duration (hours), allowing for sophisticated risk/reward optimization.
*   **Vectorized Data Pipeline**: High-speed binary search (merge_asof) to map macro-economic indicators to over 700k+ historical swing segments in milliseconds.
*   **Macro-Aware Features**: Leverages Z-scored regime indicators (SPY, VXX, USO, UUP, IEF) to provide the model with market context.
*   **Asynchronous Walk-Forward Tester**: A realistic simulation engine that accounts for signal latency and early target hits.
*   **Interactive Oracle**: Generates Plotly-based visual reports projecting predicted price paths.
*   **Live Dashboard**: A Flask-based API and HTML/JS frontend for real-time performance analytics.

## 🛠 Tech Stack

*   **Language**: Python 3.9+
*   **Deep Learning**: TensorFlow / Keras (LSTM architecture)
*   **Data Engineering**: Pandas, NumPy, Scikit-learn (RobustScaling)
*   **Database**: SQLite (optimized with WAL journaling)
*   **API Integration**: Polygon.io (Market data)
*   **Visualization**: Plotly, Flask, Tailwind CSS

## 📂 Project Structure

```text
├── src/                    # Core library
│   ├── LSTMModel.py        # Multi-target LSTM architecture
│   ├── DatasetBuilder.py   # 3D Tensor generation
│   ├── DBManager.py        # SQLite interaction layer
│   └── Visualizer.py       # Plotly reporting engine
├── archive/v1/             # Legacy single-target implementations
├── dashboard/              # Frontend web dashboard
├── run_pipeline.py         # Main orchestrator (Setup -> Train)
├── basic_model_tester.py   # Backtesting & Walk-forward analysis
├── predict_ticker.py       # Live inference tool
└── config.example.yaml     # Configuration template
```

## ⚙️ Quick Start

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/yourusername/SwingLab.git
    cd SwingLab
    ```

2.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    cp config.example.yaml config.yaml
    ```
    *Add your Polygon.io API key to `config.yaml`.*

3.  **Run the Pipeline**:
    ```bash
    python run_pipeline.py
    ```
    Follow the interactive menu to download data, generate segments, and train the model.

4.  **Live Prediction**:
    ```bash
    python predict_ticker.py
    ```

5.  **Launch Dashboard**:
    ```bash
    python dashboard_api.py
    ```
    View at `http://localhost:5000`.

---

## 📊 Methodology: The "Swing" Segment
Unlike fixed-interval predictions (e.g., "predict price in 5 hours"), SwingLab decomposes price action into natural **extrema points** (peaks and valleys). The model learns to identify the *next* major turning point, making it more resilient to intraday noise.

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
