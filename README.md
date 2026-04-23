# SwingLab: Multi-Target LSTM Swing Trading Engine

SwingLab is a quantitative research framework and algorithmic trading pipeline designed to predict short-term stock market "swing" movements. Unlike traditional models that try to predict the price of the next individual minute, SwingLab focuses on the **large geometric moves** between peaks and valleys.

![Interactive Dashboard Placeholder](https://via.placeholder.com/800x400.png?text=SwingLab+Dashboard+Interface+Mockup)

## 💡 The Core Idea: Geometric "Swings"

Financial markets are noisy. Predicting the next individual candle is often impossible. SwingLab handles this by:

1.  **Breaking Down Price Action**: We identify the major "peaks" and "valleys" (highs and lows) in stock data.
2.  **Drawing Geometric Segments**: We draw straight lines (linear segments) between these points. This "connects the dots" to reveal the underlying path of the stock.
3.  **Feature Fusion**: For each segment, we add context:
    *   **Stock Data**: Price returns, durations, and volume.
    *   **Company Metadata**: What sector is it in? What is its beta (risk level)?
    *   **Market Environment**: What are the overall market "regimes" doing? (e.g., SPY, Volatility, Oil, and Bonds).
4.  **The Oracle (LSTM)**: We feed this sequence of segments into a Long Short-Term Memory (LSTM) neural network to teach it the pattern and predict the **next** geometric segment.

---

## 🚀 Key Features

*   **Interactive Visual Oracle**: Generates professional reports showing the predicted "future path" of a stock directly on a Plotly chart.
*   **Multi-Target Prediction**: The model predicts both **how much** a stock will return and **how long** the move will take, allowing for better profit targeting.
*   **Realistic Market Simulation**: Includes a backtesting engine that accounts for real-world delays, slippage, and late entry confirmations.
*   **Environment Awareness**: The model uses "Macro" context (Z-scored market indicators) so it knows if it's trading in a bull market, a crash, or a sideways period.

---

## 📐 Mathematical Foundations (Concise)

To ensure the model learns correctly, we use two key mathematical concepts:

*   **Z-Scores (Market Normalization)**: We convert raw market prices into Z-scores. This tells the model how many "standard deviations" a price is from its average. It allows the model to understand if a market is "overbought" or "oversold" regardless of whether the stock price is $10 or $1,000.
*   **Huber Loss (Robust Training)**: During training, we use Huber Loss. Standard models can be "tricked" by extreme market events (like flash crashes). Huber Loss protects the model's learning by treating small errors normally but limiting the impact of extreme market "outliers."

---

## 🏗 Project Architecture (Single Responsibility Principle)

SwingLab is built with modern software architecture in mind. Each component follows the **Single Responsibility Principle (SRP)**—meaning each file has exactly **one job** it does perfectly.

### Core Library (`src/`)

*   **[DBManager.py](file:///c:/Daniel/Studies/Python/SwingLab/src/DBManager.py)**: Solely responsible for **Data Persistence**. It manages the SQLite database, ensuring all raw data, segments, and predictions are stored and retrieved reliably.
*   **[DataDownloader.py](file:///c:/Daniel/Studies/Python/SwingLab/src/DataDownloader.py)**: Solely responsible for **Data Acquisition**. It interfaces with external APIs (Polygon and Yahoo Finance) to bring fresh market data into the system.
*   **[DataProcessor.py](file:///c:/Daniel/Studies/Python/SwingLab/src/DataProcessor.py)**: Solely responsible for **Geometric Signal Processing**. This is where the magic happens: it identifies the high/low points to create the "Swing Segments."
*   **[FeatureEngineer.py](file:///c:/Daniel/Studies/Python/SwingLab/src/FeatureEngineer.py)**: Solely responsible for **Data Transformation**. It scales raw numbers into a model-readable range and calculates the market "Z-score" regimes.
*   **[DatasetBuilder.py](file:///c:/Daniel/Studies/Python/SwingLab/src/DatasetBuilder.py)**: Solely responsible for **Tensor Construction**. It merges segments, market indicators, and company metadata into the 3D data blocks required for the LSTM.
*   **[LSTMModel.py](file:///c:/Daniel/Studies/Python/SwingLab/src/LSTMModel.py)**: Solely responsible for **Deep Learning Operations**. It defines the neural network architecture and handles the training loops.
*   **[StockUniverse.py](file:///c:/Daniel/Studies/Python/SwingLab/src/StockUniverse.py)**: Solely responsible for **Portfolio Selection**. It manages the ~1,500 stocks being tracked and applies strict liquidity filters.
*   **[Visualizer.py](file:///c:/Daniel/Studies/Python/SwingLab/src/Visualizer.py)**: Solely responsible for **Reporting & Visualization**. It generates the interactive HTML reports that visualize the model's predictions.

### Supporting Scripts

*   **[run_pipeline.py](file:///c:/Daniel/Studies/Python/SwingLab/run_pipeline.py)**: The main orchestrator for project setup and training.
*   **[basic_model_tester.py](file:///c:/Daniel/Studies/Python/SwingLab/basic_model_tester.py)**: Real-world simulation and backtesting engine.
*   **[predict_ticker.py](file:///c:/Daniel/Studies/Python/SwingLab/predict_ticker.py)**: Real-time inference tool for any stock symbol.
*   **[tune_hyperparameters.py](file:///c:/Daniel/Studies/Python/SwingLab/tune_hyperparameters.py)**: Automatically optimizes the model settings using Optuna.
*   **[dashboard_api.py](file:///c:/Daniel/Studies/Python/SwingLab/dashboard_api.py)**: The backend server for the live monitoring dashboard.

---

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

4.  **Live Prediction**:
    ```bash
    python predict_ticker.py
    ```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
