# SwingLab Run Instructions

This document explains how to set up and run the entire SwingLab pipeline, from data gathering to the React dashboard.

## 1. Prerequisites
- Python 3.9+
- Node.js & npm (for the React frontend)
- Polygon.io API Key (Free tier works, but fetching might be slower)

## 2. Setup
### Backend Setup
1. Open a terminal in the root directory.
2. Install Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `config.example.yaml` to `config.yaml` and insert your API key:
   ```bash
   cp config.example.yaml config.yaml
   ```

### Frontend Setup
1. Open a new terminal.
2. Navigate to the `swinglab-ui` directory:
   ```bash
   cd swinglab-ui
   ```
3. Install frontend dependencies:
   ```bash
   npm install
   ```

## 3. Running the Machine Learning Pipeline
To run the complete data engineering and ML pipeline (download data, process segments, build datasets, train the LSTM model, and make base predictions):
```bash
python run_pipeline.py
```
*Note: The first run will take a significant amount of time as it downloads historical data (prices and metadata) for the ~1500 stock universe into the local SQLite database.*

For specific parts, you can run:
- **Live Stock Predictor:** `python predict_ticker.py`
- **Backtesting/Simulation:** `python basic_model_tester.py`
- **Hyperparameter Tuning:** `python tune_hyperparameters.py`

## 4. Running the Interactive Dashboard

The project includes a Vite/React dashboard and a FastAPI backend to visualize predictions and model metrics.

### Step A: Start the FastAPI Backend
1. Open a terminal in the project root directory.
2. Start the API server:
   ```bash
   python dashboard_api.py
   ```
   *(Running on `http://127.0.0.1:8000`)*

### Step B: Start the React Frontend
1. Open a second terminal.
2. Navigate to the frontend directory:
   ```bash
   cd swinglab-ui
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Output will display a Local URL (e.g., `http://localhost:5173/`). Open this in your browser to view the SwingLab Interface.
