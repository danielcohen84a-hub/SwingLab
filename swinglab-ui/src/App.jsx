import React, { useState, useEffect, useRef, useCallback } from 'react';
import Plotly from 'plotly.js-dist-min';
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Activity, 
  CheckCircle2, 
  Clock, 
  AlertCircle,
  X,
  BarChart2
} from 'lucide-react';

const API_URL = 'http://localhost:8000/api';

// ──────────────────────────────────────────────────────────────────────────────
// Chart Modal
// ──────────────────────────────────────────────────────────────────────────────
function ChartModal({ predictionId, ticker, onClose }) {
  const plotRef = useRef(null);
  const [chartStatus, setChartStatus] = useState('loading'); // 'loading' | 'ok' | 'error'
  const [chartError, setChartError] = useState('');

  const buildChart = useCallback(async () => {
    setChartStatus('loading');
    try {
      const res = await fetch(`${API_URL}/chart/${predictionId}`);
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();

      const { candles, zigzag, prediction } = data;

      if (!candles || candles.length === 0) {
        throw new Error('No OHLCV data found in DB for this ticker. The chart requires raw_stock_data to be populated.');
      }

      // --- Trace 1: Candlesticks ---
      const candlestick = {
        type: 'candlestick',
        x: candles.map(c => c.t),
        open:  candles.map(c => c.o),
        high:  candles.map(c => c.h),
        low:   candles.map(c => c.l),
        close: candles.map(c => c.c),
        name: 'Price',
        increasing: { line: { color: '#10B981' }, fillcolor: 'rgba(16,185,129,0.4)' },
        decreasing: { line: { color: '#F43F5E' }, fillcolor: 'rgba(244,63,94,0.4)' },
        opacity: 0.75,
      };

      // --- Trace 2: Zig-zag swing segments ---
      const zigzagTrace = zigzag.length > 0 ? {
        type: 'scatter',
        x: zigzag.map(z => z.t),
        y: zigzag.map(z => z.price),
        mode: 'lines+markers',
        name: 'Swing Segments',
        line: { color: '#FBBF24', width: 3 },
        marker: { size: 8, symbol: 'diamond', color: '#FBBF24' },
      } : null;

      // --- Trace 3: Prediction arrow ---
      // The entry point MUST be the exact time/price the prediction was issued,
      // even if the stored zig-zag segments in the DB haven't caught up to this date yet.
      const entryTime  = prediction.entry_time;
      const entryPrice = prediction.entry_price;

      // Normalize entryTime for Date parsing (replace space with T if needed)
      const normalizedEntryTime = entryTime.replace(' ', 'T');
      const entryTimeMs = new Date(normalizedEntryTime).getTime();

      // Project target time: find the candle that is predicted_duration_bars steps after entry
      const entryIdx = candles.findIndex(c => new Date(c.t).getTime() >= entryTimeMs);
      const anchorIdx = entryIdx >= 0 ? entryIdx : candles.length - 1;
      const targetIdx = anchorIdx + prediction.predicted_duration_bars;

      let targetTime;
      if (targetIdx < candles.length) {
        targetTime = candles[targetIdx].t;
      } else {
        // Estimate using average candle spacing
        const spacing = candles.length > 1
          ? (new Date(candles[candles.length - 1].t).getTime() - new Date(candles[0].t).getTime()) / (candles.length - 1)
          : 3600000; // 1 hour fallback
        const barsInFuture = targetIdx - (candles.length - 1);
        targetTime = new Date(new Date(candles[candles.length - 1].t).getTime() + barsInFuture * spacing).toISOString();
      }

      // Ensure the chart's entry point is exactly what we calculate
      const traceEntryTime = normalizedEntryTime;

      const isUp = prediction.predicted_return > 0;
      const predColor = isUp ? '#34D399' : '#F87171';
      const retPct = (prediction.predicted_return * 100).toFixed(2);
      const targetLabel = `<b>Target: ${retPct}%</b><br>~${prediction.predicted_duration_bars} bars`;

      const predTrace = {
        type: 'scatter',
        x: [traceEntryTime, targetTime],
        y: [entryPrice, prediction.price_target],
        mode: 'lines+markers+text',
        name: 'Prediction',
        line: { color: predColor, width: 4, dash: 'dot' },
        marker: { size: [10, 16], symbol: ['circle', 'star'], color: predColor },
        text: [null, targetLabel],
        textposition: 'top center',
        textfont: { color: predColor, size: 12 },
      };

      const traces = [candlestick, predTrace];
      if (zigzagTrace) traces.splice(1, 0, zigzagTrace);

      const layout = {
        title: {
          text: `<b>${data.ticker}</b> — SwingLab Prediction`,
          font: { color: '#E2E8F0', size: 18 },
        },
        paper_bgcolor: '#13131A',
        plot_bgcolor:  '#0A0A0F',
        font: { color: '#94A3B8', family: 'Inter, system-ui, sans-serif' },
        xaxis: {
          type: 'date',
          title: 'Time (Hourly Candles)',
          gridcolor: 'rgba(255,255,255,0.05)',
          rangeslider: { visible: false },
          color: '#94A3B8',
        },
        yaxis: {
          title: 'Price ($)',
          gridcolor: 'rgba(255,255,255,0.05)',
          color: '#94A3B8',
        },
        legend: {
          orientation: 'h',
          y: 1.06,
          font: { color: '#94A3B8' },
          bgcolor: 'rgba(0,0,0,0)',
        },
        margin: { t: 80, r: 20, b: 50, l: 60 },
        height: 520,
      };

      const config = {
        displayModeBar: true,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
        responsive: true,
      };

      await Plotly.react(plotRef.current, traces, layout, config);
      setChartStatus('ok');
    } catch (err) {
      console.error('Chart error:', err);
      setChartError(err.message);
      setChartStatus('error');
    }
  }, [predictionId]);

  useEffect(() => {
    buildChart();
    return () => {
      if (plotRef.current) Plotly.purge(plotRef.current);
    };
  }, [buildChart]);

  // Close on Escape key
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel glass-panel" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">
            <BarChart2 size={20} className="modal-title-icon" />
            <span>{ticker} — Prediction Chart</span>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {chartStatus === 'loading' && (
            <div className="chart-loading">
              <Activity className="spinner" size={36} />
              <p>Loading chart data from database…</p>
            </div>
          )}
          {chartStatus === 'error' && (
            <div className="chart-error">
              <AlertCircle size={24} />
              <p>{chartError}</p>
            </div>
          )}
          <div
            ref={plotRef}
            style={{ width: '100%', display: chartStatus === 'ok' ? 'block' : 'none' }}
          />
        </div>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Main App
// ──────────────────────────────────────────────────────────────────────────────
function App() {
  const [predictions, setPredictions] = useState([]);
  const [winRate, setWinRate] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState('all');
  const [tickerFilter, setTickerFilter] = useState('');
  const [sectorFilter, setSectorFilter] = useState('');

  // Chart modal
  const [selectedPrediction, setSelectedPrediction] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const predRes = await fetch(`${API_URL}/predictions`);
        if (!predRes.ok) throw new Error('Failed to fetch predictions');
        const predData = await predRes.json();

        const sumRes = await fetch(`${API_URL}/summary`);
        if (!sumRes.ok) throw new Error('Failed to fetch summary');
        const sumData = await sumRes.json();

        setPredictions(predData.data || []);
        setWinRate(sumData.win_rate || 0);
        setError(null);
      } catch (err) {
        console.error("API error:", err);
        setError('Could not connect to the SwingLab API. Ensure the backend is running on port 8000.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const filteredPredictions = predictions.filter(p => {
    const isClosed = p.actual_return !== "";
    if (statusFilter === 'open' && isClosed) return false;
    if (statusFilter === 'closed' && !isClosed) return false;
    if (tickerFilter && !p.ticker.toLowerCase().includes(tickerFilter.toLowerCase())) return false;
    if (sectorFilter && p.sector !== sectorFilter) return false;
    return true;
  });

  const sectors = [...new Set(predictions.map(p => p.sector).filter(Boolean))].sort();

  return (
    <div className="app-container">
      <header className="glass-header">
        <div className="header-content">
          <div>
            <h1 className="gradient-text">SwingLab Dashboard</h1>
            <p className="subtitle">Live AI Prediction Tracker</p>
          </div>
          
          <div className="metrics-card">
            <div className="metric-icon"><Target size={24} /></div>
            <div className="metric-info">
              <span className="metric-label">Model Win Rate (Graded)</span>
              <span className="metric-value">{(winRate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        {error && (
          <div className="error-banner">
            <AlertCircle size={20} />
            {error}
          </div>
        )}

        <div className="controls-bar glass-panel">
          <div className="filter-group">
            <label>Status</label>
            <div className="button-group">
              <button className={statusFilter === 'all'    ? 'active' : ''} onClick={() => setStatusFilter('all')}>All</button>
              <button className={statusFilter === 'open'   ? 'active' : ''} onClick={() => setStatusFilter('open')}>Open (Live)</button>
              <button className={statusFilter === 'closed' ? 'active' : ''} onClick={() => setStatusFilter('closed')}>Closed</button>
            </div>
          </div>

          <div className="filter-group flex-1">
            <label>Ticker search</label>
            <input 
              type="text" 
              placeholder="e.g. AAPL" 
              value={tickerFilter}
              onChange={(e) => setTickerFilter(e.target.value)}
              className="search-input"
            />
          </div>

          <div className="filter-group">
            <label>Sector</label>
            <select 
              value={sectorFilter} 
              onChange={(e) => setSectorFilter(e.target.value)}
              className="select-input"
            >
              <option value="">All Sectors</option>
              {sectors.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="table-container glass-panel">
          {loading ? (
            <div className="loading-state">
              <Activity className="spinner" size={32} />
              <p>Syncing with database...</p>
            </div>
          ) : (
            <>
              <p className="table-hint">Click any row to view its prediction chart</p>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Status</th>
                    <th>Ticker</th>
                    <th>Entry Date</th>
                    <th>Entry Price</th>
                    <th>Pred Return</th>
                    <th>Duration</th>
                    <th>Actual Return</th>
                    <th>Target Hit</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPredictions.length === 0 ? (
                    <tr>
                      <td colSpan="8" className="empty-state">No predictions found matching filters.</td>
                    </tr>
                  ) : (
                    filteredPredictions.map((row) => {
                      const isOpen = row.actual_return === "";
                      const isWin  = !isOpen && row.direction_correct === 1;
                      
                      return (
                        <tr
                          key={row.prediction_id}
                          className={`table-row clickable-row ${isOpen ? 'row-open' : ''}`}
                          onClick={() => setSelectedPrediction(row)}
                          title="Click to view prediction chart"
                        >
                          <td>
                            {isOpen ? (
                              <span className="badge badge-live">
                                <span className="pulse-dot"></span> Live
                              </span>
                            ) : (
                              <span className="badge badge-closed">
                                <CheckCircle2 size={12} /> Graded
                              </span>
                            )}
                          </td>
                          <td className="font-bold">
                            {row.ticker}
                            {row.is_bootstrap === 1 && <span className="text-xs text-muted ml-1">(B)</span>}
                          </td>
                          <td className="text-muted text-sm">
                            {new Date(row.segment_start_time).toLocaleDateString()}
                          </td>
                          <td>${parseFloat(row.price_at_prediction).toFixed(2)}</td>
                          
                          <td className={row.predicted_return > 0 ? "text-bull" : "text-bear"}>
                            <div className="flex-center gap-1">
                              {row.predicted_return > 0 ? <TrendingUp size={14}/> : <TrendingDown size={14}/>}
                              {(row.predicted_return * 100).toFixed(2)}%
                            </div>
                          </td>
                          
                          <td className="text-sm">
                            <div className="flex-center gap-1">
                              <Clock size={14} className="text-muted"/>
                              {isOpen
                                ? `${row.predicted_duration_bars} bars limit`
                                : `${row.actual_duration_bars}/${row.predicted_duration_bars} bars`}
                            </div>
                          </td>
                          
                          <td className="font-bold">
                            {isOpen ? (
                              <span className="text-muted">—</span>
                            ) : (
                              <span className={isWin ? "text-bull" : "text-bear"}>
                                {(row.actual_return * 100).toFixed(2)}%
                              </span>
                            )}
                          </td>

                          <td>
                            {isOpen ? (
                              <span className="text-muted text-sm">Target: ${parseFloat(row.price_target).toFixed(2)}</span>
                            ) : (
                              row.target_was_hit === 1 ? (
                                <span className="badge badge-hit">Hit Target</span>
                              ) : (
                                <span className="badge badge-miss">Missed Target</span>
                              )
                            )}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </>
          )}
        </div>
      </main>

      {selectedPrediction && (
        <ChartModal
          predictionId={selectedPrediction.prediction_id}
          ticker={selectedPrediction.ticker}
          onClose={() => setSelectedPrediction(null)}
        />
      )}
    </div>
  );
}

export default App;
