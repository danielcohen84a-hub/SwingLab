// API Config
const API_BASE = 'http://localhost:5000/api';

// Tab Switching Logic
function switchTab(event) {
    const targetId = event.target.getAttribute('data-target');
    if (!targetId || event.target.classList.contains('disabled')) return;

    // Update Nav
    document.querySelectorAll('.sidebar li').forEach(li => li.classList.remove('active'));
    event.target.classList.add('active');

    // Update Views
    document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
    document.getElementById(targetId).classList.add('active');
}

// Global Data State
let predictionsData = [];

// Formatters
const fmtPct = val => (val * 100).toFixed(2) + '%';
const fmtSign = val => val > 0 ? `+${fmtPct(val)}` : fmtPct(val);
const fmtDollar = val => val >= 0 ? `+$${val.toFixed(2)}` : `-$${Math.abs(val).toFixed(2)}`;

// Chart Instance
let profitChart = null;

async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();
        
        if (data.error || data.message) {
            document.getElementById('kpi-pending').innerText = data.pending_count || 0;
            return;
        }

        document.getElementById('kpi-winrate').innerText = fmtPct(data.win_rate);
        document.getElementById('kpi-total').innerText = data.total_graded;
        document.getElementById('kpi-time-acc').innerText = fmtPct(data.time_acc);
        
        document.getElementById('kpi-profit-long').innerText = fmtDollar(data.profit_long);
        document.getElementById('kpi-profit-short').innerText = fmtDollar(data.profit_short);
        
        document.getElementById('kpi-pending').innerText = data.pending_count;

        renderChart(data.chart_data);
    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

async function loadPredictions() {
    try {
        const res = await fetch(`${API_BASE}/predictions`);
        predictionsData = await res.json();
        renderGeometricTable();
        renderExecutionTable();
    } catch (err) {
        console.error("Failed to load predictions:", err);
    }
}

function renderGeometricTable() {
    const tbody = document.getElementById('geometric-body');
    tbody.innerHTML = '';

    predictionsData.forEach(row => {
        const dateStr = new Date(row.base_t_end).toLocaleString();
        const retOffset = row.actual_return !== null ? row.actual_return - row.predicted_return : null;
        const durOffset = row.actual_duration !== null ? row.actual_duration - row.predicted_duration : null;

        const tr = document.createElement('tr');
        const isDirCorrect = row.actual_return !== null && (Math.sign(row.predicted_return) === Math.sign(row.actual_return));
        const dirHTML = row.actual_return !== null ? (isDirCorrect ? '<span style="color:#10b981; font-weight:800">✅ YES</span>' : '<span style="color:#ef4444; font-weight:800">❌ NO</span>') : '...';

        tr.innerHTML = `
            <td>${dateStr}</td>
            <td style="font-weight:600">${row.symbol}</td>
            <td>${dirHTML}</td>
            <td class="${row.predicted_return > 0 ? 'text-green' : 'text-red'}">${fmtSign(row.predicted_return)}</td>
            <td>${row.actual_return !== null ? fmtSign(row.actual_return) : '...'}</td>
            <td>${retOffset !== null ? fmtOffset(retOffset, row.predicted_return, '%') : '--'}</td>
            <td>${Math.round(row.predicted_duration)}h</td>
            <td>${row.actual_duration !== null ? Math.round(row.actual_duration) + 'h' : '...'}</td>
            <td>${durOffset !== null ? fmtOffset(durOffset, row.predicted_duration, 'h') : '--'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderExecutionTable() {
    const tbody = document.getElementById('execution-body');
    tbody.innerHTML = '';

    predictionsData.forEach(row => {
        // Prediction Time is roughly 7h after anchor (Confirmation lag)
        const predTime = new Date(new Date(row.base_t_end).getTime() + 7 * 3600 * 1000).toLocaleString();
        
        let pnl = 0;
        let pnlText = '--';
        let statusTag = row.status;
        let note = '';

        if (row.was_target_hit_early) {
            note = '<span class="badge badge-loss">TARGET HIT EARLY</span>';
            pnlText = '$0.00';
            statusTag = 'SKIPPED';
        } else if (row.status === 'GRADED' && row.delayed_entry_price && row.delayed_actual_exit_price) {
            let tradeRet = 0;
            if (row.predicted_return > 0) {
                tradeRet = (row.delayed_actual_exit_price - row.delayed_entry_price) / row.delayed_entry_price;
            } else {
                tradeRet = (row.delayed_entry_price - row.delayed_actual_exit_price) / row.delayed_entry_price;
            }
            pnl = tradeRet * 100;
            pnlText = fmtDollar(pnl);
        } else if (row.status === 'PENDING') {
             pnlText = row.delayed_entry_price ? 'LIVE' : 'CONFIRMING...';
        }

        const buyPrice = row.delayed_entry_price ? `$${row.delayed_entry_price.toFixed(2)}` : '--';
        const exitPrice = row.delayed_actual_exit_price ? `$${row.delayed_actual_exit_price.toFixed(2)}` : '--';

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-size:0.85em">${predTime}</td>
            <td style="font-weight:600">${row.symbol} <span class="badge ${row.status === 'GRADED' ? 'badge-graded' : 'badge-pending'}">${statusTag}</span></td>
            <td style="font-size:0.85em">${buyPrice} &rarr; ${exitPrice}</td>
            <td>${fmtSign(row.predicted_return)}</td>
            <td>${pnlText !== '--' && row.status === 'GRADED' ? fmtSign(pnl/100) : '--'}</td>
            <td class="${pnl > 0 ? 'text-green' : (pnl < 0 ? 'text-red' : '')}">${pnlText}</td>
            <td>${row.status === 'GRADED' ? (pnl > 0 ? '✅ WIN' : '❌ LOSS') : '--'}</td>
            <td>${note}</td>
        `;
        tbody.appendChild(tr);
    });
}

function fmtOffset(val, base, unit) {
    const missPct = Math.abs((val / base) * 100).toFixed(0);
    const colorClass = Math.abs(val) < (Math.abs(base) * 0.1) ? 'text-green' : 'text-red';
    const sign = val >= 0 ? '+' : '';
    return `<span class="${colorClass}" style="font-size:0.85em">${sign}${unit === '%' ? (val*100).toFixed(1) + '%' : val.toFixed(1) + 'h'} (${missPct}% Miss)</span>`;
}

function renderChart(data) {
    const ctx = document.getElementById('profitChart').getContext('2d');
    if (profitChart) profitChart.destroy();

    const createGradient = (color) => {
        const g = ctx.createLinearGradient(0, 0, 0, 400);
        g.addColorStop(0, color.replace('1)', '0.3)'));
        g.addColorStop(1, color.replace('1)', '0)'));
        return g;
    };

    profitChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => new Date(d.date).toLocaleDateString()),
            datasets: [
                {
                    label: 'Geometric Long Alpha',
                    data: data.map(d => d.cum_long),
                    borderColor: '#10b981',
                    backgroundColor: createGradient('rgba(16, 185, 129, 1)'),
                    borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0
                },
                {
                    label: 'Geometric Short Alpha',
                    data: data.map(d => d.cum_short),
                    borderColor: '#ef4444',
                    backgroundColor: createGradient('rgba(239, 68, 68, 1)'),
                    borderWidth: 2, tension: 0.4, fill: true, pointRadius: 0
                },
                {
                    label: 'Real-World Execution P&L',
                    data: data.map(d => d.cum_execution),
                    borderColor: '#3b82f6',
                    borderWidth: 3, tension: 0.3, fill: false, pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94a3b8' } } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8', callback: v => '$' + v }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

// Init
window.onload = () => {
    loadStats();
    loadPredictions();
    setInterval(() => { loadStats(); loadPredictions(); }, 30000);
};
