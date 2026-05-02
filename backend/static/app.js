/* ══════════════════════════════════════════════════════════
   BTC Range Predictor — Frontend Logic
   Sections: Dashboard, Backtest, History, Monte Carlo, Volatility
══════════════════════════════════════════════════════════ */

let priceChart = null;
let histogramChart = null;
let volChart = null;
let winklerChart = null;
let cachedPrediction = null;

// ── Utility ──────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const fmt = (n) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(n);
const fmtShort = (n) => `$${(+n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct = (n) => `${(n * 100).toFixed(2)}%`;
const getTheme = () => document.documentElement.getAttribute('data-theme') || 'dark';

function chartColors() {
    const dark = getTheme() === 'dark';
    return {
        text: dark ? '#a0a0a0' : '#555555',
        grid: dark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.07)',
        accent: dark ? '#10b981' : '#059669',
        accent2: dark ? '#f97316' : '#ea580c',
        red: dark ? '#ef4444' : '#dc2626',
        green: dark ? '#22c55e' : '#16a34a',
        yellow: dark ? '#eab308' : '#ca8a04',
    };
}

// ── Spinner ───────────────────────────────────────────────────────
function showSpinner(on) {
    $('spinner').classList.toggle('active', on);
}

// ── Navigation ────────────────────────────────────────────────────
function activateSection(name) {
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    $(`section-${name}`).classList.add('active');
    $(`nav-${name}`).classList.add('active');
    const titles = {
        dashboard: 'Dashboard',
        backtest: '30-Day Backtest',
        history: 'Prediction History',
        montecarlo: 'Monte Carlo Simulation',
        volatility: 'Volatility Regime',
    };
    $('page-title').textContent = titles[name] || name;

    // lazy render charts that need data
    if (name === 'backtest') loadBacktest();
    if (name === 'history') loadHistory();
    if (name === 'montecarlo' && cachedPrediction) renderHistogram(cachedPrediction);
    if (name === 'volatility' && cachedPrediction) renderVolChart(cachedPrediction);
}

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => activateSection(btn.dataset.section));
});

// ── Theme Toggle ─────────────────────────────────────────────────
$('theme-toggle').addEventListener('change', (e) => {
    const theme = e.target.checked ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);
    setTimeout(() => {
        if (cachedPrediction) {
            renderPriceChart(cachedPrediction);
            renderHistogram(cachedPrediction);
            renderVolChart(cachedPrediction);
        }
    }, 50);
});

// ── Refresh Button ────────────────────────────────────────────────
$('refresh-btn').addEventListener('click', () => {
    $('refresh-btn').classList.add('spinning');
    loadCurrentPrediction().finally(() => {
        $('refresh-btn').classList.remove('spinning');
    });
});

// ── LOAD CURRENT PREDICTION ───────────────────────────────────────
async function loadCurrentPrediction() {
    showSpinner(true);
    try {
        const res = await fetch('/api/prediction/current');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cachedPrediction = data;

        updateHeroCard(data);
        updateConfidenceBanner(data);
        updateTopbar(data);
        renderPriceChart(data);
        await loadMetrics();

        $('last-updated').textContent = new Date().toLocaleTimeString();
    } catch (err) {
        console.error('Failed to load prediction', err);
    } finally {
        showSpinner(false);
    }
}

// ── Hero Card ─────────────────────────────────────────────────────
function updateHeroCard(data) {
    $('hero-price').textContent = fmt(data.current_price);
    $('hero-lower').textContent = fmtShort(data.prediction.lower);
    $('hero-upper').textContent = fmtShort(data.prediction.upper);
    $('hero-timestamp').textContent = `as of ${new Date().toLocaleString()}`;
    $('range-width').textContent = `Range width: ${fmt(data.prediction.width)}`;
}

// ── Topbar ────────────────────────────────────────────────────────
function updateTopbar(data) {
    $('topbar-price').textContent = fmt(data.current_price);
}

// ── Confidence Banner ─────────────────────────────────────────────
function updateConfidenceBanner(data) {
    const banner = $('confidence-banner');
    banner.classList.remove('hidden', 'green', 'yellow', 'red');
    banner.classList.add(data.confidence_color);

    const icons = { High: '✅', Medium: '⚡', Low: '⚠️' };
    $('confidence-icon').textContent = icons[data.confidence] || '🔮';
    $('confidence-label').textContent = `Model Confidence: ${data.confidence}`;
    $('confidence-msg').textContent = data.confidence_msg;
}

// ── Metrics ────────────────────────────────────────────────────────
async function loadMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        if (data.coverage !== null) {
            $('metric-coverage').textContent = pct(data.coverage);
            $('metric-width').textContent = fmt(data.avg_width);
            $('metric-winkler').textContent = data.avg_winkler.toFixed(1);
            $('metric-preds').textContent = data.num_predictions || '—';

            // Coverage badge
            const badge = $('coverage-badge');
            const cov = data.coverage;
            if (cov >= 0.93 && cov <= 0.97) {
                badge.textContent = '✓ On Target'; badge.className = 'metric-badge good';
            } else if (cov > 0.97) {
                badge.textContent = '↑ Too wide'; badge.className = 'metric-badge warn';
            } else {
                badge.textContent = '↓ Too narrow'; badge.className = 'metric-badge bad';
            }
        }
    } catch (e) { console.warn('Metrics not ready', e); }
}

// ── PRICE CHART ────────────────────────────────────────────────────
function renderPriceChart(data) {
    const ctx = $('priceChart').getContext('2d');
    const C = chartColors();
    const chartData = data.chart_data;

    const labels = chartData.map(d => d.timestamp);
    const closes = chartData.map(d => d.close);

    // Add "Next Hour" prediction point
    labels.push('Next Hour');
    closes.push(null);

    const lowerLine = new Array(closes.length).fill(null);
    const upperLine = new Array(closes.length).fill(null);
    // Connect last actual price to prediction ribbon
    lowerLine[lowerLine.length - 2] = closes[closes.length - 2];
    upperLine[upperLine.length - 2] = closes[closes.length - 2];
    lowerLine[lowerLine.length - 1] = data.prediction.lower;
    upperLine[upperLine.length - 1] = data.prediction.upper;

    if (priceChart) priceChart.destroy();

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'BTC Close Price',
                    data: closes,
                    borderColor: C.accent2,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.15,
                    pointRadius: 0, pointHitRadius: 12,
                    order: 1,
                },
                {
                    label: '95% Lower Bound',
                    data: lowerLine,
                    borderColor: 'transparent',
                    backgroundColor: 'rgba(16,185,129,0.15)',
                    fill: '+1',
                    pointRadius: 0, tension: 0, order: 2,
                },
                {
                    label: '95% Upper Bound',
                    data: upperLine,
                    borderColor: 'rgba(16,185,129,0.5)',
                    borderDash: [5, 4],
                    backgroundColor: 'transparent',
                    pointRadius: 0, tension: 0, order: 3,
                },
            ],
        },
        options: baseChartOptions({
            plugins: {
                legend: { labels: { color: C.text, boxWidth: 14, font: { family: 'Outfit' } } },
                tooltip: { mode: 'index', intersect: false },
            },
            scales: buildScales(C, true),
        }),
    });
}

// ── HISTOGRAM (Monte Carlo) ────────────────────────────────────────
function renderHistogram(data) {
    const ctx = $('histogramChart').getContext('2d');
    const C = chartColors();
    const hist = data.histogram;
    const lower = data.prediction.lower;
    const upper = data.prediction.upper;
    const current = data.current_price;

    // Compute percentile stats from bin centers and counts
    let totalCount = hist.counts.reduce((a, b) => a + b, 0);
    let cumCount = 0;
    let p25 = null, p50 = null, p975 = null;
    for (let i = 0; i < hist.counts.length; i++) {
        cumCount += hist.counts[i];
        const frac = cumCount / totalCount;
        if (!p25 && frac >= 0.025) p25 = hist.bin_centers[i];
        if (!p50 && frac >= 0.5) p50 = hist.bin_centers[i];
        if (!p975 && frac >= 0.975) p975 = hist.bin_centers[i];
    }

    $('mc-p025').textContent = p25 ? fmtShort(p25) : '—';
    $('mc-p50').textContent = p50 ? fmtShort(p50) : '—';
    $('mc-p975').textContent = p975 ? fmtShort(p975) : '—';
    $('mc-current').textContent = fmtShort(current);

    // Color bars: green if inside 95% CI, grey otherwise
    const barColors = hist.bin_centers.map(bc =>
        bc >= lower && bc <= upper
            ? 'rgba(16,185,129,0.7)'
            : 'rgba(150,150,150,0.3)'
    );

    if (histogramChart) histogramChart.destroy();

    histogramChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hist.bin_centers.map(v => fmtShort(v)),
            datasets: [{
                label: 'Simulated Prices',
                data: hist.counts,
                backgroundColor: barColors,
                borderRadius: 3,
                borderSkipped: false,
            }],
        },
        options: {
            ...baseChartOptions(),
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Count: ${ctx.parsed.y.toLocaleString()}`,
                        title: ctx => `Price: ${ctx[0].label}`,
                    },
                },
                annotation: {
                    annotations: {
                        lower: {
                            type: 'line', xMin: hist.bin_centers.findIndex(v => v >= lower),
                            xMax: hist.bin_centers.findIndex(v => v >= lower),
                            borderColor: C.red, borderWidth: 2, borderDash: [5, 4],
                            label: { content: '2.5%', enabled: true, color: C.red, position: 'end', font: { size: 10 } }
                        },
                        upper: {
                            type: 'line', xMin: hist.bin_centers.findIndex(v => v >= upper),
                            xMax: hist.bin_centers.findIndex(v => v >= upper),
                            borderColor: C.green, borderWidth: 2, borderDash: [5, 4],
                        },
                    }
                }
            },
            scales: buildScales(C, false),
        },
    });
}

// ── VOLATILITY CHART ───────────────────────────────────────────────
function renderVolChart(data) {
    const ctx = $('volChart').getContext('2d');
    const C = chartColors();
    const vol = data.volatility;
    const last48 = vol.last_48;
    const threshold = vol.threshold;

    $('vol-current').textContent = (vol.current * 100).toFixed(4) + '%';
    $('vol-threshold').textContent = (threshold * 100).toFixed(4) + '%';
    const conf = data.confidence;
    const confColors = { High: C.green, Medium: C.yellow, Low: C.red };
    const regimeEl = $('vol-regime');
    regimeEl.textContent = conf;
    regimeEl.style.color = confColors[conf] || C.text;

    const labels = last48.map((_, i) => `T-${last48.length - i}h`);
    const barColors = last48.map(v => v > threshold ? C.red : C.accent);

    if (volChart) volChart.destroy();

    volChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Rolling 10-bar Vol',
                    data: last48.map(v => (v * 100).toFixed(5)),
                    backgroundColor: barColors,
                    borderRadius: 3,
                    order: 2,
                },
                {
                    label: 'Threshold (70th pct)',
                    data: last48.map(() => (threshold * 100).toFixed(5)),
                    type: 'line',
                    borderColor: C.yellow,
                    borderDash: [6, 4],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    order: 1,
                },
            ],
        },
        options: {
            ...baseChartOptions(),
            plugins: {
                legend: { labels: { color: C.text, font: { family: 'Outfit' } } },
                tooltip: { mode: 'index', intersect: false },
            },
            scales: buildScales(C, false),
        },
    });
}

// ── BACKTEST ──────────────────────────────────────────────────────
async function loadBacktest() {
    try {
        const [metricsRes, rowsRes] = await Promise.all([
            fetch('/api/metrics'),
            fetch('/api/backtest/results?limit=50'),
        ]);
        const metrics = await metricsRes.json();
        const rows = await rowsRes.json();

        if (metrics.coverage !== null) {
            const covPct = (metrics.coverage * 100).toFixed(2);
            $('bt-coverage').textContent = `${covPct}%`;
            $('bt-width').textContent = fmt(metrics.avg_width);
            $('bt-winkler').textContent = metrics.avg_winkler.toFixed(2);
            $('bt-preds').textContent = metrics.num_predictions || rows.length;

            // Gauge arc (semicircle ~251px circumference for half-circle)
            const arcLen = Math.min(metrics.coverage / 1.0, 1.0) * 251;
            const arcEl = $('gauge-arc');
            arcEl.style.strokeDasharray = `${arcLen} 251`;
            $('gauge-text').textContent = `${covPct}%`;
            arcEl.style.stroke = metrics.coverage >= 0.93 ? '#22c55e' : metrics.coverage >= 0.90 ? '#eab308' : '#ef4444';
        }

        renderWinklerChart(rows);
        renderBacktestTable(rows);
    } catch (e) { console.error('Backtest load failed', e); }
}

function renderBacktestTable(rows) {
    const tbody = $('backtest-tbody');
    tbody.innerHTML = '';
    rows.slice().reverse().slice(0, 50).forEach((r, i) => {
        const hit = r.covered;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${new Date(r.timestamp).toLocaleString()}</td>
            <td>${fmtShort(r.actual)}</td>
            <td class="hit-no">${fmtShort(r.lower)}</td>
            <td class="hit-yes">${fmtShort(r.upper)}</td>
            <td>${fmt(r.width)}</td>
            <td>${r.winkler.toFixed(2)}</td>
            <td class="${hit ? 'hit-yes' : 'hit-no'}">${hit ? '✓ Hit' : '✗ Miss'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderWinklerChart(rows) {
    const ctx = $('winklerChart').getContext('2d');
    const C = chartColors();
    const labels = rows.slice(-50).map((r, i) => `T-${50 - i}`);
    const winklerVals = rows.slice(-50).map(r => r.winkler.toFixed(2));
    const colors = rows.slice(-50).map(r => r.covered ? C.green : C.red);

    if (winklerChart) winklerChart.destroy();
    winklerChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Winkler Score',
                data: winklerVals,
                backgroundColor: colors,
                borderRadius: 3,
            }],
        },
        options: {
            ...baseChartOptions(),
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => `Winkler: ${parseFloat(ctx.parsed.y).toFixed(2)} — ${rows[ctx.dataIndex]?.covered ? 'Hit ✓' : 'Miss ✗'}`,
                    }
                }
            },
            scales: buildScales(C, false),
        },
    });
}

// ── HISTORY ────────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch('/api/prediction/history?limit=50');
        const rows = await res.json();

        const hits = rows.filter(r => r.hit === 1).length;
        const total = rows.filter(r => r.hit !== null).length;
        if (total > 0) {
            $('live-accuracy').textContent = `Live Accuracy: ${(hits / total * 100).toFixed(1)}%`;
        }

        const tbody = $('history-tbody');
        tbody.innerHTML = '';
        rows.forEach((r, i) => {
            const width = r.lower && r.upper ? r.upper - r.lower : null;
            let resultHtml = '<span class="hit-pending">Pending</span>';
            if (r.hit === 1) resultHtml = '<span class="hit-yes">✓ Hit</span>';
            else if (r.hit === 0) resultHtml = '<span class="hit-no">✗ Miss</span>';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td>${new Date(r.timestamp).toLocaleString()}</td>
                <td>${r.price ? fmtShort(r.price) : '—'}</td>
                <td class="hit-no">${r.lower ? fmtShort(r.lower) : '—'}</td>
                <td class="hit-yes">${r.upper ? fmtShort(r.upper) : '—'}</td>
                <td>${width ? fmt(width) : '—'}</td>
                <td>${r.actual ? fmtShort(r.actual) : '—'}</td>
                <td>${resultHtml}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) { console.error('History load failed', e); }
}

// ── Shared chart helpers ───────────────────────────────────────────
function baseChartOptions(extra = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        ...extra,
    };
}

function buildScales(C, withYLabel) {
    return {
        x: {
            grid: { color: C.grid },
            ticks: { color: C.text, font: { family: 'Outfit', size: 10 }, maxTicksLimit: 10 },
        },
        y: {
            grid: { color: C.grid },
            ticks: { color: C.text, font: { family: 'Outfit', size: 10 } },
        },
    };
}

// ── Auto-refresh every 60 seconds ──────────────────────────────────
setInterval(loadCurrentPrediction, 60000);

// ── Init ───────────────────────────────────────────────────────────
loadCurrentPrediction();

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 1 — Prediction Decay Countdown Timer
// ══════════════════════════════════════════════════════════════════
function startCountdown() {
    function tick() {
        const now = new Date();
        const secPast = now.getMinutes() * 60 + now.getSeconds();
        const secLeft = 3600 - (secPast % 3600);
        const min = Math.floor(secLeft / 60).toString().padStart(2, '0');
        const sec = (secLeft % 60).toString().padStart(2, '0');
        const textEl = document.getElementById('countdown-text');
        const arc = document.getElementById('countdown-arc');
        if (textEl) textEl.textContent = `${min}:${sec}`;
        if (arc) {
            const pct = secLeft / 3600;
            const circumference = 2 * Math.PI * 18; // r=18
            arc.style.strokeDashoffset = circumference * (1 - pct);
            arc.style.stroke = pct > 0.5 ? 'var(--accent)' : pct > 0.2 ? 'var(--yellow)' : 'var(--red)';
        }
    }
    tick();
    setInterval(tick, 1000);
}
startCountdown();

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 2 — Market Regime Classifier
// ══════════════════════════════════════════════════════════════════
async function loadRegime() {
    try {
        const res = await fetch('/api/regime');
        const d = await res.json();
        const $ = (id) => document.getElementById(id);
        const regimeColors = { 'Trending': '#f97316', 'Range-Bound': '#10b981', 'Pre-Breakout': '#eab308', 'High-Fear': '#ef4444' };
        const regimeDescs = {
            'Trending': 'Strong directional momentum. Model widens range in trend direction.',
            'Range-Bound': 'Price oscillating near mean. Mean-reversion likely. Expect tighter ranges.',
            'Pre-Breakout': 'Volatility accelerating from low base. A significant move may be imminent.',
            'High-Fear': 'Extreme negative momentum and high volatility. Exercise caution.'
        };
        $('regime-dominant').textContent = d.dominant_regime;
        $('regime-dominant').style.color = regimeColors[d.dominant_regime] || 'var(--accent)';
        $('regime-desc').textContent = regimeDescs[d.dominant_regime] || '';

        // Probability bars
        const barsEl = $('regime-prob-bars');
        barsEl.innerHTML = '';
        Object.entries(d.regime_probabilities).forEach(([name, pct]) => {
            const color = regimeColors[name] || 'var(--accent)';
            barsEl.innerHTML += `
                <div class="regime-prob-bar-wrap">
                    <div class="rpb-label"><span>${name}</span><span style="color:${color};font-weight:700">${pct}%</span></div>
                    <div class="regime-prob-bar-track">
                        <div class="regime-prob-bar-fill" style="width:${pct}%;background:${color}"></div>
                    </div>
                </div>`;
        });

        // Signal cards
        const m = d.momentum;
        $('sig-momentum-dir').textContent = m.direction;
        $('sig-momentum-dir').style.color = m.direction === 'Bullish' ? 'var(--green)' : 'var(--red)';
        $('sig-momentum-str').textContent = `${m.strength}%`;
        $('sig-momentum-10').textContent = `${m.rolling_10h_pct > 0 ? '+' : ''}${m.rolling_10h_pct}%`;
        $('sig-momentum-run').textContent = `${m.consecutive_bars} bars`;

        const mr = d.mean_reversion;
        $('sig-zscor').textContent = mr.z_score.toFixed(3);
        $('sig-mean').textContent = `$${mr['48h_mean'].toLocaleString()}`;
        const diff = mr.price_vs_48h_mean;
        $('sig-vs-mean').textContent = `${diff >= 0 ? '+' : ''}$${diff.toFixed(2)}`;
        $('sig-vs-mean').style.color = diff >= 0 ? 'var(--green)' : 'var(--red)';

        const va = d.volatility_accel;
        $('sig-vol-ratio').textContent = va.ratio.toFixed(3) + '×';
        $('sig-vol-status').textContent = va.interpretation;
        $('sig-range-pct').textContent = `${d.range_24h.range_pct.toFixed(3)}%`;

        const lev = d.levels;
        const fmtLev = (v) => v ? `$${v.toLocaleString()}` : '—';
        $('sig-r1').textContent = fmtLev(lev.resistances[0]);
        $('sig-r2').textContent = fmtLev(lev.resistances[1]);
        $('sig-s1').textContent = fmtLev(lev.supports[0]);
        $('sig-s2').textContent = fmtLev(lev.supports[1]);
    } catch (e) { console.error('Regime load failed', e); }
}

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 3 — What-If Scenario Studio
// ══════════════════════════════════════════════════════════════════
let scenarioChart = null;
let scenarioDebounce = null;
let baseModelWidth = null;

async function runScenario() {
    const volMultiplier = parseFloat(document.getElementById('sl-vol').value);
    const driftBias = parseFloat(document.getElementById('sl-drift').value);
    const confLevel = parseFloat(document.getElementById('sl-conf').value) / 100;

    document.getElementById('sl-vol-val').textContent = `${volMultiplier.toFixed(1)}×`;
    document.getElementById('sl-drift-val').textContent = `${driftBias >= 0 ? '+' : ''}${driftBias.toFixed(2)}%`;
    document.getElementById('sl-conf-val').textContent = `${(confLevel * 100).toFixed(0)}%`;

    try {
        const res = await fetch('/api/scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ vol_multiplier: volMultiplier, drift_bias_pct: driftBias, confidence_level: confLevel, num_simulations: 10000 })
        });
        const d = await res.json();
        document.getElementById('sc-lower').textContent = fmtShort(d.lower);
        document.getElementById('sc-upper').textContent = fmtShort(d.upper);
        document.getElementById('sc-width').textContent = `Width: ${fmt(d.width)}`;
        if (baseModelWidth !== null) {
            const delta = d.width - baseModelWidth;
            const vsEl = document.getElementById('sc-vs');
            vsEl.textContent = `${delta >= 0 ? '+' : ''}${fmt(delta)}`;
            vsEl.style.color = delta > 0 ? 'var(--red)' : 'var(--green)';
        }
        renderScenarioChart(d);
    } catch (e) { console.error('Scenario failed', e); }
}

function renderScenarioChart(d) {
    const ctx = document.getElementById('scenarioChart').getContext('2d');
    const C = chartColors();
    const hist = d.histogram;
    const inBand = hist.bin_centers.map(bc => bc >= d.lower && bc <= d.upper ? 'rgba(16,185,129,0.6)' : 'rgba(150,150,150,0.2)');
    if (scenarioChart) scenarioChart.destroy();
    scenarioChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hist.bin_centers.map(v => fmtShort(v)),
            datasets: [{ label: 'Simulated Prices', data: hist.counts, backgroundColor: inBand, borderRadius: 3 }]
        },
        options: { ...baseChartOptions(), plugins: { legend: { display: false } }, scales: buildScales(C, false) }
    });
}

function initScenarioSliders() {
    ['sl-vol', 'sl-drift', 'sl-conf'].forEach(id => {
        document.getElementById(id).addEventListener('input', () => {
            clearTimeout(scenarioDebounce);
            scenarioDebounce = setTimeout(runScenario, 300);
        });
    });
}
initScenarioSliders();

// ══════════════════════════════════════════════════════════════════
// UNIQUE FEATURE 4 — Tail Risk Dashboard
// ══════════════════════════════════════════════════════════════════
let tailChart = null;

async function loadTailRisk() {
    try {
        const res = await fetch('/api/tail-risk');
        const d = await res.json();
        const $ = (id) => document.getElementById(id);

        $('tr-var95').textContent = `${fmt(d.var.var_95_dollar)} (${d.var.var_95_pct}%)`;
        $('tr-var99').textContent = `${fmt(d.var.var_99_dollar)} (${d.var.var_99_pct}%)`;
        $('tr-cvar95').textContent = `${fmt(d.cvar.cvar_95_dollar)} (${d.cvar.cvar_95_pct}%)`;
        $('tr-cvar99').textContent = `${fmt(d.cvar.cvar_99_dollar)} (${d.cvar.cvar_99_pct}%)`;

        const mp = d.move_probabilities;
        $('tr-up1').textContent = `${mp.up_1pct.toFixed(2)}%`;
        $('tr-dn1').textContent = `${mp.down_1pct.toFixed(2)}%`;
        $('tr-up2').textContent = `${mp.up_2pct.toFixed(2)}%`;
        $('tr-dn2').textContent = `${mp.down_2pct.toFixed(2)}%`;

        const ds = d.distribution_shape;
        $('tr-skew').textContent = ds.skewness.toFixed(4);
        $('tr-kurt').textContent = ds.excess_kurtosis.toFixed(4);
        $('tr-df').textContent = ds.student_t_df.toFixed(2);
        $('tr-annvol').textContent = `${d.annualised_vol_pct.toFixed(2)}%`;

        $('tr-skew-interp').textContent = `⟳ Skewness: ${ds.interpretation_skew}`;
        $('tr-kurt-interp').textContent = `⟳ Kurtosis: ${ds.interpretation_kurt}`;
        $('tr-maxdd').textContent = `📉 Max Drawdown (24h): ${d.max_drawdown_24h_pct.toFixed(4)}%`;

        // Percentile distribution chart
        const ctx = $('tailChart').getContext('2d');
        const C = chartColors();
        const pctData = d.price_percentile_distribution;
        const labels = Object.keys(pctData).map(k => `${k}th`);
        const values = Object.values(pctData);
        const barColors = values.map((v, i) => {
            const pct = parseInt(Object.keys(pctData)[i]);
            if (pct <= 5) return C.red;
            if (pct >= 95) return C.green;
            if (pct === 50) return C.accent;
            return 'rgba(150,150,150,0.4)';
        });
        if (tailChart) tailChart.destroy();
        tailChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{ label: 'Price at Percentile', data: values, backgroundColor: barColors, borderRadius: 4 }]
            },
            options: {
                ...baseChartOptions(),
                plugins: {
                    legend: { display: false },
                    tooltip: { callbacks: { label: ctx => `Price: ${fmtShort(ctx.parsed.y)}` } }
                },
                scales: buildScales(C, false)
            }
        });
    } catch (e) { console.error('Tail risk load failed', e); }
}

// ── Extend activateSection to load new sections ──────────────────
const _origActivate = activateSection;
// Patch nav to call new loaders
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const sec = btn.dataset.section;
        if (sec === 'regime') loadRegime();
        if (sec === 'scenario') { if (cachedPrediction) { baseModelWidth = cachedPrediction.prediction.width; } runScenario(); }
        if (sec === 'tailrisk') loadTailRisk();
    });
});

// ══════════════════════════════════════════════════════════════════
// CURRENCY SYSTEM — Live exchange rates from frankfurter.app
// ══════════════════════════════════════════════════════════════════

// Currency config: symbol, decimals
const CURRENCY_META = {
    USD: { symbol: '$',  locale: 'en-US', decimals: 2 },
    EUR: { symbol: '€',  locale: 'de-DE', decimals: 2 },
    GBP: { symbol: '£',  locale: 'en-GB', decimals: 2 },
    INR: { symbol: '₹',  locale: 'en-IN', decimals: 0 },
    JPY: { symbol: '¥',  locale: 'ja-JP', decimals: 0 },
    CNY: { symbol: '¥',  locale: 'zh-CN', decimals: 2 },
    AUD: { symbol: 'A$', locale: 'en-AU', decimals: 2 },
    CAD: { symbol: 'C$', locale: 'en-CA', decimals: 2 },
    CHF: { symbol: 'Fr', locale: 'de-CH', decimals: 2 },
    SGD: { symbol: 'S$', locale: 'en-SG', decimals: 2 },
    AED: { symbol: 'د.إ',locale: 'ar-AE', decimals: 2 },
    SAR: { symbol: '﷼',  locale: 'ar-SA', decimals: 2 },
    BRL: { symbol: 'R$', locale: 'pt-BR', decimals: 2 },
    MXN: { symbol: '$',  locale: 'es-MX', decimals: 2 },
    KRW: { symbol: '₩',  locale: 'ko-KR', decimals: 0 },
    HKD: { symbol: 'HK$',locale: 'zh-HK', decimals: 2 },
    SEK: { symbol: 'kr', locale: 'sv-SE', decimals: 2 },
    NOK: { symbol: 'kr', locale: 'nb-NO', decimals: 2 },
    DKK: { symbol: 'kr', locale: 'da-DK', decimals: 2 },
    NZD: { symbol: 'NZ$',locale: 'en-NZ', decimals: 2 },
    ZAR: { symbol: 'R',  locale: 'en-ZA', decimals: 2 },
    TRY: { symbol: '₺',  locale: 'tr-TR', decimals: 2 },
    IDR: { symbol: 'Rp', locale: 'id-ID', decimals: 0 },
    MYR: { symbol: 'RM', locale: 'ms-MY', decimals: 2 },
    THB: { symbol: '฿',  locale: 'th-TH', decimals: 2 },
    PHP: { symbol: '₱',  locale: 'fil-PH',decimals: 2 },
    PLN: { symbol: 'zł', locale: 'pl-PL', decimals: 2 },
    CZK: { symbol: 'Kč', locale: 'cs-CZ', decimals: 2 },
    HUF: { symbol: 'Ft', locale: 'hu-HU', decimals: 0 },
    ILS: { symbol: '₪',  locale: 'he-IL', decimals: 2 },
};

let activeCurrency = 'USD';
let exchangeRates = { USD: 1 }; // rates relative to USD

async function fetchExchangeRates() {
    try {
        // frankfurter.app — free, no key, updated daily
        const res = await fetch('https://api.frankfurter.app/latest?from=USD');
        if (!res.ok) throw new Error('Rate fetch failed');
        const data = await res.json();
        exchangeRates = { USD: 1, ...data.rates };
        console.log('Exchange rates loaded:', exchangeRates);
    } catch (e) {
        console.warn('Could not fetch live rates, using fallback', e);
        // Reasonable fallback rates (approximate)
        exchangeRates = {
            USD:1, EUR:0.92, GBP:0.79, INR:83.5, JPY:149.8, CNY:7.24,
            AUD:1.53, CAD:1.36, CHF:0.90, SGD:1.34, AED:3.67, SAR:3.75,
            BRL:4.97, MXN:17.2, KRW:1330, HKD:7.82, SEK:10.4, NOK:10.6,
            DKK:6.89, NZD:1.63, ZAR:18.7, TRY:32.1, IDR:15700, MYR:4.72,
            THB:35.2, PHP:56.5, PLN:3.97, CZK:23.1, HUF:360, ILS:3.65,
        };
    }
    updateRateBadge();
    refreshAllDisplayedPrices();
}

function updateRateBadge() {
    const rate = exchangeRates[activeCurrency] || 1;
    const meta = CURRENCY_META[activeCurrency] || CURRENCY_META.USD;
    const badge = document.getElementById('exchange-rate-badge');
    if (badge) badge.textContent = `1 USD = ${rate.toLocaleString(meta.locale, { maximumFractionDigits: 4 })} ${activeCurrency}`;
}

// Override the global fmt/fmtShort to respect currency
window.convertPrice = function(usdAmount) {
    const rate = exchangeRates[activeCurrency] || 1;
    return usdAmount * rate;
};

window.fmt = function(n) {
    const converted = convertPrice(n);
    const meta = CURRENCY_META[activeCurrency] || CURRENCY_META.USD;
    return new Intl.NumberFormat(meta.locale, {
        style: 'currency', currency: activeCurrency,
        minimumFractionDigits: meta.decimals,
        maximumFractionDigits: meta.decimals,
    }).format(converted);
};

window.fmtShort = function(n) {
    const converted = convertPrice(n);
    const meta = CURRENCY_META[activeCurrency] || CURRENCY_META.USD;
    return new Intl.NumberFormat(meta.locale, {
        style: 'currency', currency: activeCurrency,
        minimumFractionDigits: meta.decimals,
        maximumFractionDigits: meta.decimals,
    }).format(converted);
};

function refreshAllDisplayedPrices() {
    if (!cachedPrediction) return;
    updateHeroCard(cachedPrediction);
    updateTopbar(cachedPrediction);
    updateRateBadge();
    renderPriceChart(cachedPrediction);
    // Re-load metrics if visible
    loadMetrics();
}

// Currency selector listener
document.getElementById('currency-select').addEventListener('change', (e) => {
    activeCurrency = e.target.value;
    updateRateBadge();
    refreshAllDisplayedPrices();
    // Also update visible section values
    const activeSection = document.querySelector('.section.active');
    if (activeSection) {
        const id = activeSection.id;
        if (id === 'section-backtest') loadBacktest();
        if (id === 'section-history') loadHistory();
        if (id === 'section-tailrisk') loadTailRisk();
        if (id === 'section-scenario') runScenario();
    }
});

// Init exchange rates on load
fetchExchangeRates();
