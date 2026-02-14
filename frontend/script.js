document.addEventListener('DOMContentLoaded', () => {
    // === TABS & VIEWS ===
    const tabs = document.querySelectorAll('.mode-tab');
    const views = document.querySelectorAll('.view-section');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            views.forEach(v => v.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(tab.dataset.target).classList.add('active');

            // If clicking analytics, load data if not loaded
            if (tab.dataset.target === 'analytics-view') {
                loadAnalytics();
            }
        });
    });

    // === ANALYTICS & DASHBOARD LOGIC ===
    let metricsData = null;
    let rocChart = null;
    let featureChart = null;

    async function loadAnalytics() {
        if (metricsData) return; // Already loaded via cache

        try {
            const res = await fetch('/analytics/metrics');
            if (!res.ok) throw new Error("Failed to load metrics. Ensure model is trained.");
            metricsData = await res.json();

            renderDashboard(metricsData);
        } catch (e) {
            console.error("Analytics Error:", e);
            // alert("Could not fetch analytics data.");
        }
    }

    function renderDashboard(data) {
        // 1. Summary Cards
        document.getElementById('dash-accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
        document.getElementById('dash-auc').textContent = data.auc.toFixed(3);
        document.getElementById('dash-recall').textContent = (data.recall * 100).toFixed(1) + '%';

        // 2. ROC Chart
        const ctxRoc = document.getElementById('rocChart').getContext('2d');
        if (rocChart) rocChart.destroy();

        rocChart = new Chart(ctxRoc, {
            type: 'line',
            data: {
                labels: data.roc_curve.fpr.map(v => v.toFixed(2)), // X-axis: FPR
                datasets: [{
                    label: 'ROC Curve (AUC = ' + data.auc.toFixed(2) + ')',
                    data: data.roc_curve.tpr, // Y-Axis: TPR
                    borderColor: '#2563EB',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.1
                }, {
                    label: 'Random Chance',
                    data: data.roc_curve.fpr, // Diagonal
                    borderColor: '#9CA3AF',
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
                    y: { title: { display: true, text: 'True Positive Rate (Recall)' }, min: 0, max: 1 }
                },
                plugins: { legend: { position: 'bottom' } }
            }
        });

        // 3. Feature Importance Chart
        const ctxFeat = document.getElementById('featureChart').getContext('2d');
        if (featureChart) featureChart.destroy();

        featureChart = new Chart(ctxFeat, {
            type: 'bar',
            data: {
                labels: data.feature_importance.map(item => item.feature),
                datasets: [{
                    label: 'Importance',
                    data: data.feature_importance.map(item => item.importance),
                    backgroundColor: '#1E40AF',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y', // Horizontal Bar
                plugins: { legend: { display: false } }
            }
        });

        // 4. Threshold Slider Logic
        const slider = document.getElementById('threshold-slider');
        const valDisplay = document.getElementById('threshold-val');

        // Initial Render (Threshold 0.5)
        updateThresholdImpact(0.5);

        slider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value) / 10; // 1-9 -> 0.1-0.9
            valDisplay.textContent = val.toFixed(1);
            updateThresholdImpact(val);
        });

        // 5. Confusion Matrix (Initial 0.5)
        updateCM(0.5);
    }

    function updateThresholdImpact(threshold) {
        if (!metricsData) return;

        // Find closest threshold in analysis data
        // Our analysis data has steps, let's just find the closest
        const closest = metricsData.threshold_analysis.reduce((prev, curr) => {
            return (Math.abs(curr.threshold - threshold) < Math.abs(prev.threshold - threshold) ? curr : prev);
        });

        document.getElementById('impact-recall').textContent = (closest.recall * 100).toFixed(1) + '%';
        document.getElementById('impact-fp').textContent = closest.fp.toLocaleString();
        document.getElementById('impact-fn').textContent = closest.fn.toLocaleString();

        updateCM_UI(closest.tn, closest.fp, closest.fn, closest.tp);
    }

    function updateCM(threshold) {
        // Reuse the logic above since we map threshold to CM stats
        updateThresholdImpact(threshold);
    }

    function updateCM_UI(tn, fp, fn, tp) {
        document.getElementById('cm-tn').querySelector('.value').textContent = tn.toLocaleString();
        document.getElementById('cm-fp').querySelector('.value').textContent = fp.toLocaleString();
        document.getElementById('cm-fn').querySelector('.value').textContent = fn.toLocaleString();
        document.getElementById('cm-tp').querySelector('.value').textContent = tp.toLocaleString();
    }

    // === PAYMENT SIMULATION LOGIC (Existing) ===
    const paymentForm = document.getElementById('payment-form');
    const payBtn = document.getElementById('pay-btn');
    if (paymentForm && payBtn) {
        // Formatting (Same as before)
        const cardInput = document.getElementById('card_number');
        if (cardInput) {
            cardInput.addEventListener('input', (e) => {
                let val = e.target.value.replace(/\D/g, '');
                let newVal = '';
                for (let i = 0; i < val.length; i++) {
                    if (i > 0 && i % 4 === 0) newVal += ' ';
                    newVal += val[i];
                }
                e.target.value = newVal;
            });
        }
        const expiryInput = document.getElementById('expiry');
        if (expiryInput) {
            expiryInput.addEventListener('input', (e) => {
                let val = e.target.value.replace(/\D/g, '');
                if (val.length >= 2) {
                    val = val.substring(0, 2) + '/' + val.substring(2);
                }
                e.target.value = val;
            });
        }

        const loader = payBtn.querySelector('.loader');
        const btnText = payBtn.querySelector('.btn-text');
        const initialState = document.getElementById('initial-state');
        const resultState = document.getElementById('result-state');
        const resetBtn = document.getElementById('reset-btn');

        // Result Elements
        const statusIcon = document.getElementById('status-icon');
        const statusTitle = document.getElementById('status-title');
        const statusMessage = document.getElementById('status-message');
        const probDisplay = document.getElementById('prob-display');
        const meterFill = document.getElementById('meter-fill');
        const confidenceDisplay = document.getElementById('confidence-display');
        const timeDisplay = document.getElementById('time-display');

        paymentForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            btnText.textContent = 'Scoring...';
            loader.classList.remove('hidden');
            payBtn.disabled = true;

            const formData = new FormData(paymentForm);
            const data = {
                card_number: formData.get('card_number').replace(/\s/g, ''),
                expiry_date: formData.get('expiry_date'),
                cvv: formData.get('cvv'),
                amount: parseFloat(formData.get('amount'))
            };

            try {
                await new Promise(r => setTimeout(r, 600));
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                if (!response.ok) throw new Error((await response.json()).detail || 'Failed');
                const result = await response.json();

                initialState.classList.add('hidden');
                resultState.classList.remove('hidden');

                if (result.label === 'Transaction Flagged') {
                    statusIcon.textContent = '🚨';
                    statusTitle.textContent = 'High Fraud Risk';
                    statusTitle.style.color = '#DC2626';
                    statusMessage.textContent = 'Model flagged suspicious patterns.';
                    meterFill.style.backgroundColor = '#DC2626';
                } else {
                    statusIcon.textContent = '✅';
                    statusTitle.textContent = 'Low Risk';
                    statusTitle.style.color = '#059669';
                    statusMessage.textContent = 'Transaction appears legitimate.';
                    meterFill.style.backgroundColor = '#059669';
                }

                const probPercent = (result.probability * 100).toFixed(1) + '%';
                probDisplay.textContent = probPercent;
                meterFill.style.width = probPercent;

                let confidence = 'Medium';
                if (result.probability > 0.8 || result.probability < 0.2) confidence = 'High';
                if (result.probability > 0.4 && result.probability < 0.6) confidence = 'Low';
                confidenceDisplay.textContent = confidence;
                timeDisplay.textContent = result.processing_time_ms.toFixed(0) + 'ms';

            } catch (error) {
                alert(`Error: ${error.message}`);
            } finally {
                btnText.textContent = 'Score Transaction';
                loader.classList.add('hidden');
                payBtn.disabled = false;
            }
        });

        if (resetBtn) resetBtn.addEventListener('click', () => {
            resultState.classList.add('hidden');
            initialState.classList.remove('hidden');
            paymentForm.reset();
        });
    }

    // === ML DEBUGGER LOGIC (Existing re-integrated) ===
    const featuresForm = document.getElementById('features-form');
    const vFeaturesContainer = document.getElementById('v-features-container');

    if (vFeaturesContainer && vFeaturesContainer.children.length === 0) {
        for (let i = 1; i <= 28; i++) {
            const div = document.createElement('div');
            div.className = 'feature-input';
            div.innerHTML = `
                <label>V${i}</label>
                <input type="number" step="any" name="V${i}" placeholder="0.0">
            `;
            vFeaturesContainer.appendChild(div);
        }
    }

    const loadRandomBtn = document.getElementById('load-random-btn');
    if (loadRandomBtn) {
        loadRandomBtn.addEventListener('click', async () => {
            try {
                const res = await fetch('/random-sample');
                if (!res.ok) throw new Error("Failed to fetch sample");
                const data = await res.json();

                if (data.Time) document.getElementById('feat-Time').value = data.Time;
                if (data.Amount) document.getElementById('feat-Amount').value = data.Amount;

                for (let i = 1; i <= 28; i++) {
                    const input = document.querySelector(`input[name="V${i}"]`);
                    if (input && data[`V${i}`] !== undefined) {
                        input.value = data[`V${i}`];
                    }
                }
            } catch (e) {
                alert("Error loading sample: " + e.message);
            }
        });
    }

    if (featuresForm) {
        featuresForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(featuresForm);
            const data = {};
            data['Time'] = parseFloat(formData.get('Time') || 0);
            data['Amount'] = parseFloat(formData.get('Amount') || 0);
            for (let i = 1; i <= 28; i++) {
                data[`V${i}`] = parseFloat(formData.get(`V${i}`) || 0);
            }

            try {
                const res = await fetch('/predict-features', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                if (!res.ok) throw new Error((await res.json()).detail);
                const result = await res.json();

                const resDiv = document.getElementById('debugger-result');
                const badge = document.getElementById('debug-badge');
                const meta = document.getElementById('debug-meta');

                resDiv.classList.remove('hidden');
                badge.textContent = result.label;
                if (result.label === 'Fraudulent') {
                    badge.className = 'status-badge status-fraud';
                } else {
                    badge.className = 'status-badge status-normal';
                }

                meta.innerHTML = `Prob: <strong>${(result.probability * 100).toFixed(1)}%</strong> | Time: ${result.processing_time_ms.toFixed(0)}ms`;

            } catch (e) {
                alert("Error: " + e.message);
            }
        });
    }

    const batchUpload = document.getElementById('csv-upload');
    if (batchUpload) {
        batchUpload.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/predict-batch', {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) throw new Error((await res.json()).detail);
                const result = await res.json();

                const batchRes = document.getElementById('batch-results');
                batchRes.classList.remove('hidden');

                document.getElementById('batch-total').textContent = result.total_transactions;
                document.getElementById('batch-fraud').textContent = result.fraud_count;
                document.getElementById('batch-percent').textContent = result.fraud_percentage.toFixed(2) + '%';

            } catch (e) {
                alert("Batch error: " + e.message);
            }
        });
    }

    // Toggle Model Details
    const toggleModelBtn = document.getElementById('toggle-model-details');
    const modelPanel = document.getElementById('model-details-panel');
    if (toggleModelBtn) {
        toggleModelBtn.addEventListener('click', () => {
            modelPanel.classList.toggle('hidden');
        });
    }
});
