/**
 * app.js — Heart Disease Prediction & ML Analysis Dashboard
 * ════════════════════════════════════════════════════════════════════════════
 * Handles:
 *  - Fetching statistics and pre-calculated model results from /api/stats
 *  - Rendering 6 comprehensive tabs:
 *      1. Data Overview + Overall Diagrams (2D PCA Feature Space & Clinical Radar)
 *      2. Feature Distributions (Histograms + Categorical Counts + Box Plot)
 *      3. Correlations & Relationships (Heatmap + Scatters by Target)
 *      4. Feature vs Target (8 Grouped Bar Charts)
 *      5. Hyperparameter Tuning & Kernels (GridSearchCV Heatmap, Sensitivity Curves,
 *         Kernel Radar, Kernel ROC Curves, Interactive Confusion Matrix, 5 Classifiers)
 *      6. Patient Prediction (Inference with Tuned SVM / selected Kernel)
 * ════════════════════════════════════════════════════════════════════════════
 */

document.addEventListener("DOMContentLoaded", () => {
    let DATA = null;
    const charts = {};

    // Cohesive dark theme color palette
    const P = {
        blue:   "#3b82f6",
        purple: "#8b5cf6",
        teal:   "#06b6d4",
        green:  "#10b981",
        orange: "#f59e0b",
        red:    "#ef4444",
        pink:   "#ec4899",
        gray:   "#64748b",
    };
    const gridColor = "rgba(0, 0, 0, 0.08)";
    const textColor = "#475569";

    // Chart.js global defaults
    Chart.defaults.color = textColor;
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.plugins.tooltip.backgroundColor = "rgba(255, 255, 255, 0.95)";
    Chart.defaults.plugins.tooltip.titleColor = "#1e293b";
    Chart.defaults.plugins.tooltip.bodyColor = "#475569";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(0, 0, 0, 0.1)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;

    function mkChart(id, type, data, options = {}) {
        if (charts[id]) {
            charts[id].destroy();
        }
        const canvas = document.getElementById(id);
        if (!canvas) return null;
        const ctx = canvas.getContext("2d");
        const defaultOpts = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#475569", font: { size: 11, weight: 600 } } }
            }
        };
        charts[id] = new Chart(ctx, {
            type,
            data,
            options: deepMerge(defaultOpts, options)
        });
        return charts[id];
    }

    function deepMerge(target, source) {
        const out = Object.assign({}, target);
        for (const key of Object.keys(source)) {
            if (source[key] instanceof Object && !Array.isArray(source[key])) {
                out[key] = deepMerge(target[key] || {}, source[key]);
            } else {
                out[key] = source[key];
            }
        }
        return out;
    }

    // ── Tab Switching Logic ──
    const renderedTabs = {};
    document.querySelectorAll(".tab").forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.tab;
            document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".pane").forEach(p => p.classList.remove("active"));
            tab.classList.add("active");
            const pane = document.getElementById("tab-" + target);
            if (pane) pane.classList.add("active");
            renderTab(target);
        });
    });

    function renderTab(tab) {
        if (renderedTabs[tab]) return;
        renderedTabs[tab] = true;

        if (tab === "overview")      renderOverview();
        if (tab === "distributions") renderDistributions();
        if (tab === "relationships") renderRelationships();
        if (tab === "target")        renderTargetAnalysis();
        if (tab === "models")        renderModels();
        if (tab === "predict")       initForm();
    }

    // ── Fetch Initial Data ──
    fetch("/api/stats")
        .then(r => r.json())
        .then(data => {
            DATA = data;
            document.getElementById("hdr-records").textContent = data.total_records || 297;
            document.getElementById("hdr-features").textContent = data.features_count || 13;
            renderOverview();
            renderedTabs["overview"] = true;
            initSampleButton();
        })
        .catch(err => console.error("Failed to load dataset stats:", err));

    /* ═══════════════════════════════════════════════════════════════
       TAB 1 — DATA OVERVIEW & OVERALL DIAGRAMS
       ═══════════════════════════════════════════════════════════════ */
    function renderOverview() {
        const cd = DATA.class_distribution;
        setText("ov-total", DATA.total_records);
        setText("ov-features", DATA.features_count);
        setText("ov-no", cd["No Disease (0)"]);
        setText("ov-yes", cd["Heart Disease (1)"]);

        // 1. Overall Diagram: 2D PCA Patient Projection
        const overall = DATA.overall_diagrams;
        if (overall && overall.pca) {
            const pcaRecs = overall.pca.records;
            const pts0 = [];
            const pts1 = [];
            pcaRecs.forEach(r => {
                const pt = { x: r.pc1, y: r.pc2, age: r.age, sex: r.sex };
                if (r.hd === 1) pts1.push(pt);
                else pts0.push(pt);
            });

            mkChart("ch-pca-projection", "scatter", {
                datasets: [
                    {
                        label: "No Disease (Class 0)",
                        data: pts0,
                        backgroundColor: "rgba(16, 185, 129, 0.75)",
                        borderColor: "#10b981",
                        pointRadius: 4.5,
                        pointHoverRadius: 6,
                    },
                    {
                        label: "Heart Disease (Class 1)",
                        data: pts1,
                        backgroundColor: "rgba(239, 68, 68, 0.75)",
                        borderColor: "#ef4444",
                        pointRadius: 4.5,
                        pointHoverRadius: 6,
                    }
                ]
            }, {
                plugins: {
                    legend: { position: "top" },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const pt = ctx.raw;
                                return `Patient: Age ${pt.age}, ${pt.sex} | (PC1: ${pt.x}, PC2: ${pt.y})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: `Principal Component 1 (${overall.pca.variance_ratio[0]}% var)`, color: textColor },
                        grid: { color: gridColor }, ticks: { color: textColor }
                    },
                    y: {
                        title: { display: true, text: `Principal Component 2 (${overall.pca.variance_ratio[1]}% var)`, color: textColor },
                        grid: { color: gridColor }, ticks: { color: textColor }
                    }
                }
            });

            const varText = document.getElementById("pca-var-text");
            if (varText) {
                varText.innerHTML = `<strong>Variance Explained:</strong> PC1 = ${overall.pca.variance_ratio[0]}%, PC2 = ${overall.pca.variance_ratio[1]}% (Cumulative: ${(overall.pca.variance_ratio[0] + overall.pca.variance_ratio[1]).toFixed(1)}%)`;
            }
        }

        // 2. Overall Diagram: Clinical Biomarker Radar
        if (overall && overall.clinical_radar) {
            const rad = overall.clinical_radar;
            mkChart("ch-clinical-radar", "radar", {
                labels: rad.labels,
                datasets: [
                    {
                        label: "Healthy Cohort (No Disease)",
                        data: rad.healthy,
                        backgroundColor: "rgba(16, 185, 129, 0.25)",
                        borderColor: "#10b981",
                        pointBackgroundColor: "#10b981",
                        pointBorderColor: "#fff",
                        borderWidth: 2,
                    },
                    {
                        label: "Heart Disease Cohort",
                        data: rad.disease,
                        backgroundColor: "rgba(239, 68, 68, 0.25)",
                        borderColor: "#ef4444",
                        pointBackgroundColor: "#ef4444",
                        pointBorderColor: "#fff",
                        borderWidth: 2,
                    }
                ]
            }, {
                plugins: { legend: { position: "top" } },
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        angleLines: { color: gridColor },
                        grid: { color: gridColor },
                        pointLabels: { color: "#475569", font: { size: 11, weight: 600 } },
                        ticks: { display: false }
                    }
                }
            });
        }

        // 3. Class Distribution Doughnut
        mkChart("ch-classDist", "doughnut", {
            labels: ["No Heart Disease (0)", "Heart Disease (1)"],
            datasets: [{
                data: [cd["No Disease (0)"], cd["Heart Disease (1)"]],
                backgroundColor: [P.green, P.red],
                borderColor: "#ffffff",
                borderWidth: 3,
            }]
        }, {
            plugins: {
                legend: { position: "bottom" },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const total = cd["No Disease (0)"] + cd["Heart Disease (1)"];
                            const pct = ((ctx.raw / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${ctx.raw} (${pct}%)`;
                        }
                    }
                }
            },
            cutout: "68%"
        });

        // 4. Numerical Stats Table
        const tbody = document.querySelector("#num-stats-table tbody");
        if (tbody) {
            tbody.innerHTML = "";
            const labels = {
                age: "Age (years)",
                restbp: "Resting BP (mm Hg)",
                chol: "Cholesterol (mg/dl)",
                thalach: "Max Heart Rate (bpm)",
                oldpeak: "ST Depression (mm)",
            };
            for (const [col, s] of Object.entries(DATA.num_stats)) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td><strong>${labels[col] || col}</strong></td>
                    <td>${s.mean}</td><td>${s.median}</td><td>${s.std}</td>
                    <td>${s.min}</td><td>${s.q25}</td><td>${s.q75}</td><td>${s.max}</td>`;
                tbody.appendChild(tr);
            }
        }
    }

    /* ═══════════════════════════════════════════════════════════════
       TAB 2 — FEATURE DISTRIBUTIONS
       ═══════════════════════════════════════════════════════════════ */
    function renderDistributions() {
        const recs = DATA.records;

        // Histograms
        renderHistogram("ch-ageDist",     recs, "age",     "Age (years)", 10, P.blue);
        renderHistogram("ch-restbpDist",  recs, "restbp",  "Resting Blood Pressure", 10, P.purple);
        renderHistogram("ch-cholDist",    recs, "chol",    "Serum Cholesterol", 12, P.teal);
        renderHistogram("ch-thalachDist", recs, "thalach", "Max Heart Rate", 10, P.green);
        renderHistogram("ch-oldpeakDist", recs, "oldpeak", "ST Depression", 10, P.orange);

        // Categorical Value Counts
        const cc = DATA.cat_counts;
        renderBar("ch-sexCat",     ["Female (0)", "Male (1)"], [cc.sex["0"] || cc.sex["Female"] || 96, cc.sex["1"] || cc.sex["Male"] || 201], P.blue);
        renderBar("ch-cpCat",      Object.keys(cc.cp), Object.values(cc.cp), P.purple);
        renderBar("ch-fbsCat",     ["FBS <= 120 (0)", "FBS > 120 (1)"], Object.values(cc.fbs), P.teal);
        renderBar("ch-restecgCat", Object.keys(cc.restecg), Object.values(cc.restecg), P.green);
        renderBar("ch-exangCat",   ["No Angina (0)", "Exercise Angina (1)"], Object.values(cc.exang), P.orange);
        renderBar("ch-slopeCat",   Object.keys(cc.slope), Object.values(cc.slope), P.pink);
        renderBar("ch-caCat",      Object.keys(cc.ca).sort(), Object.keys(cc.ca).sort().map(k => cc.ca[k]), P.blue);
        renderBar("ch-thalCat",    Object.keys(cc.thal), Object.values(cc.thal), P.purple);

        // Five-number summary box-plot
        const numCols = [
            { k: "age",     label: "Age", scale: 1 },
            { k: "restbp",  label: "Rest BP", scale: 1 },
            { k: "chol",    label: "Cholesterol", scale: 1 },
            { k: "thalach", label: "Max HR", scale: 1 },
            { k: "oldpeak", label: "Oldpeak (x10)", scale: 10 },
        ];
        const colors = [P.blue, P.purple, P.teal, P.green, P.orange];
        const datasets = numCols.map((f, i) => {
            const s = DATA.num_stats[f.k];
            return {
                label: f.label,
                data: [s.min * f.scale, s.q25 * f.scale, s.median * f.scale, s.q75 * f.scale, s.max * f.scale],
                backgroundColor: colors[i],
                borderRadius: 4,
            };
        });

        mkChart("ch-boxAll", "bar", {
            labels: ["Min", "Q1", "Median", "Q3", "Max"],
            datasets
        }, {
            plugins: { legend: { position: "bottom" } },
            scales: barScales()
        });
    }

    function renderHistogram(canvasId, recs, field, label, numBins, color) {
        const vals = recs.map(r => parseFloat(r[field])).filter(v => !isNaN(v));
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        const binWidth = (max - min) / numBins;
        const bins = Array(numBins).fill(0);
        const binLabels = [];

        for (let i = 0; i < numBins; i++) {
            const bMin = (min + i * binWidth).toFixed(1);
            const bMax = (min + (i + 1) * binWidth).toFixed(1);
            binLabels.push(`${bMin}-${bMax}`);
        }

        vals.forEach(v => {
            let idx = Math.floor((v - min) / binWidth);
            if (idx >= numBins) idx = numBins - 1;
            bins[idx]++;
        });

        mkChart(canvasId, "bar", {
            labels: binLabels,
            datasets: [{
                label: `Patients (${label})`,
                data: bins,
                backgroundColor: color,
                borderRadius: 4,
            }]
        }, {
            plugins: { legend: { display: false } },
            scales: barScales()
        });
    }

    function renderBar(canvasId, labels, data, color) {
        mkChart(canvasId, "bar", {
            labels,
            datasets: [{
                data,
                backgroundColor: color,
                borderRadius: 4,
            }]
        }, {
            plugins: { legend: { display: false } },
            scales: barScales()
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       TAB 3 — CORRELATIONS & RELATIONSHIPS
       ═══════════════════════════════════════════════════════════════ */
    function renderRelationships() {
        const cm = DATA.correlation_matrix;
        if (cm) renderHeatmap(cm);

        const recs = DATA.records;
        scatterByTarget("ch-scatter-age-hr", recs, "age", "thalach", "Age", "Max Heart Rate");
        scatterByTarget("ch-scatter-chol-hr", recs, "chol", "thalach", "Cholesterol", "Max Heart Rate");
        scatterByTarget("ch-scatter-bp-chol", recs, "restbp", "chol", "Resting BP", "Cholesterol");
    }

    function renderHeatmap(cm) {
        const container = document.getElementById("corr-heatmap");
        const cols = cm.columns;
        const vals = cm.values;
        const labels = { age: "Age", restbp: "Rest BP", chol: "Chol", thalach: "Max HR", oldpeak: "Oldpeak", hd: "Target" };

        let html = '<table class="hm-table"><thead><tr><th></th>';
        cols.forEach(c => { html += `<th>${labels[c] || c}</th>`; });
        html += "</tr></thead><tbody>";
        vals.forEach((row, i) => {
            html += `<tr><th>${labels[cols[i]] || cols[i]}</th>`;
            row.forEach(v => {
                html += `<td style="background:${corrColor(v)};color:${Math.abs(v) > 0.4 ? '#fff' : '#475569'}">${v.toFixed(2)}</td>`;
            });
            html += "</tr>";
        });
        html += "</tbody></table>";
        container.innerHTML = html;
    }

    function corrColor(v) {
        if (v >= 0.6)  return "rgba(239, 68, 68, 0.75)";
        if (v >= 0.3)  return "rgba(245, 158, 11, 0.55)";
        if (v >= 0.1)  return "rgba(245, 158, 11, 0.25)";
        if (v > -0.1)  return "rgba(0, 0, 0, 0.05)";
        if (v > -0.3)  return "rgba(59, 130, 246, 0.25)";
        if (v > -0.6)  return "rgba(59, 130, 246, 0.55)";
        return "rgba(59, 130, 246, 0.75)";
    }

    function scatterByTarget(canvasId, recs, xField, yField, xLabel, yLabel) {
        const d0 = [], d1 = [];
        recs.forEach(r => {
            const pt = { x: parseFloat(r[xField]), y: parseFloat(r[yField]) };
            if (isNaN(pt.x) || isNaN(pt.y)) return;
            (r.hd === 1 ? d1 : d0).push(pt);
        });
        mkChart(canvasId, "scatter", {
            datasets: [
                { label: "No Disease (0)", data: d0, backgroundColor: P.green, pointRadius: 3.5 },
                { label: "Heart Disease (1)", data: d1, backgroundColor: P.red, pointRadius: 3.5 },
            ]
        }, {
            plugins: { legend: { position: "bottom" } },
            scales: {
                x: { title: { display: true, text: xLabel, color: textColor }, grid: { color: gridColor }, ticks: { color: textColor } },
                y: { title: { display: true, text: yLabel, color: textColor }, grid: { color: gridColor }, ticks: { color: textColor } }
            }
        });
    }

    /* ═══════════════════════════════════════════════════════════════
       TAB 4 — FEATURE vs TARGET
       ═══════════════════════════════════════════════════════════════ */
    function renderTargetAnalysis() {
        const recs = DATA.records;
        const configs = [
            { id: "ch-sex-tgt",   field: "sex",     map: null },
            { id: "ch-cp-tgt",    field: "cp",      map: null },
            { id: "ch-fbs-tgt",   field: "fbs",     map: { "0.0": "No", "1.0": "Yes", "0": "No", "1": "Yes" } },
            { id: "ch-exang-tgt", field: "exang",   map: { "0.0": "No", "1.0": "Yes", "0": "No", "1": "Yes" } },
            { id: "ch-thal-tgt",  field: "thal",    map: null },
            { id: "ch-slope-tgt", field: "slope",   map: null },
            { id: "ch-ecg-tgt",   field: "restecg", map: { "0.0": "Normal", "1.0": "ST-T Abnorm", "2.0": "LV Hypertrophy", "0": "Normal", "1": "ST-T Abnorm", "2": "LV Hypertrophy" } },
            { id: "ch-ca-tgt",    field: "ca",      map: null },
        ];
        configs.forEach(c => groupedBarByTarget(c.id, recs, c.field, c.map));
    }

    function groupedBarByTarget(canvasId, recs, field, labelMap) {
        const groups = {};
        recs.forEach(r => {
            const k = String(r[field]);
            if (!groups[k]) groups[k] = { 0: 0, 1: 0 };
            groups[k][r.hd]++;
        });
        const keys = Object.keys(groups).sort();
        const displayLabels = keys.map(k => (labelMap && labelMap[k]) ? labelMap[k] : k);
        mkChart(canvasId, "bar", {
            labels: displayLabels,
            datasets: [
                { label: "No Disease (0)", data: keys.map(k => groups[k][0]), backgroundColor: P.green, borderRadius: 3 },
                { label: "Heart Disease (1)", data: keys.map(k => groups[k][1]), backgroundColor: P.red, borderRadius: 3 },
            ]
        }, { plugins: { legend: { position: "top" } }, scales: barScales() });
    }

    /* ═══════════════════════════════════════════════════════════════
       TAB 5 — HYPERPARAMETER TUNING, KERNELS & MODELS
       ═══════════════════════════════════════════════════════════════ */
    function renderModels() {
        const td = DATA.tuning_data;
        const kr = DATA.kernel_results;
        const cr = DATA.classifier_results;

        // ── 1. Hyperparameter Tuning Section ──
        if (td) {
            setText("tb-cv-score", (td.best_cv_score * 100).toFixed(2) + "%");

            // Render Tuning Grid Heatmap Table
            const container = document.getElementById("tuning-heatmap-container");
            if (container) {
                const cVals = td.c_values;
                const gVals = td.gamma_values;
                const matrix = td.score_matrix;

                let html = '<table class="tuning-hm-table"><thead><tr><th>C \\ &gamma;</th>';
                gVals.forEach(g => { html += `<th>&gamma; = ${g}</th>`; });
                html += '</tr></thead><tbody>';

                cVals.forEach((c, i) => {
                    html += `<tr><th>C = ${c}</th>`;
                    matrix[i].forEach((score, j) => {
                        const isOptimal = (c === td.best_params.C && gVals[j] === td.best_params.gamma);
                        const pct = (score * 100).toFixed(1);
                        const bg = getTuningColor(score);
                        html += `<td class="${isOptimal ? 'optimal' : ''}" style="background:${bg};color:#fff" title="C=${c}, gamma=${gVals[j]}: ${pct}%">${pct}%</td>`;
                    });
                    html += `</tr>`;
                });
                html += '</tbody></table>';
                container.innerHTML = html;
            }

            // Parameter Sensitivity Curve Chart
            const cColors = [P.teal, P.blue, P.green, P.purple];
            const curveDatasets = td.c_values.map((c, i) => ({
                label: `C = ${c}`,
                data: td.score_matrix[i].map(s => +(s * 100).toFixed(1)),
                borderColor: cColors[i],
                backgroundColor: cColors[i],
                tension: 0.25,
                borderWidth: c === td.best_params.C ? 3.5 : 2,
                pointRadius: c === td.best_params.C ? 6 : 4,
            }));

            mkChart("ch-tuning-curve", "line", {
                labels: td.gamma_values.map(g => `γ = ${g}`),
                datasets: curveDatasets
            }, {
                plugins: { legend: { position: "top" } },
                scales: {
                    x: { title: { display: true, text: "Kernel Coefficient (gamma)", color: textColor }, grid: { color: gridColor }, ticks: { color: textColor } },
                    y: { title: { display: true, text: "5-Fold CV Accuracy (%)", color: textColor }, min: 50, max: 90, grid: { color: gridColor }, ticks: { color: textColor } }
                }
            });
        }

        // ── 2. SVM Kernel Comparison Diagrams ──
        if (kr) {
            const kKeys = ["linear", "poly", "rbf", "sigmoid"];
            const kLabels = kKeys.map(k => kr[k].kernel || k.toUpperCase());
            const metrics = ["accuracy", "precision", "recall", "f1", "auc"];
            const colors = [P.blue, P.purple, P.green, P.orange, P.teal];

            // Kernel Grouped Bar Chart
            mkChart("ch-kernels", "bar", {
                labels: kLabels,
                datasets: metrics.map((m, i) => ({
                    label: m.charAt(0).toUpperCase() + m.slice(1) + " (%)",
                    data: kKeys.map(k => +(kr[k][m] * 100).toFixed(1)),
                    backgroundColor: colors[i],
                    borderRadius: 4,
                }))
            }, {
                plugins: { legend: { position: "top" } },
                scales: {
                    y: { min: 65, max: 100, grid: { color: gridColor }, ticks: { color: textColor } },
                    x: { grid: { color: gridColor }, ticks: { color: textColor } }
                }
            });

            // Kernel Capabilities Radar Chart
            const radarColors = [P.blue, P.purple, P.green, P.orange];
            const radarDatasets = kKeys.map((k, i) => ({
                label: kr[k].kernel,
                data: metrics.map(m => +(kr[k][m] * 100).toFixed(1)),
                borderColor: radarColors[i],
                backgroundColor: radarColors[i] + "22",
                borderWidth: 2,
                pointRadius: 3.5,
            }));

            mkChart("ch-kernel-radar", "radar", {
                labels: ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"],
                datasets: radarDatasets
            }, {
                plugins: { legend: { position: "top" } },
                scales: {
                    r: {
                        min: 65, max: 100,
                        angleLines: { color: gridColor },
                        grid: { color: gridColor },
                        pointLabels: { color: "#475569", font: { size: 10, weight: 600 } },
                        ticks: { display: false }
                    }
                }
            });

            // Kernel Table
            const ktbody = document.querySelector("#kernel-table tbody");
            if (ktbody) {
                ktbody.innerHTML = "";
                kKeys.forEach(k => {
                    const m = kr[k];
                    const tr = document.createElement("tr");
                    tr.innerHTML = `<td><strong>${m.kernel}</strong></td>
                        <td>${(m.accuracy * 100).toFixed(1)}%</td>
                        <td>${(m.precision * 100).toFixed(1)}%</td>
                        <td><strong class="text-green">${(m.recall * 100).toFixed(1)}%</strong></td>
                        <td>${(m.f1 * 100).toFixed(1)}%</td>
                        <td>${(m.auc * 100).toFixed(1)}%</td>
                        <td>${m.support_vectors || '—'}</td>`;
                    ktbody.appendChild(tr);
                });
            }

            // Interactive Kernel Confusion Matrix Switcher
            initKernelConfusionMatrix(kr);
        }

        // SVM Kernel ROC Curves Chart
        const rocData = DATA.kernel_roc_curves;
        if (rocData) {
            const rocColors = { linear: P.blue, poly: P.purple, rbf: P.green, sigmoid: P.orange };
            const rocDatasets = Object.keys(rocData).map(k => ({
                label: `${rocData[k].name} (AUC = ${rocData[k].auc})`,
                data: rocData[k].fpr.map((x, idx) => ({ x, y: rocData[k].tpr[idx] })),
                borderColor: rocColors[k] || P.blue,
                backgroundColor: "transparent",
                borderWidth: k === "rbf" ? 3 : 2,
                pointRadius: 2,
                showLine: true,
            }));

            // Baseline random guess diagonal
            rocDatasets.push({
                label: "Random Guess (AUC = 0.50)",
                data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                borderColor: "#64748b",
                borderDash: [5, 5],
                borderWidth: 1.5,
                pointRadius: 0,
                showLine: true,
            });

            mkChart("ch-kernel-roc", "scatter", {
                datasets: rocDatasets
            }, {
                plugins: { legend: { position: "bottom" } },
                scales: {
                    x: { title: { display: true, text: "False Positive Rate (1 - Specificity)", color: textColor }, min: 0, max: 1, grid: { color: gridColor }, ticks: { color: textColor } },
                    y: { title: { display: true, text: "True Positive Rate (Sensitivity / Recall)", color: textColor }, min: 0, max: 1, grid: { color: gridColor }, ticks: { color: textColor } }
                }
            });
        }

        // ── 3. 5-Classifier Benchmark ──
        if (cr) {
            const names = Object.keys(cr);
            const metrics = ["accuracy", "precision", "recall", "f1", "auc"];
            const colors = [P.blue, P.purple, P.green, P.orange, P.teal];

            mkChart("ch-5clf", "bar", {
                labels: names,
                datasets: metrics.map((m, i) => ({
                    label: m.charAt(0).toUpperCase() + m.slice(1) + " (%)",
                    data: names.map(n => +(cr[n][m] * 100).toFixed(1)),
                    backgroundColor: colors[i],
                    borderRadius: 4,
                }))
            }, {
                plugins: { legend: { position: "top" } },
                scales: {
                    y: { min: 50, max: 100, grid: { color: gridColor }, ticks: { color: textColor } },
                    x: { grid: { color: gridColor }, ticks: { color: textColor } }
                }
            });

            const tbody = document.querySelector("#clf-table tbody");
            if (tbody) {
                tbody.innerHTML = "";
                let bestName = "", bestAuc = 0;
                names.forEach(n => {
                    const m = cr[n];
                    if (m.auc > bestAuc) { bestAuc = m.auc; bestName = n; }
                    const tr = document.createElement("tr");
                    tr.innerHTML = `<td><strong>${n}</strong></td>
                        <td>${(m.accuracy * 100).toFixed(1)}%</td>
                        <td>${(m.precision * 100).toFixed(1)}%</td>
                        <td><strong class="text-green">${(m.recall * 100).toFixed(1)}%</strong></td>
                        <td>${(m.f1 * 100).toFixed(1)}%</td>
                        <td>${(m.auc * 100).toFixed(1)}%</td>`;
                    tbody.appendChild(tr);
                });

                // Set best classifier cards
                const targetBest = cr["SVM (RBF Tuned)"] || cr[bestName];
                if (targetBest) {
                    setText("mc-acc", (targetBest.accuracy * 100).toFixed(1) + "%");
                    setText("mc-pre", (targetBest.precision * 100).toFixed(1) + "%");
                    setText("mc-rec", (targetBest.recall * 100).toFixed(1) + "%");
                    setText("mc-f1",  (targetBest.f1 * 100).toFixed(1) + "%");
                    setText("mc-auc", (targetBest.auc * 100).toFixed(1) + "%");
                }
            }
        }
    }

    function getTuningColor(score) {
        if (score >= 0.80) return "rgba(16, 185, 129, 0.65)";
        if (score >= 0.78) return "rgba(16, 185, 129, 0.45)";
        if (score >= 0.75) return "rgba(59, 130, 246, 0.45)";
        if (score >= 0.65) return "rgba(245, 158, 11, 0.35)";
        return "rgba(0, 0, 0, 0.05)";
    }

    function initKernelConfusionMatrix(kr) {
        const pills = document.querySelectorAll(".k-pill");
        function updateCM(kKey) {
            const data = kr[kKey];
            if (!data || !data.confusion_matrix) return;
            const cm = data.confusion_matrix;
            setText("k-cm-tn", cm[0][0]);
            setText("k-cm-fp", cm[0][1]);
            setText("k-cm-fn", cm[1][0]);
            setText("k-cm-tp", cm[1][1]);

            const alertEl = document.getElementById("k-cm-alert");
            if (alertEl) {
                alertEl.innerHTML = `<strong>${data.kernel}:</strong> Classified ${cm[0][0] + cm[1][1]} / 60 patients correctly. ` +
                    `False Negatives: <strong class="text-red">${cm[1][0]}</strong> (Recall: ${(data.recall * 100).toFixed(1)}%). Lower FN is preferred for heart disease screening.`;
            }
        }

        pills.forEach(pill => {
            pill.addEventListener("click", () => {
                pills.forEach(p => p.classList.remove("active"));
                pill.classList.add("active");
                updateCM(pill.dataset.kernel);
            });
        });

        // Initialize with default (rbf)
        updateCM("rbf");
    }

    /* ═══════════════════════════════════════════════════════════════
       TAB 6 — PATIENT PREDICTION
       ═══════════════════════════════════════════════════════════════ */
    function initForm() {
        const form = document.getElementById("patient-form");
        if (!form || form.dataset.bound) return;
        form.dataset.bound = "true";

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            const btn = document.getElementById("btn-predict");
            if (btn) btn.innerHTML = '<i data-lucide="loader-2"></i> Analyzing...';

            const payload = {
                kernel:  v("f-kernel") || "rbf",
                age:     v("f-age"),
                sex:     v("f-sex"),
                cp:      v("f-cp"),
                restbp:  v("f-restbp"),
                chol:    v("f-chol"),
                fbs:     v("f-fbs"),
                restecg: v("f-restecg"),
                thalach: v("f-thalach"),
                exang:   v("f-exang"),
                oldpeak: v("f-oldpeak"),
                slope:   v("f-slope"),
                ca:      v("f-ca"),
                thal:    v("f-thal"),
            };

            try {
                const resp = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload)
                });
                const res = await resp.json();
                if (btn) btn.innerHTML = '<i data-lucide="play-circle"></i> Predict &amp; Analyze with Tuned Model';
                lucide.createIcons();

                if (res.error) {
                    alert("Prediction Error: " + res.error);
                    return;
                }
                renderPrediction(res);
            } catch (err) {
                if (btn) btn.innerHTML = '<i data-lucide="play-circle"></i> Predict &amp; Analyze with Tuned Model';
                lucide.createIcons();
                alert("Prediction request failed: " + err.message);
            }
        });
    }

    function renderPrediction(res) {
        const badge = document.getElementById("pred-badge");
        const cls   = document.getElementById("pred-class");
        const modelBadge = document.getElementById("res-model-badge");

        if (modelBadge && res.model_used) {
            modelBadge.textContent = res.model_used;
        }

        if (res.prediction === 1) {
            badge.textContent = "HEART DISEASE POSITIVE (Class 1)";
            badge.className = "pred-badge yes";
            cls.textContent = "HEART DISEASE DETECTED";
            cls.style.color = "var(--red)";
        } else {
            badge.textContent = "NO HEART DISEASE (Class 0)";
            badge.className = "pred-badge no";
            cls.textContent = "NO HEART DISEASE DETECTED";
            cls.style.color = "var(--green)";
        }

        setText("pred-prob", (res.probability * 100).toFixed(1) + "%");
        setText("pred-score", (res.decision_score > 0 ? "+" : "") + res.decision_score.toFixed(3));

        // Patient vs Dataset Comparison Bars
        const compList = document.getElementById("comparison-list");
        if (compList && res.patient_comparison) {
            compList.innerHTML = "";
            const labels = {
                age: "Age", restbp: "Resting BP", chol: "Serum Cholesterol",
                thalach: "Max Heart Rate", oldpeak: "ST Depression"
            };
            const units = {
                age: "yrs", restbp: "mm Hg", chol: "mg/dl",
                thalach: "bpm", oldpeak: "mm"
            };

            for (const [col, c] of Object.entries(res.patient_comparison)) {
                const range = (c.max - c.min) || 1;
                const pct = Math.max(0, Math.min(100, ((c.value - c.min) / range) * 100));
                const item = document.createElement("div");
                item.className = "comp-item";
                item.innerHTML = `
                    <div class="comp-head">
                        <span class="comp-lbl">${labels[col] || col}</span>
                        <span class="comp-vals">${c.value} ${units[col] || ''} <span style="color:var(--txt2);font-weight:400">(Median: ${c.median})</span></span>
                    </div>
                    <div class="comp-bar"><div class="comp-fill" style="width:${pct}%"></div></div>
                `;
                compList.appendChild(item);
            }
        }

        // Similar Patients Table
        const simBody = document.getElementById("sim-body");
        if (simBody && res.similar_patients) {
            simBody.innerHTML = "";
            res.similar_patients.forEach((sp, i) => {
                const r = sp.record;
                const tr = document.createElement("tr");
                const hasDisease = r.hd === 1;
                tr.innerHTML = `
                    <td>#${i + 1}</td>
                    <td><strong class="text-green">${sp.similarity_pct}%</strong></td>
                    <td>${r.age}</td>
                    <td>${r.sex}</td>
                    <td>${r.restbp}</td>
                    <td>${r.chol}</td>
                    <td>${r.thalach}</td>
                    <td>${r.cp}</td>
                    <td><span class="badge ${hasDisease ? 'red' : 'green'}">${hasDisease ? 'Disease' : 'Healthy'}</span></td>
                `;
                simBody.appendChild(tr);
            });
        }
    }

    function initSampleButton() {
        const btn = document.getElementById("load-sample");
        if (!btn) return;
        btn.addEventListener("click", () => {
            document.getElementById("f-age").value = "58";
            document.getElementById("f-sex").value = "Male";
            document.getElementById("f-cp").value = "asymptomatic";
            document.getElementById("f-restbp").value = "140";
            document.getElementById("f-chol").value = "260";
            document.getElementById("f-fbs").value = "0.0";
            document.getElementById("f-restecg").value = "2.0";
            document.getElementById("f-thalach").value = "125";
            document.getElementById("f-exang").value = "1.0";
            document.getElementById("f-oldpeak").value = "2.2";
            document.getElementById("f-slope").value = "flat";
            document.getElementById("f-ca").value = "2.0";
            document.getElementById("f-thal").value = "reversable defect";
            document.getElementById("f-kernel").value = "rbf";
        });
    }

    // ── Helper Utilities ──
    function v(id) {
        const el = document.getElementById(id);
        return el ? el.value : null;
    }

    function setText(id, txt) {
        const el = document.getElementById(id);
        if (el) el.textContent = txt;
    }

    function barScales() {
        return {
            x: { grid: { color: gridColor }, ticks: { color: textColor } },
            y: { grid: { color: gridColor }, ticks: { color: textColor } }
        };
    }
});
