/**
 * AMR-Predict Application Logic
 * Interactive Doctor Dashboard with Real-Time Multi-Model Predictions,
 * SHAP Waterfall & Contribution Visualizers, Benchmarks, and Dataset EDA.
 */

// Global State
const state = {
    config: null,
    currentPatient: null,
    currentPredictions: null,
    selectedAntibiotic: "Ciprofloxacin",
    shapMode: "local", // 'local' or 'global'
    modelsData: null,
    edaData: null,
    chartInstances: {},
};

// Initialize Application
document.addEventListener("DOMContentLoaded", async () => {
    setupTabNavigation();
    setupFormListeners();
    setupExplainListeners();

    await loadSystemConfig();
    await loadModelBenchmarks();
    await loadDatasetEda();

    // Trigger initial prediction with default form values
    const form = document.getElementById("patientForm");
    if (form) {
        form.dispatchEvent(new Event("submit"));
    }
});

/**
 * Tab Navigation Handler
 */
function setupTabNavigation() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetId = tab.getAttribute("data-tab");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }

            // Trigger chart resize when switching tabs
            if (targetId === "tab-explain") {
                if (state.shapMode === "local") {
                    refreshLocalShapChart();
                } else {
                    renderGlobalShapChart();
                }
            } else if (targetId === "tab-models") {
                resizeCharts(["rocChartCanvas", "calibChartCanvas"]);
            } else if (targetId === "tab-eda") {
                resizeCharts(["edaResistanceChart", "edaOrganismChart", "edaInfectionChart", "edaComorbiditiesChart"]);
            }
        });
    });
}

function resizeCharts(canvasIds) {
    canvasIds.forEach(id => {
        if (state.chartInstances[id]) {
            state.chartInstances[id].resize();
        }
    });
}

/**
 * Form and Preset Listeners
 */
function setupFormListeners() {
    const form = document.getElementById("patientForm");
    const presetSelect = document.getElementById("presetSelect");

    if (presetSelect) {
        presetSelect.addEventListener("change", (e) => {
            const presetId = e.target.value;
            if (!presetId || !state.config || !state.config.presets) return;

            const preset = state.config.presets.find(p => p.id === presetId);
            if (preset && preset.data) {
                populateFormWithData(preset.data);
                form.dispatchEvent(new Event("submit"));
            }
        });
    }

    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            await runPatientAnalysis();
        });
    }
}

/**
 * Explainability Tab Listeners
 */
function setupExplainListeners() {
    const abxSelect = document.getElementById("explainAbxSelect");
    const btnLocal = document.getElementById("btnLocalShap");
    const btnGlobal = document.getElementById("btnGlobalShap");

    if (abxSelect) {
        abxSelect.addEventListener("change", (e) => {
            state.selectedAntibiotic = e.target.value;
            updateShapExplanationView();
        });
    }

    if (btnLocal && btnGlobal) {
        btnLocal.addEventListener("click", () => {
            btnLocal.classList.add("active");
            btnGlobal.classList.remove("active");
            state.shapMode = "local";
            document.getElementById("shapChartTitle").innerText = `SHAP Local Waterfall Breakdown (${state.selectedAntibiotic})`;
            updateShapExplanationView();
        });

        btnGlobal.addEventListener("click", () => {
            btnGlobal.classList.add("active");
            btnLocal.classList.remove("active");
            state.shapMode = "global";
            document.getElementById("shapChartTitle").innerText = `Dataset Global Feature Importance (${state.selectedAntibiotic})`;
            renderGlobalShapChart();
        });
    }
}

/**
 * Load System Configuration and Presets
 */
async function loadSystemConfig() {
    try {
        const res = await fetch("/api/config");
        if (!res.ok) throw new Error("Failed to load config");
        state.config = await res.json();

        // Populate Presets Dropdown
        const presetSelect = document.getElementById("presetSelect");
        if (presetSelect && state.config.presets) {
            presetSelect.innerHTML = `<option value="">-- Choose Clinical Preset --</option>`;
            state.config.presets.forEach(p => {
                const opt = document.createElement("option");
                opt.value = p.id;
                opt.textContent = p.name;
                presetSelect.appendChild(opt);
            });
        }
    } catch (err) {
        console.error("Config load error:", err);
    }
}

/**
 * Populate form with data
 */
function populateFormWithData(data) {
    document.getElementById("inputAge").value = data.age;
    document.getElementById("inputSex").value = data.sex;
    document.getElementById("inputCreatinine").value = data.creatinine_mg_dl;
    document.getElementById("inputWbc").value = data.wbc_count_k_ul;
    document.getElementById("inputInfection").value = data.infection_type;
    document.getElementById("inputOrganism").value = data.organism;
    document.getElementById("inputPriorUti").value = data.num_prior_uti_1yr;

    document.getElementById("inputHospital").checked = Boolean(data.recent_hospitalization_90d);
    document.getElementById("inputAbx").checked = Boolean(data.recent_antibiotic_use_90d);
    document.getElementById("inputResistantCulture").checked = Boolean(data.prior_resistant_culture_1yr);
    document.getElementById("inputDiabetes").checked = Boolean(data.diabetes);
    document.getElementById("inputCatheter").checked = Boolean(data.catheter_use);
    document.getElementById("inputImmuno").checked = Boolean(data.immunocompromised);
    document.getElementById("inputNursing").checked = Boolean(data.nursing_home_resident);
    document.getElementById("inputTravel").checked = Boolean(data.travel_last_6mo);
    document.getElementById("inputHcw").checked = Boolean(data.healthcare_worker);
}

/**
 * Collect Form Data
 */
function getFormData() {
    return {
        age: parseFloat(document.getElementById("inputAge").value) || 50,
        sex: document.getElementById("inputSex").value,
        infection_type: document.getElementById("inputInfection").value,
        organism: document.getElementById("inputOrganism").value,
        diabetes: document.getElementById("inputDiabetes").checked ? 1 : 0,
        recent_hospitalization_90d: document.getElementById("inputHospital").checked ? 1 : 0,
        recent_antibiotic_use_90d: document.getElementById("inputAbx").checked ? 1 : 0,
        num_prior_uti_1yr: parseInt(document.getElementById("inputPriorUti").value) || 0,
        catheter_use: document.getElementById("inputCatheter").checked ? 1 : 0,
        immunocompromised: document.getElementById("inputImmuno").checked ? 1 : 0,
        nursing_home_resident: document.getElementById("inputNursing").checked ? 1 : 0,
        prior_resistant_culture_1yr: document.getElementById("inputResistantCulture").checked ? 1 : 0,
        creatinine_mg_dl: parseFloat(document.getElementById("inputCreatinine").value) || 1.0,
        wbc_count_k_ul: parseFloat(document.getElementById("inputWbc").value) || 9.5,
        travel_last_6mo: document.getElementById("inputTravel").checked ? 1 : 0,
        healthcare_worker: document.getElementById("inputHcw").checked ? 1 : 0,
    };
}

/**
 * Main Inference Action: Run Multi-Antibiotic Prediction
 */
async function runPatientAnalysis() {
    const btn = document.getElementById("btnAnalyze");
    const latencyText = document.getElementById("latencyText");
    const grid = document.getElementById("resultsGrid");

    try {
        if (btn) btn.disabled = true;
        if (latencyText) latencyText.innerText = "Evaluating 8 models...";

        const patientData = getFormData();
        state.currentPatient = patientData;

        const res = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(patientData),
        });

        if (!res.ok) {
            const errJson = await res.json();
            throw new Error(errJson.detail || "Prediction request failed");
        }

        const data = await res.json();
        state.currentPredictions = data.predictions;

        // Update Latency Badge
        if (latencyText) {
            latencyText.innerText = data.execution_time_display || `Inference: ${data.execution_time_ms} ms`;
        }

        // Render Cards Grid
        renderAntibioticCards(data.predictions);

        // Update SHAP explanation for currently selected antibiotic
        await updateShapExplanationView();

    } catch (err) {
        console.error("Inference Error:", err);
        if (latencyText) latencyText.innerText = "Error in prediction";
        if (grid) {
            grid.innerHTML = `
                <div class="empty-state" style="color: #f43f5e;">
                    <div class="empty-icon">⚠️</div>
                    <h3>Prediction Error</h3>
                    <p>${err.message}</p>
                </div>
            `;
        }
    } finally {
        if (btn) btn.disabled = false;
    }
}

/**
 * Render 8 Antibiotic Prediction Cards
 */
function renderAntibioticCards(predictions) {
    const grid = document.getElementById("resultsGrid");
    if (!grid) return;

    if (!predictions || predictions.length === 0) {
        grid.innerHTML = `<div class="empty-state"><p>No predictions returned.</p></div>`;
        return;
    }

    grid.innerHTML = "";

    predictions.forEach(p => {
        const riskClass = p.risk_category.toLowerCase();
        const fillClass = riskClass === "high" ? "high" : riskClass === "moderate" ? "moderate" : "low";
        const badgeClass = `badge-${riskClass}`;
        const pct = (p.estimated_resistance_probability * 100).toFixed(1);

        const card = document.createElement("div");
        card.className = `abx-card risk-${riskClass}`;
        card.innerHTML = `
            <div class="abx-card-header">
                <span class="abx-name">${p.antibiotic}</span>
                <div class="abx-meta-right">
                    <span class="abx-model-badge">${p.model_type}</span>
                    <span class="abx-risk-badge ${badgeClass}">${p.risk_category} Risk</span>
                </div>
            </div>
            <div class="prob-bar-wrapper">
                <div class="prob-track">
                    <div class="prob-fill ${fillClass}" style="width: ${pct}%"></div>
                </div>
                <span class="prob-percentage">${pct}%</span>
            </div>
            <div class="abx-card-footer">
                <span>${p.interpretation_label}</span>
                <button class="btn-explain" data-abx="${p.antibiotic}">
                    <span>Explain with SHAP</span> ➔
                </button>
            </div>
        `;

        // Attach Click Listener on "Explain with SHAP"
        const explainBtn = card.querySelector(".btn-explain");
        explainBtn.addEventListener("click", () => {
            state.selectedAntibiotic = p.antibiotic;
            const abxSelect = document.getElementById("explainAbxSelect");
            if (abxSelect) abxSelect.value = p.antibiotic;

            // Switch to explainability tab
            const tabExplain = document.querySelector('[data-tab="tab-explain"]');
            if (tabExplain) tabExplain.click();
        });

        grid.appendChild(card);
    });
}

/**
 * Update SHAP Explanation View (Local Waterfall & Lists)
 */
async function updateShapExplanationView() {
    if (!state.currentPatient) return;

    const abx = state.selectedAntibiotic;
    const explainAbxName = document.getElementById("explainAbxName");
    const explainProbText = document.getElementById("explainProbText");
    const explainRiskBadge = document.getElementById("explainRiskBadge");
    const posList = document.getElementById("positiveFactorsList");
    const negList = document.getElementById("negativeFactorsList");

    if (explainAbxName) explainAbxName.innerText = abx;

    try {
        const res = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                patient_data: state.currentPatient,
                antibiotic: abx,
            }),
        });

        if (!res.ok) throw new Error("Failed to fetch SHAP explanation");

        const data = await res.json();

        // Update Summary Pill
        const probPct = (data.estimated_resistance_probability * 100).toFixed(1);
        if (explainProbText) explainProbText.innerText = `${probPct}% Prob`;

        let riskTier = "Low";
        let riskBadgeClass = "badge-low";
        if (data.estimated_resistance_probability > 0.65) {
            riskTier = "High";
            riskBadgeClass = "badge-high";
        } else if (data.estimated_resistance_probability >= 0.35) {
            riskTier = "Moderate";
            riskBadgeClass = "badge-moderate";
        }

        if (explainRiskBadge) {
            explainRiskBadge.className = `pill-badge ${riskBadgeClass}`;
            explainRiskBadge.innerText = `${riskTier} Risk`;
        }

        // Render Positive Factors
        if (posList) {
            if (data.top_positive_factors && data.top_positive_factors.length > 0) {
                posList.innerHTML = data.top_positive_factors.map(f => `
                    <div class="factor-item">
                        <span class="factor-name">${f.display_name}</span>
                        <span class="factor-val pos">+${f.shap_value.toFixed(4)}</span>
                    </div>
                `).join("");
            } else {
                posList.innerHTML = `<div class="empty-text">No significant risk-increasing factors.</div>`;
            }
        }

        // Render Negative Factors
        if (negList) {
            if (data.top_negative_factors && data.top_negative_factors.length > 0) {
                negList.innerHTML = data.top_negative_factors.map(f => `
                    <div class="factor-item">
                        <span class="factor-name">${f.display_name}</span>
                        <span class="factor-val neg">${f.shap_value.toFixed(4)}</span>
                    </div>
                `).join("");
            } else {
                negList.innerHTML = `<div class="empty-text">No significant risk-decreasing factors.</div>`;
            }
        }

        // Render Plotly Waterfall Chart
        if (state.shapMode === "local") {
            renderPlotlyWaterfall(data);
        }

    } catch (err) {
        console.error("SHAP View Error:", err);
    }
}

/**
 * Render Interactive Plotly Waterfall Chart
 */
function renderPlotlyWaterfall(data) {
    const container = document.getElementById("plotlyShapDiv");
    if (!container || !window.Plotly) return;

    const baseVal = data.base_value;
    const finalProb = data.estimated_resistance_probability;
    const features = data.waterfall_features || [];

    // Construct waterfall steps
    const labels = ["Base Expected Value"];
    const deltas = [baseVal];
    const measures = ["absolute"];
    const textVals = [`${(baseVal * 100).toFixed(1)}%`];

    features.forEach(f => {
        labels.push(f.display_name);
        deltas.push(f.shap_value);
        measures.push("relative");
        textVals.push(`${f.shap_value > 0 ? "+" : ""}${(f.shap_value * 100).toFixed(1)}%`);
    });

    labels.push("Final Probability");
    deltas.push(finalProb);
    measures.push("total");
    textVals.push(`${(finalProb * 100).toFixed(1)}%`);

    const plotData = [{
        type: "waterfall",
        orientation: "v",
        measure: measures,
        x: labels,
        y: deltas,
        text: textVals,
        textposition: "outside",
        decreasing: { marker: { color: "#10b981" } },
        increasing: { marker: { color: "#f43f5e" } },
        totals: { marker: { color: "#38bdf8" } },
        connector: { line: { color: "#475569", width: 1, dash: "dot" } },
    }];

    const layout = {
        title: {
            text: `Patient SHAP Contribution Waterfall: ${data.antibiotic}`,
            font: { color: "#f8fafc", size: 14, family: "Inter" }
        },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#94a3b8", family: "Inter", size: 11 },
        margin: { l: 40, r: 20, t: 40, b: 120 },
        xaxis: {
            tickangle: -35,
            gridcolor: "rgba(51, 65, 85, 0.3)",
        },
        yaxis: {
            title: "Probability Contribution",
            gridcolor: "rgba(51, 65, 85, 0.3)",
            range: [0, Math.min(1.0, finalProb + 0.2)],
        },
        autosize: true,
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot("plotlyShapDiv", plotData, layout, config);
}

function refreshLocalShapChart() {
    updateShapExplanationView();
}

/**
 * Render Dataset-Wide Global SHAP Bar Chart
 */
async function renderGlobalShapChart() {
    const abx = state.selectedAntibiotic;
    try {
        const res = await fetch(`/api/global-shap?antibiotic=${encodeURIComponent(abx)}`);
        if (!res.ok) throw new Error("Failed to load global SHAP summary");

        const data = await res.json();
        const topFeatures = data.global_shap.top_features || [];

        const labels = topFeatures.map(f => f.feature.replace("_", " ").replace("organism ", "Pathogen: ").replace("infection type ", "Infection: "));
        const values = topFeatures.map(f => f.mean_shap);

        // Reverse for horizontal bar chart (top at top)
        labels.reverse();
        values.reverse();

        const plotData = [{
            type: "bar",
            orientation: "h",
            x: values,
            y: labels,
            marker: {
                color: "#38bdf8",
            },
            text: values.map(v => v.toFixed(4)),
            textposition: "auto",
        }];

        const layout = {
            title: {
                text: `Global Feature Importance (Mean |SHAP|): ${abx}`,
                font: { color: "#f8fafc", size: 14, family: "Inter" }
            },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { color: "#94a3b8", family: "Inter", size: 11 },
            margin: { l: 160, r: 30, t: 40, b: 40 },
            xaxis: {
                title: "Mean |SHAP Value| (Impact on Model Output)",
                gridcolor: "rgba(51, 65, 85, 0.3)",
            },
            yaxis: {
                gridcolor: "rgba(51, 65, 85, 0.3)",
            },
            autosize: true,
        };

        const config = { responsive: true, displayModeBar: false };
        Plotly.newPlot("plotlyShapDiv", plotData, layout, config);

    } catch (err) {
        console.error("Global SHAP Error:", err);
    }
}

/**
 * Load Model Benchmarks and Render Comparison Table + ROC / Calibration Charts
 */
async function loadModelBenchmarks() {
    try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error("Failed to load model benchmarks");

        const data = await res.json();
        state.modelsData = data;

        const tbody = document.getElementById("modelsTableBody");
        if (tbody && data.registry) {
            tbody.innerHTML = Object.entries(data.registry).map(([abx, meta]) => {
                const m = meta.metrics;
                const prev = (meta.prevalence * 100).toFixed(1);
                return `
                    <tr>
                        <td><strong>${abx}</strong></td>
                        <td>${prev}%</td>
                        <td><span class="abx-model-badge">${meta.model_type}</span></td>
                        <td><strong style="color:#38bdf8">${m.roc_auc.toFixed(3)}</strong></td>
                        <td>${m.f1.toFixed(3)}</td>
                        <td>${m.recall.toFixed(3)}</td>
                        <td>${m.precision.toFixed(3)}</td>
                        <td>${m.accuracy.toFixed(3)}</td>
                        <td>${m.brier_score.toFixed(3)}</td>
                    </tr>
                `;
            }).join("");
        }

        // Render ROC Curve Chart
        renderRocCurves(data.detailed_metrics);
        // Render Calibration Diagram
        renderCalibrationCurves(data.detailed_metrics);

    } catch (err) {
        console.error("Benchmarks Load Error:", err);
    }
}

/**
 * Render Multi-Model ROC Curves
 */
function renderRocCurves(detailedMetrics) {
    const ctx = document.getElementById("rocChartCanvas");
    if (!ctx || !window.Chart || !detailedMetrics) return;

    const colors = ["#38bdf8", "#f43f5e", "#10b981", "#fbbf24", "#a855f7", "#ec4899", "#6366f1", "#14b8a6"];
    const datasets = [];

    let colorIdx = 0;
    Object.entries(detailedMetrics).forEach(([abx, item]) => {
        const selectedModel = item.selected_model;
        const candidate = item.candidates[selectedModel];
        if (candidate && candidate.roc_curve) {
            const fpr = candidate.roc_curve.fpr;
            const tpr = candidate.roc_curve.tpr;
            const points = fpr.map((x, i) => ({ x, y: tpr[i] }));

            datasets.push({
                label: `${abx} (${selectedModel}, AUC: ${candidate.roc_auc.toFixed(3)})`,
                data: points,
                borderColor: colors[colorIdx % colors.length],
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.1,
            });
            colorIdx++;
        }
    });

    // Reference chance diagonal
    datasets.push({
        label: "Chance Baseline (AUC: 0.500)",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(148, 163, 184, 0.4)",
        borderWidth: 1.5,
        borderDash: [4, 4],
        pointRadius: 0,
        fill: false,
    });

    if (state.chartInstances["rocChartCanvas"]) {
        state.chartInstances["rocChartCanvas"].destroy();
    }

    state.chartInstances["rocChartCanvas"] = new Chart(ctx, {
        type: "line",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
                    grid: { color: "rgba(51, 65, 85, 0.3)" },
                    ticks: { color: "#94a3b8" }
                },
                y: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "True Positive Rate (Recall)", color: "#94a3b8" },
                    grid: { color: "rgba(51, 65, 85, 0.3)" },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "#cbd5e1", font: { size: 10 } },
                    position: "bottom"
                }
            }
        }
    });
}

/**
 * Render Probability Calibration Diagram (Reliability Curve)
 */
function renderCalibrationCurves(detailedMetrics) {
    const ctx = document.getElementById("calibChartCanvas");
    if (!ctx || !window.Chart || !detailedMetrics) return;

    const colors = ["#38bdf8", "#f43f5e", "#10b981", "#fbbf24", "#a855f7", "#ec4899", "#6366f1", "#14b8a6"];
    const datasets = [];

    let colorIdx = 0;
    Object.entries(detailedMetrics).forEach(([abx, item]) => {
        const selectedModel = item.selected_model;
        const candidate = item.candidates[selectedModel];
        if (candidate && candidate.calibration && candidate.calibration.prob_pred) {
            const pred = candidate.calibration.prob_pred;
            const actual = candidate.calibration.prob_true;
            const points = pred.map((x, i) => ({ x, y: actual[i] }));

            datasets.push({
                label: `${abx} (Brier: ${candidate.brier_score.toFixed(3)})`,
                data: points,
                borderColor: colors[colorIdx % colors.length],
                borderWidth: 2,
                pointRadius: 3,
                fill: false,
            });
            colorIdx++;
        }
    });

    // Perfectly calibrated diagonal
    datasets.push({
        label: "Perfect Calibration",
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: "rgba(148, 163, 184, 0.5)",
        borderWidth: 1.5,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
    });

    if (state.chartInstances["calibChartCanvas"]) {
        state.chartInstances["calibChartCanvas"].destroy();
    }

    state.chartInstances["calibChartCanvas"] = new Chart(ctx, {
        type: "line",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "Mean Predicted Resistance Probability", color: "#94a3b8" },
                    grid: { color: "rgba(51, 65, 85, 0.3)" },
                    ticks: { color: "#94a3b8" }
                },
                y: {
                    type: "linear",
                    min: 0,
                    max: 1,
                    title: { display: true, text: "Observed Fraction of Resistant Cases", color: "#94a3b8" },
                    grid: { color: "rgba(51, 65, 85, 0.3)" },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "#cbd5e1", font: { size: 10 } },
                    position: "bottom"
                }
            }
        }
    });
}

/**
 * Load Dataset EDA and Render Visualizations
 */
async function loadDatasetEda() {
    try {
        const res = await fetch("/api/eda");
        if (!res.ok) throw new Error("Failed to load EDA data");

        const data = await res.json();
        state.edaData = data;

        // Populate KPIs
        if (data.dataset_metadata) {
            document.getElementById("kpiPatients").innerText = data.dataset_metadata.total_records.toLocaleString();
        }
        if (data.age_distribution) {
            document.getElementById("kpiAgeMean").innerText = `${data.age_distribution.mean} yrs`;
        }

        // Resistance Prevalence Chart
        renderEdaResistanceChart(data.resistance_prevalence);
        // Organism Distribution Chart
        renderEdaOrganismChart(data.organism_distribution);
        // Infection Syndrome Chart
        renderEdaInfectionChart(data.infection_distribution);
        // Comorbidities Chart
        renderEdaComorbiditiesChart(data.clinical_risk_factors_prevalence);

    } catch (err) {
        console.error("EDA Load Error:", err);
    }
}

function renderEdaResistanceChart(resPrev) {
    const ctx = document.getElementById("edaResistanceChart");
    if (!ctx || !window.Chart || !resPrev) return;

    const labels = Object.keys(resPrev);
    const values = Object.values(resPrev).map(v => (v.resistance_rate * 100).toFixed(1));

    if (state.chartInstances["edaResistanceChart"]) {
        state.chartInstances["edaResistanceChart"].destroy();
    }

    state.chartInstances["edaResistanceChart"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Resistance Prevalence (%)",
                data: values,
                backgroundColor: values.map(v => v > 50 ? "rgba(244, 63, 94, 0.7)" : "rgba(6, 182, 212, 0.7)"),
                borderColor: values.map(v => v > 50 ? "#f43f5e" : "#06b6d4"),
                borderWidth: 1,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } },
                y: { min: 0, max: 100, ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderEdaOrganismChart(orgDist) {
    const ctx = document.getElementById("edaOrganismChart");
    if (!ctx || !window.Chart || !orgDist) return;

    const labels = Object.keys(orgDist).map(k => k.replace("_", " "));
    const values = Object.values(orgDist);

    if (state.chartInstances["edaOrganismChart"]) {
        state.chartInstances["edaOrganismChart"].destroy();
    }

    state.chartInstances["edaOrganismChart"] = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: ["#0284c7", "#06b6d4", "#10b981", "#f59e0b", "#a855f7", "#ec4899"],
                borderColor: "#111827",
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "right", labels: { color: "#cbd5e1", font: { size: 11 } } }
            }
        }
    });
}

function renderEdaInfectionChart(infDist) {
    const ctx = document.getElementById("edaInfectionChart");
    if (!ctx || !window.Chart || !infDist) return;

    const labels = Object.keys(infDist).map(k => k.replace("_", " "));
    const values = Object.values(infDist);

    if (state.chartInstances["edaInfectionChart"]) {
        state.chartInstances["edaInfectionChart"].destroy();
    }

    state.chartInstances["edaInfectionChart"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Patient Count",
                data: values,
                backgroundColor: "rgba(56, 189, 248, 0.7)",
                borderColor: "#38bdf8",
                borderWidth: 1,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } },
                y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderEdaComorbiditiesChart(comorb) {
    const ctx = document.getElementById("edaComorbiditiesChart");
    if (!ctx || !window.Chart || !comorb) return;

    const labels = Object.keys(comorb);
    const values = Object.values(comorb).map(v => (v * 100).toFixed(1));

    if (state.chartInstances["edaComorbiditiesChart"]) {
        state.chartInstances["edaComorbiditiesChart"].destroy();
    }

    state.chartInstances["edaComorbiditiesChart"] = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Prevalence in Cohort (%)",
                data: values,
                backgroundColor: "rgba(16, 185, 129, 0.7)",
                borderColor: "#10b981",
                borderWidth: 1,
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { min: 0, max: 100, ticks: { color: "#94a3b8" }, grid: { color: "rgba(51, 65, 85, 0.3)" } },
                y: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });
}
