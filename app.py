"""
app.py — Flask Web Application for Heart Disease Prediction & Analysis Dashboard
════════════════════════════════════════════════════════════════════════════════
Serves:
  - Interactive Dashboard UI (HTML / CSS / Vanilla JavaScript with Chart.js)
  - /api/stats: Comprehensive dataset statistics, EDA records, 5-classifier metrics,
    SVM kernel comparisons, hyperparameter tuning grid, and overall diagrams.
  - /api/predict: Real-time patient inference using the Hyperparameter-Tuned SVM
    pipeline (or selected kernel), patient-vs-dataset comparison, and nearest-neighbor
    similarity analysis.
════════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

PIPELINE_PATH = "svm_model_pipeline.joblib"
STATS_PATH    = "dataset_stats.json"

# Validate artifacts presence
if not os.path.exists(PIPELINE_PATH) or not os.path.exists(STATS_PATH):
    raise RuntimeError("Model artifacts missing. Run build_model_artifacts.py first.")

# Load pre-trained models and dataset statistics
pipeline = joblib.load(PIPELINE_PATH)
preprocessor   = pipeline["preprocessor"]
tuned_model    = pipeline["model"]
kernel_models  = pipeline.get("kernel_models", {})
num_cols       = pipeline["num_cols"]
cat_cols       = pipeline["cat_cols"]
best_params    = pipeline.get("best_params", {})

with open(STATS_PATH, "r") as f:
    dataset_stats = json.load(f)

# Pre-transform reference dataset records for fast Euclidean distance lookup
records_df = pd.DataFrame(dataset_stats["records"])
X_ref = records_df[num_cols + cat_cols].copy()
for col in cat_cols:
    X_ref[col] = X_ref[col].astype(str)
for col in num_cols:
    X_ref[col] = pd.to_numeric(X_ref[col])
X_ref_scaled = preprocessor.transform(X_ref)


@app.route("/")
def index():
    """Renders the main single-page dashboard."""
    return render_template("index.html")


@app.route("/api/stats")
def get_stats():
    """Returns dataset summary, classifier results, kernels, and tuning data."""
    return jsonify(dataset_stats)


@app.route("/api/predict", methods=["POST"])
def predict_patient():
    """
    Predicts heart disease for a given patient profile using the
    Hyperparameter-Tuned SVM (or an optional specified kernel).
    """
    try:
        data = request.json or {}

        # 1. Parse and sanitize patient attributes
        patient = {
            "age":     float(data.get("age", 55)),
            "sex":     str(data.get("sex", "Male")),
            "cp":      str(data.get("cp", "asymptomatic")),
            "restbp":  float(data.get("restbp", 130)),
            "chol":    float(data.get("chol", 240)),
            "fbs":     str(data.get("fbs", "0.0")),
            "restecg": str(data.get("restecg", "0.0")),
            "thalach": float(data.get("thalach", 150)),
            "exang":   str(data.get("exang", "0.0")),
            "oldpeak": float(data.get("oldpeak", 1.0)),
            "slope":   str(data.get("slope", "flat")),
            "ca":      str(data.get("ca", "0.0")),
            "thal":    str(data.get("thal", "normal")),
        }

        # 2. Select model (default: Hyperparameter-Tuned RBF SVM)
        selected_kernel = str(data.get("kernel", "rbf")).lower()
        if selected_kernel in kernel_models:
            active_model = kernel_models[selected_kernel]
            active_name  = f"SVM ({selected_kernel.upper()})"
        else:
            active_model = tuned_model
            active_name  = "Hyperparameter-Tuned SVM (RBF)"

        # 3. Preprocess and run inference
        patient_df = pd.DataFrame([patient])
        patient_scaled = preprocessor.transform(patient_df)

        prediction = int(active_model.predict(patient_scaled)[0])

        if hasattr(active_model, "predict_proba"):
            probability = float(active_model.predict_proba(patient_scaled)[0][1])
        else:
            probability = 0.5

        if hasattr(active_model, "decision_function"):
            decision_score = float(active_model.decision_function(patient_scaled)[0])
        else:
            decision_score = float(probability - 0.5)

        # 4. Patient-vs-Dataset comparison
        ns = dataset_stats["num_stats"]
        comparison = {}
        for col in num_cols:
            val = patient[col]
            comparison[col] = {
                "value":  val,
                "median": ns[col]["median"],
                "min":    ns[col]["min"],
                "max":    ns[col]["max"],
            }

        # 5. Nearest neighbors (Euclidean distance on scaled features)
        distances = euclidean_distances(patient_scaled, X_ref_scaled)[0]
        top_indices = np.argsort(distances)[:4]
        max_dist = max(float(np.max(distances)), 1.0)

        similar_patients = []
        for idx in top_indices:
            row = records_df.iloc[idx].to_dict()
            dist = float(distances[idx])
            sim_pct = round(max(0.0, min(100.0, (1.0 - dist / (max_dist + 1.0)) * 100.0)), 1)
            similar_patients.append({
                "record": row,
                "distance": round(dist, 3),
                "similarity_pct": sim_pct,
            })

        return jsonify({
            "status": "success",
            "model_used": active_name,
            "prediction": prediction,
            "prediction_label": "HEART DISEASE POSITIVE" if prediction == 1 else "NO HEART DISEASE",
            "probability": round(probability, 4),
            "decision_score": round(decision_score, 4),
            "patient_comparison": comparison,
            "similar_patients": similar_patients,
        })

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400


if __name__ == "__main__":
    print("=" * 63)
    print(" Heart Disease Prediction Dashboard Running on http://127.0.0.1:5000")
    print("=" * 63)
    app.run(host="0.0.0.0", port=5000, debug=True)
