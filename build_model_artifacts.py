"""
build_model_artifacts.py
═══════════════════════════════════════════════════════════════════════════════
Clean, robust Machine Learning pipeline for Heart Disease Prediction (UCI Cleveland).
Implements:
  1. Data loading, cleaning, and preprocessing (StandardScaler + OneHotEncoder).
  2. Hyperparameter Tuning using GridSearchCV on SVM (C and gamma optimization).
  3. SVM Kernel Comparison (Linear, Polynomial, RBF Tuned, Sigmoid).
  4. 5-Classifier Comparison from project syllabus (RF, SVM, LR, KNN, DT).
  5. Overall Data Analysis Diagrams (2D PCA projection, Clinical Radar Profiles).
  6. Artifact export for the Flask interactive web dashboard.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Suppress minor future warnings for clean console output
warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_clean_data(filepath="Heart Disease UCI.xlsx"):
    """
    Loads raw UCI dataset, filters for the Cleveland cohort, cleans missing
    values ('?'), and formats target and feature datatypes.
    """
    print("[1/6] Loading & cleaning Cleveland Heart Disease dataset...")
    df_raw = pd.read_excel(filepath)
    cleveland = df_raw[df_raw["dataset"] == "Cleveland"].copy()
    cleveland = cleveland.drop(columns=["id", "dataset"], errors="ignore")

    # Rename to canonical names used in the syllabus
    cleveland = cleveland.rename(columns={
        "trestbps": "restbp",
        "thalch":   "thalach",
        "num":      "hd",
    })

    # Remove incomplete records
    cleveland = cleveland.replace("?", np.nan).dropna().copy()

    # Binarize target: 0 = No Disease, >0 = Heart Disease
    cleveland["hd"] = (cleveland["hd"] > 0).astype(int)

    num_cols = ["age", "restbp", "chol", "thalach", "oldpeak"]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    for col in num_cols:
        cleveland[col] = pd.to_numeric(cleveland[col])
    for col in cat_cols:
        cleveland[col] = cleveland[col].astype(str)

    print(f"      Loaded {len(cleveland)} records with {len(num_cols) + len(cat_cols)} features.")
    print(f"      Class distribution: {dict(cleveland['hd'].value_counts())}")
    return cleveland, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """
    Constructs a ColumnTransformer that standardizes numerical features
    and one-hot encodes categorical variables without data leakage.
    """
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
    ])


def tune_svm_hyperparameters(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Performs grid search cross-validation across C and gamma parameters for RBF SVM.
    Compares baseline default parameters vs the optimal tuned parameters.
    """
    print("[2/6] Performing Hyperparameter Tuning for SVM (GridSearchCV)...")

    # 1. Baseline Untuned Model (Default C=1.0, gamma='scale')
    baseline_svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    baseline_svm.fit(X_train_scaled, y_train)
    y_pred_base = baseline_svm.predict(X_test_scaled)
    y_prob_base = baseline_svm.predict_proba(X_test_scaled)[:, 1]

    baseline_metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred_base)), 4),
        "precision": round(float(precision_score(y_test, y_pred_base, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred_base, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred_base, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(y_test, y_prob_base)), 4),
    }

    # 2. Grid Search over C and gamma
    c_values = [0.1, 1.0, 10.0, 100.0]
    gamma_values = [0.001, 0.01, 0.1, 1.0]
    param_grid = {"C": c_values, "gamma": gamma_values}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=42),
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train_scaled, y_train)

    best_svm = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = round(float(grid_search.best_score_), 4)

    y_pred_tuned = best_svm.predict(X_test_scaled)
    y_prob_tuned = best_svm.predict_proba(X_test_scaled)[:, 1]

    tuned_metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred_tuned)), 4),
        "precision": round(float(precision_score(y_test, y_pred_tuned, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred_tuned, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred_tuned, zero_division=0)), 4),
        "auc":       round(float(roc_auc_score(y_test, y_prob_tuned)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_tuned).tolist(),
    }

    # Extract 2D grid matrix of mean CV scores for the Heatmap diagram
    mean_scores = grid_search.cv_results_["mean_test_score"]
    score_matrix = mean_scores.reshape(len(c_values), len(gamma_values)).round(4).tolist()

    tuning_data = {
        "c_values": c_values,
        "gamma_values": gamma_values,
        "score_matrix": score_matrix,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
    }

    print(f"      Optimal Parameters: C={best_params['C']}, gamma={best_params['gamma']} (CV Score: {best_cv_score})")
    print(f"      Tuned Test Accuracy: {tuned_metrics['accuracy']}, Recall: {tuned_metrics['recall']}")
    return best_svm, best_params, tuning_data


def evaluate_svm_kernels(X_train_scaled, y_train, X_test_scaled, y_test, tuned_rbf_model):
    """
    Trains and compares Linear, Polynomial, RBF (Tuned), and Sigmoid kernels.
    Computes performance metrics, confusion matrices, and ROC curve coordinates.
    """
    print("[3/6] Comparing SVM Kernels (Linear, Polynomial, RBF Tuned, Sigmoid)...")

    kernel_models = {
        "linear":  SVC(kernel="linear",  C=1.0, probability=True, random_state=42),
        "poly":    SVC(kernel="poly",    C=1.0, degree=3, probability=True, random_state=42),
        "rbf":     tuned_rbf_model,
        "sigmoid": SVC(kernel="sigmoid", C=1.0, gamma="scale", probability=True, random_state=42),
    }

    kernel_results = {}
    kernel_roc_curves = {}

    for name, model in kernel_models.items():
        if name != "rbf":
            model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Sample points to keep payload compact
        step = max(1, len(fpr) // 15)
        sampled_fpr = fpr[::step].round(3).tolist()
        sampled_tpr = tpr[::step].round(3).tolist()
        if sampled_fpr[-1] != 1.0:
            sampled_fpr.append(1.0)
            sampled_tpr.append(1.0)

        label_map = {
            "linear":  "Linear Kernel",
            "poly":    "Polynomial (d=3)",
            "rbf":     "RBF (Tuned)",
            "sigmoid": "Sigmoid Kernel",
        }

        kernel_results[name] = {
            "kernel":           label_map[name],
            "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":        round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":           round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1":               round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc":              round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "support_vectors":  int(np.sum(model.n_support_)) if hasattr(model, "n_support_") else 0,
        }

        kernel_roc_curves[name] = {
            "name": label_map[name],
            "fpr":  sampled_fpr,
            "tpr":  sampled_tpr,
            "auc":  kernel_results[name]["auc"],
        }

    return kernel_results, kernel_models, kernel_roc_curves


def evaluate_five_classifiers(X_train_scaled, y_train, X_test_scaled, y_test, tuned_svm):
    """
    Evaluates the 5 classifiers specified in the project syllabus:
    Random Forest, SVM (Tuned), Logistic Regression, KNN, Decision Tree.
    """
    print("[4/6] Evaluating 5 Classifiers from project syllabus...")

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Tuned)":     tuned_svm,
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }

    results = {}
    for name, clf in models.items():
        if name != "SVM (RBF Tuned)":
            clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = clf.decision_function(X_test_scaled)

        results[name] = {
            "accuracy":         round(float(accuracy_score(y_test, y_pred)), 4),
            "precision":        round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":           round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1":               round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc":              round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    return results


def compute_overall_diagrams(cleveland, X_scaled_all, num_cols):
    """
    Computes high-level dataset summary diagrams:
      1. 2D PCA projection of all 297 patients to visualize separation in feature space.
      2. Clinical Radar Profile comparing normalized averages for healthy vs disease cohorts.
    """
    print("[5/6] Generating Overall Data Diagrams (PCA Projection & Clinical Radar)...")

    # 1. 2D PCA Dimensionality Reduction
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled_all)
    var_ratio = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

    pca_records = []
    for i in range(len(cleveland)):
        pca_records.append({
            "pc1": round(float(pca_coords[i, 0]), 3),
            "pc2": round(float(pca_coords[i, 1]), 3),
            "hd":  int(cleveland.iloc[i]["hd"]),
            "age": int(cleveland.iloc[i]["age"]),
            "sex": str(cleveland.iloc[i]["sex"]),
        })

    # 2. Clinical Profile Comparison (Normalized Means for Radar Chart)
    # Compare 6 key clinical biomarkers: Age, RestBP, Chol, MaxHR, Oldpeak, Vessels (ca)
    radar_features = [
        {"key": "age",     "label": "Age",             "min": 25,  "max": 80},
        {"key": "restbp",  "label": "Resting BP",      "min": 90,  "max": 200},
        {"key": "chol",    "label": "Cholesterol",     "min": 120, "max": 400},
        {"key": "thalach", "label": "Max Heart Rate",  "min": 70,  "max": 210},
        {"key": "oldpeak", "label": "ST Depression",   "min": 0,   "max": 6.0},
    ]

    healthy_mask = cleveland["hd"] == 0
    disease_mask = cleveland["hd"] == 1

    radar_labels = [f["label"] for f in radar_features]
    healthy_norm_vals = []
    disease_norm_vals = []

    for f in radar_features:
        k = f["key"]
        f_min, f_max = f["min"], f["max"]
        h_mean = float(cleveland.loc[healthy_mask, k].mean())
        d_mean = float(cleveland.loc[disease_mask, k].mean())

        # Scale 0-100 for clear radar visualization
        h_norm = round(max(0, min(100, (h_mean - f_min) / (f_max - f_min) * 100)), 1)
        d_norm = round(max(0, min(100, (d_mean - f_min) / (f_max - f_min) * 100)), 1)

        healthy_norm_vals.append(h_norm)
        disease_norm_vals.append(d_norm)

    return {
        "pca": {
            "records": pca_records,
            "variance_ratio": var_ratio,
        },
        "clinical_radar": {
            "labels":  radar_labels,
            "healthy": healthy_norm_vals,
            "disease": disease_norm_vals,
        }
    }


def compute_statistics(cleveland, num_cols, cat_cols):
    """
    Computes numerical statistics, categorical value counts, and correlation matrix.
    """
    num_stats = {}
    for c in num_cols:
        s = cleveland[c]
        num_stats[c] = {
            "mean":   round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std":    round(float(s.std()), 2),
            "min":    round(float(s.min()), 2),
            "max":    round(float(s.max()), 2),
            "q25":    round(float(s.quantile(0.25)), 2),
            "q75":    round(float(s.quantile(0.75)), 2),
        }

    cat_counts = {c: cleveland[c].value_counts().to_dict() for c in cat_cols}

    corr_cols = num_cols + ["hd"]
    corr_df = cleveland[corr_cols].astype(float).corr().round(4)
    correlation_matrix = {
        "columns": corr_cols,
        "values":  corr_df.values.tolist(),
    }

    class_dist = {
        "No Disease (0)":    int((cleveland["hd"] == 0).sum()),
        "Heart Disease (1)": int((cleveland["hd"] == 1).sum()),
    }

    return num_stats, cat_counts, correlation_matrix, class_dist


def main():
    cleveland, num_cols, cat_cols = load_and_clean_data("Heart Disease UCI.xlsx")

    X = cleveland.drop(columns=["hd"])
    y = cleveland["hd"]

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Stratified 80/20 train/test split matching scientific evaluation standards
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_s = preprocessor.fit_transform(X_train)
    X_test_s  = preprocessor.transform(X_test)
    X_all_s   = preprocessor.transform(X)

    # 1. Hyperparameter Tuning
    best_svm, best_params, tuning_data = tune_svm_hyperparameters(
        X_train_s, y_train, X_test_s, y_test
    )

    # 2. Kernel Comparison
    kernel_results, kernel_models, kernel_roc_curves = evaluate_svm_kernels(
        X_train_s, y_train, X_test_s, y_test, best_svm
    )

    # 3. 5-Classifier Comparison
    classifier_results = evaluate_five_classifiers(
        X_train_s, y_train, X_test_s, y_test, best_svm
    )

    # 4. Overall Diagrams (PCA & Clinical Radar)
    overall_diagrams = compute_overall_diagrams(cleveland, X_all_s, num_cols)

    # 5. Core Dataset Statistics
    num_stats, cat_counts, correlation_matrix, class_dist = compute_statistics(
        cleveland, num_cols, cat_cols
    )

    # 6. Artifact Export
    print("[6/6] Exporting artifacts (svm_model_pipeline.joblib & dataset_stats.json)...")

    pipeline_export = {
        "preprocessor":   preprocessor,
        "model":          best_svm,
        "kernel_models":  kernel_models,
        "num_cols":       num_cols,
        "cat_cols":       cat_cols,
        "best_params":    best_params,
        "tuning_data":    tuning_data,
        "kernel_results": kernel_results,
    }
    joblib.dump(pipeline_export, "svm_model_pipeline.joblib")

    stats_export = {
        "total_records":      len(cleveland),
        "features_count":     len(X.columns),
        "class_distribution": class_dist,
        "num_stats":          num_stats,
        "cat_counts":         cat_counts,
        "correlation_matrix": correlation_matrix,
        "classifier_results": classifier_results,
        "kernel_results":     kernel_results,
        "kernel_roc_curves":  kernel_roc_curves,
        "tuning_data":        tuning_data,
        "overall_diagrams":   overall_diagrams,
        "records":            cleveland.to_dict(orient="records"),
    }
    with open("dataset_stats.json", "w") as f:
        json.dump(stats_export, f, indent=2)

    print("[SUCCESS] All model artifacts and comprehensive dataset statistics exported successfully.")


if __name__ == "__main__":
    main()
