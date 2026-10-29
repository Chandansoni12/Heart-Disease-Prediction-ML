# Heart Disease Prediction Using Machine Learning

**Author:** [Chandansoni12](https://github.com/Chandansoni12)  
**Repository:** [https://github.com/Chandansoni12/Heart-Disease-Prediction-ML](https://github.com/Chandansoni12/Heart-Disease-Prediction-ML)

---

## Overview
An interactive Machine Learning web application and clinical analytical dashboard for predicting heart disease risk based on the UCI Cleveland Heart Disease dataset. The project implements feature engineering, hyperparameter tuning, multi-kernel SVM evaluations, and real-time patient prediction with nearest-neighbor similarity.

## Key Features
- **Interactive Web Dashboard**: Modern responsive UI with 6 dedicated analytical views:
  1. **Data Overview & Overall Diagrams**: Dataset summary, class balance distributions, and 2D PCA projection.
  2. **Feature Distributions**: Histograms and boxplots for continuous biomarkers (Age, BP, Cholesterol, Max HR, ST Depression).
  3. **Correlations & Relationships**: Correlation heatmap matrix and pairwise biomarker relationships.
  4. **Feature vs Target**: Categorical risk analysis across Chest Pain types, Thalassemia, ST slope, and ECG readings.
  5. **Model, Tuning & Kernels**: 5-classifier benchmark (Random Forest, SVM, Logistic Regression, KNN, Decision Tree), hyperparameter grid search heatmap ($C$ vs $\gamma$), and kernel evaluations (Linear, Polynomial, RBF, Sigmoid) with ROC curves.
  6. **Patient Prediction**: Real-time interactive clinical diagnosis form with risk probability, SVM decision score, comparative biomarker percentiles, and top 4 similar patients (Euclidean distance).
- **Optimized SVM Pipeline**: Hyperparameter-tuned Support Vector Machine (RBF kernel) utilizing stratified cross-validation.
- **RESTful API**: Fast endpoints for metrics (`/api/stats`) and inference (`/api/predict`).

---

## Project Structure
```
├── app.py                      # Flask web server & inference API
├── build_model_artifacts.py    # ML pipeline, model training & artifact generation
├── dataset_stats.json          # Precomputed dataset statistics & evaluation metrics
├── svm_model_pipeline.joblib   # Trained SVM pipeline model
├── Heart Disease UCI.xlsx      # UCI Cleveland dataset
├── static/
│   ├── app.js                  # Frontend interactive charts & prediction logic
│   └── style.css               # Modern custom stylesheet
├── templates/
│   └── index.html              # Dashboard single-page template
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/Chandansoni12/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## (Optional) Retrain or Rebuild Artifacts
To rerun the machine learning training pipeline and regenerate model artifacts:
```bash
python build_model_artifacts.py
```

---

## License & Credits
- **Dataset**: [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- Developed & Maintained by **[Chandansoni12](https://github.com/Chandansoni12)**.
