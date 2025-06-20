
# 🧠 Alzheimer’s Disease Prediction using Machine Learning

This repository contains a machine learning pipeline for predicting the stages of Alzheimer’s disease using clinical, cognitive, and neuroimaging data from the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset (`ADNIMERGE.csv`). The project includes preprocessing, feature selection, classification using multiple ML models, evaluation, and optional hyperparameter tuning.

---

## 📁 Project Structure

```bash
├── initialising.py              # File upload and basic dataset loading
├── data_preprocessing.py       # Data cleaning, imputation, encoding
├── feature_selection.py        # Feature selection with RandomForest importance
├── model_development.py        # Core pipeline definition and model evaluation
├── model_pipeline.py           # Model training for Logistic Regression, Random Forest, and SVM
├── model_optimization.py       # Hyperparameter tuning using GridSearchCV
└── ADNIMERGE.csv               # Dataset (to be uploaded manually via Colab)
```

---

## 📊 Dataset Overview

- **Source**: [ADNI Dataset](https://adni.loni.usc.edu/)
- **File used**: `ADNIMERGE.csv`
- **Target Column**: `DX` (Diagnosis)
- **Classes**: 
  - CN (Cognitively Normal)
  - MCI (Mild Cognitive Impairment)
  - Dementia
  - NaN (Unspecified/Missing class — handled in preprocessing)

---

## ⚙️ Workflow Summary

1. **Initialization & Upload**:
   - Upload `ADNIMERGE.csv` manually via Google Colab interface.
   - Read into a Pandas DataFrame.

2. **Data Preprocessing**:
   - Replace empty strings/whitespace with NaN.
   - Convert relevant columns to numeric.
   - Drop columns with >70% missing values.
   - Remove identifiers and irrelevant metadata.
   - Encode target labels using `LabelEncoder`.

3. **Feature Selection**:
   - Applied `SelectFromModel` using `RandomForestClassifier`.
   - Only features above median importance are retained.
   - Pipeline combines numerical + categorical feature handling.

4. **Model Training**:
   - Models Trained:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
   - Each model is trained and evaluated on 80/20 train-test split.

5. **Model Evaluation**:
   - Confusion Matrix
   - Classification Report (Accuracy, Precision, Recall, F1-Score)
   - ROC AUC Score (Weighted, Multi-class)
   - Feature importance visualization

6. **Model Saving**:
   - Trained models are saved as `.pkl` files for reuse.

7. **Optional Optimization**:
   - Hyperparameter tuning via `GridSearchCV`
   - Parameter grids defined for Logistic Regression and Random Forest.

---

## 🧪 Results Summary

| Model              | Accuracy | Weighted AUC | Notes                            |
|-------------------|----------|--------------|----------------------------------|
| Logistic Regression | ~0.89   | 0.9749       | Multinomial, requires scaling    |
| Random Forest       | ~0.95   | 0.9947       | Robust, top performance          |
| SVM (RBF kernel)    | ~0.93   | 0.9878       | Sensitive to feature scaling     |

---

## 🛠 Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- seaborn / matplotlib
- joblib
- Google Colab (`files.upload()`)

Install packages (locally):

```bash
pip install pandas scikit-learn seaborn matplotlib joblib
```

---

## 🧠 How to Run (Colab Recommended)

1. Open Google Colab.
2. Upload all `.py` files into the Colab environment.
3. Run `initialising.py` to upload `ADNIMERGE.csv`.
4. Run the rest of the scripts in order:
   - `data_preprocessing.py`
   - `feature_selection.py`
   - `model_development.py`
   - `model_pipeline.py` (to run all models)
   - `model_optimization.py` (optional)

---

## 📌 Project Highlights

- Designed for **multi-class classification** of Alzheimer stages.
- Flexible architecture to integrate other classifiers or tuning methods.
- Visual outputs make evaluation intuitive (heatmaps, metrics).
- Final models are saved as `.pkl` pipelines for deployment.

---



