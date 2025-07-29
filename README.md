# Fraud Detection Code
- This repository contains a complete pipeline to detect fraudulent transactions using e-commerce and credit card data. 

## Structure
- `data/raw/`: Original CSV datasets.
- `data/processed/`: Cleaned and merged fraud data.
- `notebooks/`: EDA, feature engineering, model training, explainability.
- `src/`: Python modules for each pipeline step.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocessing: `python src/preprocess.py`
3. Train models: `python src/train.py`
4. Evaluate models: `python src/evaluate.py`
5. Visualize SHAP explanations: `python src/shap_explain.py`

## Models
- Logistic Regression
- XGBoost Classifier

## Metrics
- F1 Score
- ROC AUC
- Average Precision
- Confusion Matrix

## Notes
- Handles imbalanced classes using SMOTE.
- SHAP is used for model interpretability.

---

Developed for Week 8 of the Adey Innovations Inc. ❤️ YN