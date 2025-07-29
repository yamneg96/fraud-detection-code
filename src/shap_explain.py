import shap
import joblib

def explain_model(X):
    model = joblib.load('models/xgboost_model.pkl')
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)