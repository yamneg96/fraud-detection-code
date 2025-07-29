from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib

def evaluate_model(X_test, y_test):
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'XGBoost': joblib.load('models/xgboost_model.pkl')
    }
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\nModel: {name}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("ROC AUC:", roc_auc_score(y_test, y_pred))
        print("Average Precision:", average_precision_score(y_test, y_pred))
