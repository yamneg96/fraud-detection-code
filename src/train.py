import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib

from feature_engineering import encode_features, get_features_and_labels

def train_models(df):
    df = encode_features(df)
    X, y = get_features_and_labels(df)
    X_res, y_res = SMOTE().fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'models/logistic_regression.pkl')

    # XGBoost
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss')
    xgb_clf.fit(X_train, y_train)
    joblib.dump(xgb_clf, 'models/xgboost_model.pkl')

    return X_test, y_test