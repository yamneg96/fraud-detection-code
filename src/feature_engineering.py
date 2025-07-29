import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def get_features_and_labels(df, label_col='is_fraud'):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y