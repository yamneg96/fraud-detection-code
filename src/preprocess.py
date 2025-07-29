import pandas as pd

def load_data():
    fraud_df = pd.read_csv('data/raw/Fraud_Data.csv')
    credit_df = pd.read_csv('data/raw/creditcard.csv')
    ip_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    return fraud_df, credit_df, ip_df

def merge_geo_data(fraud_df, ip_df):
    return fraud_df.merge(ip_df, left_on='ip_address', right_on='ip', how='left')

def preprocess_fraud_data():
    fraud_df, _, ip_df = load_data()
    fraud_df = merge_geo_data(fraud_df, ip_df)
    fraud_df.drop(['ip', 'ip_address'], axis=1, inplace=True)
    fraud_df.to_csv('data/processed/fraud_processed.csv', index=False)