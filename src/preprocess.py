import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_cols):
    x = df.drop(columns=target_cols)

    numeric_features = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x.select_dtypes(include=['object', 'category']).columns

    numeric_pipeline = SimpleImputer(strategy = 'median')
    categorical_pipeline = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', categorical_pipeline, cat_cols),
        ])
    return preprocessor, x