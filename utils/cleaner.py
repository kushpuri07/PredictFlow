import pandas as pd
import numpy as np

def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        return None
    return df

def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.drop_duplicates()
    
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in [np.float64, np.int64]:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    return df_cleaned

def get_basic_info(df):
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "column_types": df.dtypes.astype(str).to_dict()
    }
    return info