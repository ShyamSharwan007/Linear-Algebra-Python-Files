import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalize_data(df):
    """
    Normalizes the dataframe using Min-Max and Z-Score methods.
    
    Args:
        df: pandas DataFrame containing raw scores
        
    Returns:
        df_minmax: DataFrame with values scaled 0-1
        df_zscore: DataFrame with values centered around mean with unit variance
    """
    # Min-Max Normalization (0 to 1)
    # Useful for weighted scoring
    scaler_minmax = MinMaxScaler()
    df_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    # Z-Score Normalization (StandardScaler)
    # Useful for PCA/SVD
    scaler_zscore = StandardScaler()
    df_zscore = pd.DataFrame(
        scaler_zscore.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    return df_minmax, df_zscore
