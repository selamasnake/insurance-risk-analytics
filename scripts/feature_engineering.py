import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class FeatureEngineering:
    """
    Handles feature creation, encoding, and preparation of modeling datasets 
    for insurance claim frequency (classification) and severity (regression) tasks.
    Assumes the data is already cleaned and preprocessed.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
    
    def create_features(self) -> pd.DataFrame:
        """
        Creates additional features and encodes categorical variables.
        """
        df = self.data.copy()

        # ---  Feature Creation ---

        # Binary Target for Claim Frequency
        if 'TotalClaims' in df.columns:
            df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

        #  Vehicle Age
        if 'TransactionMonth' in df.columns and pd.api.types.is_datetime64_any_dtype(df['TransactionMonth']):
            current_year = df['TransactionMonth'].dt.year.max()
        else:
            current_year = 2015  # fallback if no datetime
        if 'RegistrationYear' in df.columns:
            df['VehicleAge'] = current_year - df['RegistrationYear']

        #  Drop original date columns after transformation
        df = df.drop(columns=['TransactionMonth', 'VehicleIntroDate'], errors='ignore')

        # ---  Encoding Categorical Variables ---
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        self.data = df
        return df

    def prepare_modeling_data(self, target_freq: str = 'HasClaim', target_sev: str = 'TotalClaims', test_size: float = 0.2):
        """
        Splits data into training and test sets for frequency and severity modeling.
        Returns:
            X_train_full, X_test_full, y_freq_train, y_freq_test,
            X_train_sev, y_sev_train, X_test_sev, y_sev_test
        """
        df = self.data.copy()
        X = df.drop(columns=[target_freq, target_sev], errors='ignore')
        y_freq = df[target_freq] if target_freq in df.columns else None
        y_sev = df[target_sev] if target_sev in df.columns else None

        # --- 1. Train-Test Split for Frequency ---
        if y_freq is not None:
            X_train_full, X_test_full, y_freq_train, y_freq_test = train_test_split(
                X, y_freq, test_size=test_size, random_state=42, stratify=y_freq
            )
        else:
            X_train_full, X_test_full, y_freq_train, y_freq_test = X, X, None, None

        # --- 2. Severity Subset (only claims > 0) ---
        if y_sev is not None:
            y_sev_train_all = y_sev.loc[X_train_full.index]
            claim_indices_train = y_sev_train_all[y_sev_train_all > 0].index
            X_train_sev = X_train_full.loc[claim_indices_train]
            y_sev_train = y_sev_train_all.loc[claim_indices_train]

            y_sev_test_all = y_sev.loc[X_test_full.index]
            claim_indices_test = y_sev_test_all[y_sev_test_all > 0].index
            X_test_sev = X_test_full.loc[claim_indices_test]
            y_sev_test = y_sev_test_all.loc[claim_indices_test]
        else:
            X_train_sev, y_sev_train, X_test_sev, y_sev_test = None, None, None, None

        return (X_train_full, X_test_full, y_freq_train, y_freq_test,
                X_train_sev, y_sev_train, X_test_sev, y_sev_test)
