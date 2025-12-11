# scripts/models.py - ENHANCED for Classification and Regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, \
                            roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

class ModelTrainer:
    """
    A class to train and evaluate models for both Regression (Severity) 
    and Classification (Frequency) tasks.
    """
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    # --- Classification Training Methods ---
    
    def train_logistic_regression(self):
        self.lr_clf_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        self.lr_clf_model.fit(self.X_train, self.y_train)
        return self.lr_clf_model

    def train_rf_classifier(self, n_estimators=200, max_depth=10):
        self.rf_clf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        self.rf_clf_model.fit(self.X_train, self.y_train)
        return self.rf_clf_model

    def train_xgb_classifier(self, n_estimators=200, max_depth=6, learning_rate=0.1):
        self.xgb_clf_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, 
                                                use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        self.xgb_clf_model.fit(self.X_train, self.y_train)
        return self.xgb_clf_model

    # --- Regression Training Methods (Your existing code) ---
    
    def train_linear_regression(self):
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)
        return self.lr_model

    def train_random_forest(self, n_estimators=200, max_depth=10):
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        self.rf_model.fit(self.X_train, self.y_train)
        return self.rf_model

    def train_xgboost(self, n_estimators=200, max_depth=6, learning_rate=0.1):
        self.xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, 
                                          random_state=42, n_jobs=-1)
        self.xgb_model.fit(self.X_train, self.y_train)
        return self.xgb_model

    # --- Evaluation Methods ---
    
    def evaluate_regression(self, model, name="Model"):
        y_pred = model.predict(self.X_test)

        # Ensure predictions are non-negative for claim amounts
        y_pred = np.maximum(0, y_pred) 

        # Compute RMSE manually for compatibility
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)

        print(f"{name} (Regression): RMSE={rmse:.2f}, R²={r2:.4f}")
        return y_pred, rmse, r2


    def evaluate_classification(self, model, name="Model"):
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        results = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(self.y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        print(f"\n--- {name} (Classification) ---")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
            
        return y_pred_proba, results