# scripts/feature_importance.py - ENHANCED for model type handling

import pandas as pd
import shap
import numpy as np

class FeatureInterpreter:
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test

    def shap_summary(self, top_n=10):
        
        # Determine the appropriate explainer type
        model_name = type(self.model).__name__
        if 'XGB' in model_name or 'RandomForest' in model_name:
            explainer = shap.TreeExplainer(self.model)
        elif 'Linear' in model_name or 'Logistic' in model_name:
            # Use background data for Linear Explainer
            explainer = shap.LinearExplainer(self.model, self.X_test) 
        else:
            explainer = shap.Explainer(self.model)
            
        shap_values = explainer.shap_values(self.X_test)
        
        # If classification model, use SHAP values for the positive class (index 1)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1] 
            
        # Get Mean Absolute SHAP values for quantitative comparison
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values(by='Mean_Abs_SHAP', ascending=False)
        
        # Generate plot (will render in the notebook)
        print(f"\n--- SHAP Feature Importance (Top {top_n}) ---")
        shap.summary_plot(shap_values, self.X_test, max_display=top_n) 
        
        
        return feature_importance_df.head(top_n), shap_values