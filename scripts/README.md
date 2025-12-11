# Scripts Directory

This folder contains modular Python scripts for data preprocessing, exploratory data analysis, and plotting related to the insurance portfolio dataset.

## Files

- `data_preprocessing.py`  
  Handles data cleaning, type conversion, and missing value treatment.

- `eda.py`  
  Contains the `ExploratoryDataAnalysis` class which performs statistical summaries and bivariate analysis.

- `plot.py`  
  Implements the `EDAPlots` class with functions to create various insightful visualizations (e.g., loss ratio by segment, vehicle risk profiles, monthly trends).

- `utils.py`  
  Utility functions such as data loading and other helpers.

- `feature_engineering.py`
Provides the `FeatureEngineering` class for creating additional features, encoding categorical variables, and preparing datasets for modeling (classification for claim frequency, regression for claim severity).

- `feature_importance.py`
Contains the `FeatureInterpreter` class which computes SHAP values to explain model predictions and determine feature importance.

- `models.py`
Implements the `ModelTrainer` class for training and evaluating multiple models for classification (frequency) and regression (severity), including Logistic Regression, Random Forest, and XGBoost.

- `hypothesis_testing.py`
Performs statistical tests and hypothesis evaluation on dataset segments to identify significant relationships.

- `__init__.py`
Makes the scripts directory a Python package for easy imports.

## Usage

Each script provides classes and functions that can be imported and used independently or together in analysis workflows.

