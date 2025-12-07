# Insurance Portfolio EDA and Analysis

This repository contains exploratory data analysis (EDA) and preprocessing scripts for an insurance portfolio dataset. The analysis aims to uncover patterns in risk, profitability, and temporal trends within the data.

## Project Structure

- `data/`  
  Contains the raw and processed data files.

- `notebooks/`  
  Jupyter notebooks for interactive exploration and analysis.  
  Key notebook: `eda.ipynb` — contains the main exploratory data analysis workflow.

- `scripts/`  
  Python modules for data preprocessing, EDA, and plotting, designed to be reusable and modular.  
  - `data_preprocessing.py` — data cleaning and type conversions.  
  - `eda.py` — statistical summaries and EDA logic.  
  - `plot.py` — visualization functions for univariate, bivariate, and segmented analyses.  
  - `utils.py` — utility functions (e.g., data loading).

## How to Use

1. Clone the repo and install dependencies from `requirements.txt`.  
2. Use the `data_preprocessing` module to clean and prepare data.  
3. Perform exploratory data analysis using the `eda` module.  
4. Generate visualizations using the `plot` module or the provided notebook.  

## Key Analyses

- Loss Ratio calculation and segmentation by Province, Vehicle Type, and Gender.  
- Distribution and outlier detection in financial variables like TotalClaims and CustomValueEstimate.  
- Temporal trends in claims and premiums over the transaction period.  
- Risk profiling by vehicle make and ZIP code.  

## Requirements

See `requirements.txt` for the necessary Python libraries.


