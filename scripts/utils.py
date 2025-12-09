import pandas as pd 
import os

class DataLoader:
    """Data loader for processed reviews"""

    def __init__(self, path=None):
        self.path = path
        self.df = None

    def load_data(self, path=None):
        """Load file into a DataFrame"""
        file_path = path or self.path
        if not file_path:
            raise ValueError("No file path specified for loading data.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.df = pd.read_csv(file_path, sep='|', low_memory=False)
        return self.df

    def save_data(self, output_path, sep=None):
        """Save current DataFrame to CSV"""
        self.sep = sep
        if self.df is None:
            raise ValueError("No data loaded to save.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, sep='|', index=False)
