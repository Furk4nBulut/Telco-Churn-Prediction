# data/data_loader.py

import pandas as pd

class DataLoader:
    @staticmethod
    def load_data(file_path):
        """Load the dataset."""
        data = pd.read_csv(file_path)
        return data
