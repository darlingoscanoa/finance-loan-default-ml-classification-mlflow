# src/data/data_loader.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        self.raw_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
        self.processed_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    
    def load_data(self, filename):
        data = pd.read_csv(os.path.join(self.raw_data_dir, filename))
        return data
    
    def get_train_test_split(self, url, target_column, test_size=0.2):
        data = pd.read_csv(url)
        data = data.head(700)  # Added line to limit to 700 records
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        os.makedirs(self.processed_data_dir, exist_ok=True)
        pd.DataFrame(X).to_csv(os.path.join(self.processed_data_dir, 'X.csv'), index=False)
        pd.DataFrame(y).to_csv(os.path.join(self.processed_data_dir, 'y.csv'), index=False)
        
        return train_test_split(X, y, test_size=test_size, random_state=42)