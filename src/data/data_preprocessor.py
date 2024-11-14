# src/data/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handle all data preprocessing tasks."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numerical_features = None
        self.categorical_features = None
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load the dataset and perform initial cleaning."""
        data = pd.read_csv(filepath)
        data = data.head(700)  # Added line to limit to 700 records
        return self._clean_data(data)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by handling missing values and outliers."""
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Handle missing values
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                data[column].fillna(data[column].median(), inplace=True)
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)
        
        return data
    
    def prepare_features(self, data: pd.DataFrame):
        """Prepare features for modeling."""
        # Identify numerical and categorical columns
        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_features = data.select_dtypes(include=['object']).columns
        
        # Handle categorical variables
        data = pd.get_dummies(data, columns=self.categorical_features)
        
        # Scale numerical features
        data[self.numerical_features] = self.scaler.fit_transform(data[self.numerical_features])
        
        return data
    
    def get_train_test_split(self, data: pd.DataFrame, target_column: str, test_size: float = 0.2):
        """Split data into training and test sets."""
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        return train_test_split(X, y, test_size=test_size, random_state=42)