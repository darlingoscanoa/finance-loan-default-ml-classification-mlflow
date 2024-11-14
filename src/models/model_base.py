#Base Model Class

# src/models/model_base.py

import mlflow
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.visualization.model_evaluation import ModelEvaluationPlotter

class BaseModel(ABC):
    """Base class for all models in the project."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.plotter = ModelEvaluationPlotter()
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model and log metrics to MLflow."""
        y_pred = self.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        with mlflow.start_run(run_name=self.model_name):
            # Log model parameters
            mlflow.log_params(self.model.get_params())
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Generate and log plots
            self.plotter.plot_confusion_matrix(y_test, y_pred, self.model_name)
            self.plotter.plot_roc_curve(y_test, y_pred_proba, self.model_name)
            
            # Log model
            mlflow.sklearn.log_model(self.model, "model")
        
        return metrics