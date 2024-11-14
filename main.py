# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from src.data.data_loader import DataLoader
from src.visualization.model_evaluation import ModelEvaluationPlotter
from src.models.rf_model import train_random_forest
from src.models.xgb_model import train_xgboost
from src.models.xgb_tuned_model import train_xgboost_tuned
from src.models.lgb_model import train_lightgbm

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data/raw', 'data/processed', 'visualization_outputs']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs[2]  # Return visualization outputs directory

def load_and_prepare_data(url):
    """Load and prepare the dataset."""
    logger.info("Loading and preparing data...")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load and split data
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        url=url,
        target_column='default',
        test_size=0.2
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def run_experiments():
    """Run all model experiments."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Setup directories
    viz_output_dir = setup_directories()
    
    # Initialize plotter
    plotter = ModelEvaluationPlotter(viz_output_dir)
    
    # Load data
    url = 'https://github.com/Safa1615/Dataset--loan/blob/main/bank-loan.csv?raw=true'
    X_train, X_test, y_train, y_test = load_and_prepare_data(url)
    
    # Dictionary to store results
    results = {}
    
    # Train and evaluate each model
    models = {
        'RandomForest': train_random_forest,
        'XGBoost': train_xgboost,
        'XGBoostTuned': train_xgboost_tuned,
        'LightGBM': train_lightgbm
    }
    
    for model_name, train_func in models.items():
        logger.info(f"Training {model_name}...")
        with mlflow.start_run(run_name=model_name):
            model, metrics = train_func(X_train, X_test, y_train, y_test)
            results[model_name] = metrics
            
            # Generate prediction probabilities for ROC curve
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Create and save plots
            plotter.plot_confusion_matrix(y_test, y_pred, model_name)
            plotter.plot_roc_curve(y_test, y_pred_proba, model_name)
    
    # Print final results
    results_df = pd.DataFrame(results).round(3)
    print("\nModel Performance Summary:")
    print(results_df)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(viz_output_dir, 'model_performance_summary.csv'))
    
    return results_df

if __name__ == "__main__":
    run_experiments()