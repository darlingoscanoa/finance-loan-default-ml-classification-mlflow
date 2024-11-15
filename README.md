# AI-Powered Credit Risk Assessment: MLflow-Tracked Classification Models for Loan Default Prediction

## Overview

This repository showcases my experience in finance-focused machine learning, specifically the development and comparison of classification models for loan default prediction. The project utilizes the MLflow platform to track, compare, and gain insights from various model experiments, demonstrating expertise in applying modern ML techniques to solve real-world problems in the finance domain.

## Key Features

- Multiple classification models (Random Forest, XGBoost, XGBoost Tuned, LightGBM) with hyperparameter tuning
- MLflow integration for experiment tracking and model versioning
- Detailed model performance analysis with custom visualizations
- Comprehensive comparison using key performance metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC)
- Visualizations including feature importance plots, confusion matrices, and ROC curves
- Production-ready code structure with proper OOP implementation

## Project Structure
```
loan_default_prediction/
│
├── data/
│ ├── raw/ # Original dataset
│ └── processed/ # Processed datasets
│
├── src/
│ ├── data/
│ │ ├── init.py
│ │ └── data_loader.py # Data loading and initial preprocessing
│ │
│ ├── models/
│ │ ├── init.py
│ │ ├── rf_model.py # Random Forest implementation
│ │ ├── xgb_model.py # XGBoost implementation
│ │ ├── xgb_tuned_model.py # Tuned XGBoost implementation
│ │ └── lgb_model.py # LightGBM implementation
│ │
│ └── visualization/
│ ├── init.py
│ └── model_evaluation.py # Model performance visualization
│
├── visualization_outputs/ # Output directory for plots and results
├── mlruns/ # MLflow tracking files
├── main.py # Main script to run experiments
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```
## Installation

```bash
# Clone the repository
git clone https://github.com/darlingoscanoa/finance-loan-default-ml-classification-mlflow.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#To Start the MLflow server
 server --host 0.0.0.0 --port 5000

```

## Usage
To run the experiments:
```bash
python main.py
```
This script will load the data, train multiple models, track experiments with MLflow, and generate performance visualizations.

## MLflow Tracking

The project uses MLflow to track:

Model parameters
Performance metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC)

Model artifacts
To view the MLflow UI, open a web browser and navigate to:

http://localhost:5000

## Model Performance
The project includes detailed performance analysis:

-Confusion Matrix visualization

-ROC curves

-Performance metrics comparison

-Cross-validation results

Results are saved in the visualization_outputs directory and can be viewed in the MLflow UI.

## Future Work

While this project focused on the core classification modeling task, future work will explore:

-Extensive feature engineering

-Data balancing techniques

-Advanced model architectures

Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
