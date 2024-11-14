# AI-Powered Credit Risk Assessment: MLflow-Tracked Classification Models for Loan Default Prediction

## Overview
This project implements multiple classification models to predict loan defaults, leveraging MLflow for experiment tracking and model management. The solution demonstrates production-grade machine learning practices in financial services, incorporating comprehensive EDA, feature engineering, and model performance analysis.

## Project Structure
```
loan_default_prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed datasets
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Data loading and initial preprocessing
│   │   └── data_preprocessor.py# Feature engineering and data preparation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py # Feature creation and transformation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_base.py      # Base model class
│   │   ├── random_forest.py   # Random Forest implementation
│   │   └── xgboost_model.py   # XGBoost implementation
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_plots.py       # EDA visualization functions
│   │   └── model_evaluation.py # Model performance visualization
│   │
│   └── utils/
│       ├── __init__.py
│       └── metrics.py         # Custom metrics and evaluation functions
│
├── notebooks/
│   ├── 1.0-eda.ipynb         # Exploratory Data Analysis
│   └── 2.0-modeling.ipynb    # Model Development and Evaluation
│
├── mlruns/                    # MLflow tracking files
├── config.yaml               # Configuration parameters
├── requirements.txt          # Project dependencies
├── setup.py                  # Package setup file
└── README.md                 # Project documentation
```

## Key Features
- Comprehensive EDA with visualizations of key financial indicators
- Multiple classification models (Random Forest, XGBoost) with hyperparameter tuning
- MLflow integration for experiment tracking and model versioning
- Advanced feature engineering specific to financial data
- Detailed model performance analysis with custom visualization
- Production-ready code structure with proper OOP implementation

## Installation
```bash
# Clone the repository
git clone https://github.com/darlingoscanoa/loan-default-prediction.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
from src.data.data_loader import DataLoader
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

# Load and preprocess data
loader = DataLoader()
X_train, X_test, y_train, y_test = loader.get_train_test_split()

# Train models
rf_model = RandomForestModel()
rf_model.train(X_train, y_train)

xgb_model = XGBoostModel()
xgb_model.train(X_train, y_train)

# Evaluate models
rf_metrics = rf_model.evaluate(X_test, y_test)
xgb_metrics = xgb_model.evaluate(X_test, y_test)
```

## MLflow Tracking
The project uses MLflow to track:
- Model parameters
- Performance metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC)
- Feature importance
- Model artifacts

To view the MLflow UI:
```bash
mlflow ui
```

## Model Performance
The project includes detailed performance analysis:
- Confusion Matrix visualization
- ROC curves
- Feature importance plots
- Performance metrics comparison
- Cross-validation results

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.