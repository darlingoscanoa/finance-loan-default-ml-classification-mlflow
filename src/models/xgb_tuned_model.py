# src/models/xgb_tuned.py
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_xgboost_tuned(X_train, X_test, y_train, y_test):
    """Train and evaluate XGBoost model with hyperparameter tuning."""
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    # Initialize the XGBoost model
    model = xgb.XGBClassifier(random_state=42)

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred_tuned = best_model.predict(X_test)
    y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'f1': f1_score(y_test, y_pred_tuned),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_tuned)
    }

    # Log parameters and metrics
    mlflow.log_params(best_model.get_params())
    mlflow.log_metrics(metrics)

    # Plot and log feature importance
    plt.figure(figsize=(10, 6))
    feat_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    sns.barplot(x='importance', y='feature', data=feat_imp.head(20))
    plt.title('XGBoost Tuned - Feature Importance')
    plt.tight_layout()
    plt.savefig('xgb_tuned_feature_importance.png')
    mlflow.log_artifact('xgb_tuned_feature_importance.png')
    plt.close()

    return best_model, metrics