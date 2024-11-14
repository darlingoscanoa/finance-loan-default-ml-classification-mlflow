# rf_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model."""
    # Initialize model with some hyperparameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Log parameters and metrics
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    
    # Plot and log feature importance
    plt.figure(figsize=(10, 6))
    feat_imp = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feat_imp.head(20))
    plt.title('Random Forest - Feature Importance')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png')
    mlflow.log_artifact('rf_feature_importance.png')
    plt.close()
    
    return model, metrics