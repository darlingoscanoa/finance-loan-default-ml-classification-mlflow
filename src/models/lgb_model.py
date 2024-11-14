# lgb_model.py
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train and evaluate LightGBM model."""
    # Initialize model with some hyperparameters
    model = lgb.LGBMClassifier(
        objective='binary',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
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
    plt.title('LightGBM - Feature Importance')
    plt.tight_layout()
    plt.savefig('lgb_feature_importance.png')
    mlflow.log_artifact('lgb_feature_importance.png')
    plt.close()

    return model, metrics