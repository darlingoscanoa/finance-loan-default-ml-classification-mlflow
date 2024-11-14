import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

class ModelEvaluationPlotter:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        labels = [f"{count}\n{percent:.1%}" for count, percent in zip(cm.flatten(), cm_percent.flatten())]
        labels = np.asarray(labels).reshape(2, 2)

        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        label_positions = [(0.3, 0.3), (1.3, 0.3), (0.3, 1.3), (1.3, 1.3)]
        label_texts = ['TN', 'FP', 'FN', 'TP']
        for pos, text in zip(label_positions, label_texts):
            plt.text(pos[0], pos[1], text, ha='center', va='center', fontweight='bold', color='gray', fontsize=12)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        plt.savefig(output_path)
        plt.close()

    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")

        output_path = os.path.join(self.output_dir, f'{model_name}_roc_curve.png')
        plt.savefig(output_path)
        plt.close()