"""Model evaluation metrics and utilities."""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score,
    recall_score, accuracy_score, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Evaluates fraud detection models using multiple metrics.
    
    Focuses on metrics that balance fraud detection with minimizing
    false positives that could reject legitimate customers.
    """
    
    def __init__(self, average: str = 'binary'):
        """
        Initialize evaluator.
        
        Args:
            average: Averaging method for multi-class metrics
        """
        self.average = average
        self.evaluation_results: Dict[str, Any] = {}
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (required for ROC-AUC)
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC (most important for imbalanced classification)
        if y_pred_proba is not None:
            results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            results['roc_auc'] = 0.0
        
        # Confusion matrix-based metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results['true_negatives'] = int(tn)
        results['false_positives'] = int(fp)
        results['false_negatives'] = int(fn)
        results['true_positives'] = int(tp)
        
        # Business metrics
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results['false_positive_rate'] = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        results['fraud_detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        self.evaluation_results = results
        return results
    
    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred))
    
    def print_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Print confusion matrix with interpretation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(f"True Negatives (TN):   {tn:,}  - Correctly identified legitimate")
        print(f"False Positives (FP):  {fp:,}  - Legitimate rejected as fraud")
        print(f"False Negatives (FN):  {fn:,}  - Fraud accepted as legitimate")
        print(f"True Positives (TP):   {tp:,}  - Correctly identified fraud")
        print("="*60)
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud']
        )
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size
            
        Returns:
            Matplotlib figure object
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Fraud Detection Rate)')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        return fig
    
    def get_summary_report(self) -> pd.DataFrame:
        """
        Get summary report of evaluation metrics.
        
        Returns:
            DataFrame with metric summary
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Call evaluate() first.")
        
        summary_data = {
            'Metric': [
                'Accuracy',
                'Precision',
                'Recall (Fraud Detection)',
                'F1-Score',
                'ROC-AUC',
                'Specificity',
                'False Positive Rate'
            ],
            'Value': [
                f"{self.evaluation_results['accuracy']:.4f}",
                f"{self.evaluation_results['precision']:.4f}",
                f"{self.evaluation_results['recall']:.4f}",
                f"{self.evaluation_results['f1']:.4f}",
                f"{self.evaluation_results['roc_auc']:.4f}",
                f"{self.evaluation_results['specificity']:.4f}",
                f"{self.evaluation_results['false_positive_rate']:.4f}"
            ],
            'Interpretation': [
                'Overall correctness',
                'Precision of fraud detection',
                'Ability to catch actual fraud',
                'Balance of precision and recall',
                'Discriminative ability',
                'Ability to identify legitimate apps',
                'False alarm rate for legitimate apps'
            ]
        }
        
        return pd.DataFrame(summary_data)
