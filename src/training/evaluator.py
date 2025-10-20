"""
Model evaluation metrics for anomaly detection
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class ModelEvaluator:
    """Evaluate anomaly detection models"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of anomaly detection
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Precision and Recall at K
        if y_scores is not None:
            for k in [1, 5, 10]:
                p_at_k, r_at_k = self._precision_recall_at_k(y_true, y_scores, k)
                metrics[f'precision_at_{k}'] = p_at_k
                metrics[f'recall_at_{k}'] = r_at_k
            
            # ROC AUC and PR AUC
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                metrics['pr_auc'] = average_precision_score(y_true, y_scores)
            except ValueError as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _precision_recall_at_k(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        k: int
    ) -> tuple:
        """Calculate Precision@K and Recall@K"""
        
        threshold_idx = int(len(y_scores) * (100 - k) / 100)
        sorted_indices = np.argsort(y_scores)
        top_k_indices = sorted_indices[threshold_idx:]
        
        y_pred_at_k = np.zeros_like(y_true)
        y_pred_at_k[top_k_indices] = 1
        
        precision = precision_score(y_true, y_pred_at_k, zero_division=0)
        recall = recall_score(y_true, y_pred_at_k, zero_division=0)
        
        return precision, recall
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix"
    ):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve"
    ):
        """Plot precision-recall curve"""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        baseline = np.mean(y_true)
        plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline: {baseline:.3f}')
        plt.legend()
        
        return plt.gcf()
    
    def print_evaluation_report(self, metrics: Dict[str, Any]):
        """Print formatted evaluation report"""
        
        logger.info("=" * 60)
        logger.info("EVALUATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info("-" * 60)
        
        if 'precision_at_1' in metrics:
            logger.info(f"Precision@1%: {metrics['precision_at_1']:.4f}")
            logger.info(f"Precision@5%: {metrics['precision_at_5']:.4f}")
            logger.info(f"Precision@10%: {metrics['precision_at_10']:.4f}")
            logger.info("-" * 60)
            logger.info(f"Recall@1%: {metrics['recall_at_1']:.4f}")
            logger.info(f"Recall@5%: {metrics['recall_at_5']:.4f}")
            logger.info(f"Recall@10%: {metrics['recall_at_10']:.4f}")
            logger.info("-" * 60)
        
        if 'roc_auc' in metrics:
            logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
            logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
            logger.info("-" * 60)
        
        logger.info("Confusion Matrix:")
        logger.info(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
        logger.info(f"  FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
        logger.info(f"  FPR: {metrics['false_positive_rate']:.4f}")
        logger.info("=" * 60)