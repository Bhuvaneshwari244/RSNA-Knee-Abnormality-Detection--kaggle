"""
Evaluation metrics for the competition
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict


def compute_auc_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list
) -> Dict[str, float]:
    """
    Compute AUC ROC for each target and macro average
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        y_pred: Predicted probabilities (N, num_classes)
        target_names: List of target names
        
    Returns:
        Dictionary with AUC scores for each target and macro average
    """
    scores = {}
    
    # Compute AUC for each target
    aucs = []
    for i, target_name in enumerate(target_names):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            scores[target_name] = auc
            aucs.append(auc)
        except ValueError:
            # Handle case where only one class is present
            scores[target_name] = 0.5
            aucs.append(0.5)
    
    # Compute macro average (competition metric)
    scores['macro_auc'] = np.mean(aucs)
    
    return scores


def format_scores(scores: Dict[str, float]) -> str:
    """
    Format scores for printing
    
    Args:
        scores: Dictionary of scores
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in scores.items():
        if key == 'macro_auc':
            lines.append(f"\n{'='*40}")
            lines.append(f"{key.upper()}: {value:.4f}")
            lines.append(f"{'='*40}")
        else:
            lines.append(f"{key:20s}: {value:.4f}")
    
    return "\n".join(lines)
