"""
Task-1 Clustering evaluation metrics.
Original implementation from Task-1/evaluation.py
"""

from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score
from typing import List, Dict
import logging


def b3_score(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Calculate B³ F1 score for clustering evaluation.
    Original implementation from Task-1.
    """
    n = len(true_labels)
    precision_list = []
    recall_list = []

    for i in range(n):
        true_cluster = {j for j in range(n) if true_labels[j] == true_labels[i]}
        pred_cluster = {j for j in range(n) if pred_labels[j] == pred_labels[i]}
        intersection = true_cluster & pred_cluster

        precision = len(intersection) / len(pred_cluster) if pred_cluster else 0
        recall = len(intersection) / len(true_cluster) if true_cluster else 0

        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = sum(precision_list) / n
    avg_recall = sum(recall_list) / n
    if avg_precision + avg_recall == 0:
        return 0.0

    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    return f1


def compute_all_metrics(true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    """
    Compute all clustering metrics.
    Original implementation from Task-1.
    """
    try:
        return {
            "B3_F1": b3_score(true_labels, pred_labels),
            "NMI": normalized_mutual_info_score(true_labels, pred_labels),
            "AMI": adjusted_mutual_info_score(true_labels, pred_labels)
        }
    except Exception as e:
        logging.error(f"Error computing clustering metrics: {e}")
        return {
            "B3_F1": 0.0,
            "NMI": 0.0,
            "AMI": 0.0
        }


def print_metrics(metrics: Dict[str, float], title: str = "") -> None:
    """
    Print clustering metrics in original format.
    """
    if title:
        print(f"\n=== {title} ===")
    print(f"B³ F1: {metrics['B3_F1']:.4f}")
    print(f"NMI   : {metrics['NMI']:.4f}")
    print(f"AMI   : {metrics['AMI']:.4f}")


class Task1ClusteringEvaluator:
    """
    Task-1 clustering evaluator with original metrics.
    """
    
    def __init__(self):
        pass
    
    def evaluate(self, true_labels: List[int], pred_labels: List[int], 
                 print_results: bool = True, title: str = "") -> Dict[str, float]:
        """
        Evaluate clustering results with original Task-1 metrics.
        
        Args:
            true_labels: Ground truth cluster labels
            pred_labels: Predicted cluster labels  
            print_results: Whether to print results
            title: Title for printing
            
        Returns:
            Dictionary with B3_F1, NMI, AMI scores
        """
        metrics = compute_all_metrics(true_labels, pred_labels)
        
        if print_results:
            print_metrics(metrics, title)
            
        return metrics
