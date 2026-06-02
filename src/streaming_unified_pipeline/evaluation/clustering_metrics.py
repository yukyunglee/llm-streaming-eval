"""
Clustering evaluation metrics: B³, NMI, AMI
"""

import logging
import numpy as np
from typing import List, Dict, Any


class ClusteringEvaluator:
    """
    Clustering evaluation using standard metrics.
    """
    
    def __init__(self):
        self.name = "clustering_evaluator"
    
    def evaluate(self, predictions: List[int], ground_truth: List[int]) -> Dict[str, Any]:
        """
        Evaluate clustering results using standard metrics.
        
        Args:
            predictions: Predicted cluster labels
            ground_truth: True cluster labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not predictions or not ground_truth:
            return {
                'error': 'No predictions or ground truth available for evaluation'
            }
        
        if len(predictions) != len(ground_truth):
            return {
                'error': f'Length mismatch: predictions={len(predictions)}, ground_truth={len(ground_truth)}'
            }
        
        try:
            # Import clustering metrics
            from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
            from sklearn.metrics.cluster import contingency_matrix
            
            # Convert to numpy arrays
            predictions = np.array(predictions)
            ground_truth = np.array(ground_truth)
            
            # Calculate metrics
            ami_score = adjusted_mutual_info_score(ground_truth, predictions)
            nmi_score = normalized_mutual_info_score(ground_truth, predictions)
            b3_f1 = self._calculate_b3_f1(predictions, ground_truth)
            
            # Cluster statistics
            num_predicted_clusters = len(np.unique(predictions))
            num_true_clusters = len(np.unique(ground_truth))
            
            return {
                'ami': float(ami_score),
                'nmi': float(nmi_score),
                'b3_f1': float(b3_f1),
                'num_predicted_clusters': int(num_predicted_clusters),
                'num_true_clusters': int(num_true_clusters),
                'num_documents': len(predictions)
            }
            
        except ImportError:
            logging.error("sklearn required for clustering evaluation")
            return {'error': 'sklearn not available for clustering evaluation'}
        except Exception as e:
            logging.error(f"Error evaluating clustering results: {e}")
            return {'error': str(e)}
    
    def _calculate_b3_f1(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate B³ F1 score.
        
        Args:
            predictions: Predicted cluster labels
            ground_truth: True cluster labels
            
        Returns:
            B³ F1 score
        """
        try:
            # Calculate B³ precision and recall
            precision_scores = []
            recall_scores = []
            
            for i in range(len(predictions)):
                # Find items in same predicted cluster
                same_pred_cluster = (predictions == predictions[i])
                # Find items in same true cluster
                same_true_cluster = (ground_truth == ground_truth[i])
                
                # B³ precision: how many items in predicted cluster are in same true cluster
                if np.sum(same_pred_cluster) > 0:
                    precision = np.sum(same_pred_cluster & same_true_cluster) / np.sum(same_pred_cluster)
                    precision_scores.append(precision)
                
                # B³ recall: how many items in true cluster are in same predicted cluster
                if np.sum(same_true_cluster) > 0:
                    recall = np.sum(same_pred_cluster & same_true_cluster) / np.sum(same_true_cluster)
                    recall_scores.append(recall)
            
            # Average precision and recall
            avg_precision = np.mean(precision_scores) if precision_scores else 0.0
            avg_recall = np.mean(recall_scores) if recall_scores else 0.0
            
            # B³ F1 score
            if avg_precision + avg_recall > 0:
                b3_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            else:
                b3_f1 = 0.0
                
            return b3_f1
            
        except Exception as e:
            logging.error(f"Error calculating B³ F1 score: {e}")
            return 0.0
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of clustering metrics."""
        return {
            'ami': 'Adjusted Mutual Information - measures agreement between clusterings, adjusted for chance',
            'nmi': 'Normalized Mutual Information - normalized measure of clustering similarity', 
            'b3_f1': 'B³ F1 Score - element-based clustering evaluation metric',
            'num_predicted_clusters': 'Number of clusters in predictions',
            'num_true_clusters': 'Number of clusters in ground truth',
            'num_documents': 'Total number of documents evaluated'
        }


def compute_all_metrics(ground_truth: List[int], predictions: List[int]) -> Dict[str, Any]:
    """
    Compute all clustering metrics (Task-1 original format).
    
    Args:
        ground_truth: True cluster labels
        predictions: Predicted cluster labels
        
    Returns:
        Dictionary containing all clustering metrics
    """
    evaluator = ClusteringEvaluator()
    return evaluator.evaluate(predictions, ground_truth)


def print_metrics(metrics: Dict[str, Any], title: str = "Clustering Evaluation"):
    """
    Print clustering metrics in a formatted way (Task-1 original format).
    
    Args:
        metrics: Dictionary containing clustering metrics
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if 'error' in metrics:
        print(f"ERROR: {metrics['error']}")
        return
    
    print(f"AMI (Adjusted Mutual Information): {metrics.get('ami', 0.0):.4f}")
    print(f"NMI (Normalized Mutual Information): {metrics.get('nmi', 0.0):.4f}")  
    print(f"B³ F1 Score: {metrics.get('b3_f1', 0.0):.4f}")
    print(f"Number of Predicted Clusters: {metrics.get('num_predicted_clusters', 0)}")
    print(f"Number of True Clusters: {metrics.get('num_true_clusters', 0)}")
    print(f"Number of Documents: {metrics.get('num_documents', 0)}")
    print(f"{'='*60}\n")
