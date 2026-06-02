"""
Temporal QA evaluation metrics: Accuracy by question type
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict, Counter


class QAEvaluator:
    """
    Temporal QA evaluation using accuracy metrics.
    """
    
    def __init__(self):
        self.name = "qa_evaluator"
    
    def evaluate(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate temporal QA results.
        
        Args:
            qa_results: List of QA result dictionaries
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not qa_results:
            return {
                'error': 'No QA results available for evaluation'
            }
        
        try:
            # Calculate metrics
            total_questions = len(qa_results)
            correct_answers = sum(1 for r in qa_results if r.get('is_correct', False))
            overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
            
            # Accuracy by question type (if available)
            type_accuracies, type_counts = self._calculate_type_accuracies(qa_results)
            
            # Answer distribution
            answer_distribution = self._calculate_answer_distribution(qa_results)
            
            # Response statistics
            response_stats = self._calculate_response_statistics(qa_results)
            
            return {
                'overall_accuracy': overall_accuracy,
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'question_type_accuracies': type_accuracies,
                'question_type_counts': type_counts,
                'answer_distribution': answer_distribution,
                'response_statistics': response_stats,
                'structured_data_usage_rate': sum(1 for r in qa_results if r.get('structured_data_used', False)) / len(qa_results)
            }
            
        except Exception as e:
            logging.error(f"Error evaluating temporal QA results: {e}")
            return {'error': str(e)}
    
    def _calculate_type_accuracies(self, qa_results: List[Dict[str, Any]]) -> tuple:
        """Calculate accuracy by question type."""
        type_correct = defaultdict(int)
        type_total = defaultdict(int)
        
        for result in qa_results:
            q_type = result.get('question_type', 'unknown')
            is_correct = result.get('is_correct', False)
            
            type_total[q_type] += 1
            if is_correct:
                type_correct[q_type] += 1
        
        # Calculate accuracies
        type_accuracies = {}
        for q_type in type_total:
            type_accuracies[q_type] = type_correct[q_type] / type_total[q_type] if type_total[q_type] > 0 else 0.0
        
        type_counts = dict(type_total)
        
        return type_accuracies, type_counts
    
    def _calculate_answer_distribution(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate distribution of answers."""
        predicted_answers = [r.get('predicted_answer', 'N/A') for r in qa_results]
        true_answers = [r.get('correct_answer', 'N/A') for r in qa_results]
        
        predicted_dist = dict(Counter(predicted_answers))
        true_dist = dict(Counter(true_answers))
        
        return {
            'predicted_answer_distribution': predicted_dist,
            'true_answer_distribution': true_dist,
            'unique_predicted_answers': len(set(predicted_answers)),
            'unique_true_answers': len(set(true_answers))
        }
    
    def _calculate_response_statistics(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate response statistics."""
        response_lengths = []
        confidence_scores = []
        
        for result in qa_results:
            # Response length
            response = result.get('predicted_answer', '')
            if isinstance(response, str):
                response_lengths.append(len(response.split()))
            
            # Confidence score (if available)
            confidence = result.get('confidence', None)
            if confidence is not None:
                confidence_scores.append(confidence)
        
        stats = {}
        
        if response_lengths:
            stats['avg_response_length'] = sum(response_lengths) / len(response_lengths)
            stats['min_response_length'] = min(response_lengths)
            stats['max_response_length'] = max(response_lengths)
        
        if confidence_scores:
            stats['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
            stats['min_confidence'] = min(confidence_scores)
            stats['max_confidence'] = max(confidence_scores)
        
        return stats
    
    def evaluate_by_temporal_aspect(self, qa_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate QA results by temporal aspects."""
        temporal_metrics = {}
        
        # Group by temporal question types
        temporal_types = ['before', 'after', 'during', 'when', 'temporal_order', 'duration']
        
        for t_type in temporal_types:
            relevant_results = [
                r for r in qa_results 
                if t_type in r.get('question', '').lower() or t_type in r.get('question_type', '').lower()
            ]
            
            if relevant_results:
                correct = sum(1 for r in relevant_results if r.get('is_correct', False))
                total = len(relevant_results)
                accuracy = correct / total if total > 0 else 0.0
                
                temporal_metrics[f'{t_type}_accuracy'] = accuracy
                temporal_metrics[f'{t_type}_count'] = total
        
        return temporal_metrics
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of QA metrics."""
        return {
            'overall_accuracy': 'Overall accuracy across all QA pairs',
            'total_questions': 'Total number of questions evaluated',
            'correct_answers': 'Number of correctly answered questions',
            'question_type_accuracies': 'Accuracy broken down by question type',
            'question_type_counts': 'Number of questions per type',
            'answer_distribution': 'Distribution of predicted vs true answers',
            'response_statistics': 'Statistics about response characteristics',
            'structured_data_usage_rate': 'Rate of structured data usage in evaluation'
        }
