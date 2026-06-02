"""
Evaluation modules for the unified streaming pipeline.
Contains specialized evaluators for each task type.
"""

from .clustering_metrics import ClusteringEvaluator
from .summarization_metrics import SummaryEvaluator as SummarizationEvaluator
from .qa_metrics import QAEvaluator

# Evaluator registry
EVALUATORS = {
    'clustering': ClusteringEvaluator,
    'summarization': SummarizationEvaluator,
    'temporal_qa': QAEvaluator
}

__all__ = [
    'ClusteringEvaluator',
    'SummarizationEvaluator', 
    'QAEvaluator',
    'EVALUATORS'
]
