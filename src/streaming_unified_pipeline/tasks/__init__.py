"""
Task modules for unified streaming evaluation pipeline.
"""

from .base_task import BaseTask
from .task1_clustering import ClusteringTask
from .task2_summarization import SummarizationTask
from .task3_temporal_qa import TemporalQATask

# Available tasks
AVAILABLE_TASKS = [
    'clustering',
    'summarization', 
    'temporal_qa'
]

# Task class mapping
TASK_CLASSES = {
    'clustering': ClusteringTask,
    'summarization': SummarizationTask,
    'temporal_qa': TemporalQATask
}