"""
Base task class for the unified streaming evaluation pipeline.
All tasks (clustering, summarization, temporal QA) inherit from this base class.
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from core.window_manager import WindowManager
from core.data_loader import UnifiedDataLoader
from core.llm_client import UnifiedLLMClient


class BaseTask(ABC):
    """
    Abstract base class for all tasks in the streaming evaluation pipeline.
    Provides common functionality and defines the interface for task implementations.
    """
    
    def __init__(self,
                 task_name: str,
                 data_file: str,
                 window_manager: WindowManager,
                 llm_client: UnifiedLLMClient,
                 output_dir: str = "results",
                 random_seed: int = 42,
                 max_input_tokens: int = 4000,
                 texts_per_event: int = 1,
                 use_structured_data: bool = False,
                 save_results: bool = True,
                 exclude_fields: list = None,
                 **kwargs):
        """
        Initialize base task.
        
        Args:
            task_name: Name of the task (e.g., "clustering", "summarization", "temporal_qa")
            data_file: Path to data file
            window_manager: WindowManager instance
            llm_client: UnifiedLLMClient instance
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
            max_input_tokens: Maximum input tokens
            texts_per_event: Number of texts to sample per event
            use_structured_data: Whether to include structured data in processing
            exclude_fields: List of structured data fields to exclude (for ablation studies)
            **kwargs: Additional task-specific arguments
        """
        self.task_name = task_name
        self.data_file = data_file
        self.window_manager = window_manager
        self.llm_client = llm_client
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.max_input_tokens = max_input_tokens
        self.texts_per_event = texts_per_event
        self.use_structured_data = use_structured_data
        self.save_results_flag = save_results  # Renamed to avoid conflict with method
        self.exclude_fields = exclude_fields or []
        
        # Initialize data loader
        self.data_loader = UnifiedDataLoader(
            json_file=data_file,
            random_seed=random_seed,
            max_input_tokens=max_input_tokens,
            texts_per_event=texts_per_event,
            exclude_fields=self.exclude_fields
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Task-specific configuration
        self.config = kwargs
        
        # Results storage
        self.results = {
            'task_name': task_name,
            'config': self._get_config_summary(),
            'execution_info': {},
            'windows': [],
            'evaluation_metrics': {},
            'execution_time': 0
        }
        
        logging.info(f"Initialized {task_name} task with {len(self.data_loader.get_all_events())} events")
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for results."""
        return {
            'data_file': self.data_file,
            'window_type': self.window_manager.window_type,
            'window_size': self.window_manager.window_size,
            'stride': self.window_manager.stride,
            'llm_engine': self.llm_client.get_engine_info(),
            'random_seed': self.random_seed,
            'max_input_tokens': self.max_input_tokens,
            'texts_per_event': self.texts_per_event,
            'use_structured_data': self.use_structured_data,
            'exclude_fields': self.exclude_fields,
            **self.config
        }
    
    @abstractmethod
    def process_window(self, window: Dict[str, Any], window_index: int) -> Dict[str, Any]:
        """
        Process a single window.
        
        Args:
            window: Window data from WindowManager
            window_index: Index of the window
            
        Returns:
            Dictionary containing window processing results
        """
        pass
    
    @abstractmethod
    def evaluate_results(self) -> Dict[str, Any]:
        """
        Evaluate overall task results.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def run(self, mode: str = "sliding") -> Dict[str, Any]:
        """
        Run the task with specified mode.
        
        Args:
            mode: Processing mode ("sliding" or "incremental" for clustering)
            
        Returns:
            Complete results dictionary
        """
        start_time = time.time()
        
        logging.info(f"Starting {self.task_name} task in {mode} mode")
        
        # Update execution info
        self.results['execution_info'] = {
            'mode': mode,
            'start_time': datetime.now().isoformat(),
            'data_statistics': self.data_loader.get_statistics()
        }
        
        try:
            if mode == "sliding":
                self._run_sliding_mode()
            elif mode == "incremental":
                self._run_incremental_mode()
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            # Evaluate results
            self.results['evaluation_metrics'] = self.evaluate_results()
            
            # Record execution time
            execution_time = time.time() - start_time
            self.results['execution_time'] = execution_time
            self.results['execution_info']['end_time'] = datetime.now().isoformat()
            
            logging.info(f"{self.task_name} task completed in {execution_time:.2f} seconds")
            
            # Save results
            self.save_results()
            
            return self.results
            
        except Exception as e:
            logging.error(f"Error running {self.task_name} task: {e}")
            raise
    
    def _run_sliding_mode(self):
        """Run task in sliding window mode."""
        events = self.data_loader.get_all_events()
        windows = self.window_manager.create_windows(events)
        
        logging.info(f"Processing {len(windows)} windows in sliding mode")
        
        for i, window in enumerate(windows):
            logging.info(f"Processing window {i+1}/{len(windows)}")
            
            try:
                window_result = self.process_window(window, i)
                window_result['window_index'] = i
                window_result['window_info'] = {
                    'start_date': window.get('start_date'),
                    'end_date': window.get('end_date'),
                    'event_count': len(window.get('events', []))
                }
                
                self.results['windows'].append(window_result)
                
            except Exception as e:
                logging.error(f"Error processing window {i}: {e}")
                # Add error info to results
                error_result = {
                    'window_index': i,
                    'error': str(e),
                    'window_info': {
                        'start_date': window.get('start_date'),
                        'end_date': window.get('end_date'),
                        'event_count': len(window.get('events', []))
                    }
                }
                self.results['windows'].append(error_result)
    
    def _run_incremental_mode(self):
        """
        Run task in incremental mode.
        Base implementation - can be overridden by tasks that support incremental mode.
        """
        if self.task_name != "clustering":
            raise NotImplementedError(f"Incremental mode not supported for {self.task_name} task")
        
        # This will be implemented in the clustering task
        raise NotImplementedError("Incremental mode must be implemented by specific task classes")
    
    def save_results(self, filename: Optional[str] = None):
        """
        Save results to JSON file.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{self.task_name}_summary.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Reorder results to put evaluation_metrics first
            ordered_results = {}
            if 'evaluation_metrics' in self.results:
                ordered_results['evaluation_metrics'] = self.results['evaluation_metrics']
            
            # Add remaining keys in original order
            for key in self.results:
                if key != 'evaluation_metrics':
                    ordered_results[key] = self.results[key]
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(ordered_results, f, indent=2, ensure_ascii=False, default=str)
            
            logging.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load results from JSON file.
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded results dictionary
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logging.info(f"Results loaded from {filepath}")
            return results
            
        except Exception as e:
            logging.error(f"Error loading results: {e}")
            raise
    
    def get_window_summary(self) -> Dict[str, Any]:
        """Get summary of processed windows."""
        if not self.results['windows']:
            return {'total_windows': 0}
        
        successful_windows = [w for w in self.results['windows'] if 'error' not in w]
        failed_windows = [w for w in self.results['windows'] if 'error' in w]
        
        return {
            'total_windows': len(self.results['windows']),
            'successful_windows': len(successful_windows),
            'failed_windows': len(failed_windows),
            'success_rate': len(successful_windows) / len(self.results['windows']) if self.results['windows'] else 0
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation metrics."""
        return self.results.get('evaluation_metrics', {})
    
    def print_summary(self):
        """Print task execution summary."""
        print(f"\n{'='*50}")
        print(f"Task: {self.task_name.upper()}")
        print(f"{'='*50}")
        
        # Execution info
        exec_info = self.results.get('execution_info', {})
        print(f"Mode: {exec_info.get('mode', 'N/A')}")
        print(f"Execution time: {self.results.get('execution_time', 0):.2f} seconds")
        
        # Data statistics
        data_stats = exec_info.get('data_statistics', {})
        print(f"Total events: {data_stats.get('total_events', 0)}")
        print(f"Total topics: {data_stats.get('total_topics', 0)}")
        
        # Task-specific summaries
        if self.task_name == 'summarization':
            # Summarization task stores results directly
            total_windows = self.results.get('total_windows', 0)
            total_summaries = self.results.get('total_summaries', 0)
            print(f"Total windows: {total_windows}")
            print(f"Total summaries: {total_summaries}")
            if total_windows > 0:
                success_rate = total_summaries / total_windows
                print(f"Success rate: {success_rate:.2%}")
            else:
                print(f"Success rate: 0.00%")
        else:
            # Other tasks use window summary
            window_summary = self.get_window_summary()
            print(f"Total windows: {window_summary.get('total_windows', 0)}")
            
            # Show task-specific success rate
            if self.task_name == 'temporal_qa' and 'accuracy' in self.results:
                print(f"Success rate: {self.results['accuracy']:.2%}")
            else:
                print(f"Success rate: {window_summary.get('success_rate', 0):.2%}")
        
        # Evaluation metrics
        metrics = self.get_metrics_summary()
        if metrics:
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        print(f"{'='*50}\n")


# Available task names
AVAILABLE_TASKS = ["clustering", "summarization", "temporal_qa"]
