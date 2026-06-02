#!/usr/bin/env python3
"""
Unified Streaming Evaluation Pipeline
Main CLI entry point for running streaming tasks with configurable parameters.
"""

import os
import sys

# Set cache directories BEFORE any imports
# This must happen before transformers/huggingface_hub/vllm are imported
if os.environ.get('HF_HOME'):
    os.environ['TRANSFORMERS_CACHE'] = os.environ['HF_HOME']
    os.environ['HF_HUB_CACHE'] = os.path.join(os.environ['HF_HOME'], 'hub')
    os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'datasets')

# vLLM uses VLLM_CACHE_ROOT (not VLLM_CACHE_DIR)
if os.environ.get('VLLM_CACHE_ROOT'):
    pass  # Already set correctly from shell
elif os.environ.get('XDG_CACHE_HOME'):
    os.environ['VLLM_CACHE_ROOT'] = os.path.join(os.environ['XDG_CACHE_HOME'], 'vllm')

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.window_manager import WindowManager
from core.llm_client import UnifiedLLMClient
from tasks import AVAILABLE_TASKS, TASK_CLASSES


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def create_output_directory(output_dir: str) -> str:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_arguments(args) -> bool:
    """Validate command line arguments."""
    # Check if task is valid
    if args.task not in AVAILABLE_TASKS:
        print(f"Error: Invalid task '{args.task}'. Available tasks: {', '.join(AVAILABLE_TASKS)}")
        return False
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file}' not found.")
        return False
    
    # Check mode validity
    if args.mode == 'incremental' and args.task != 'clustering':
        print(f"Error: Incremental mode is only supported for clustering task.")
        return False
    
    # Check window size
    if args.window_size <= 0:
        print(f"Error: Window size must be positive, got {args.window_size}")
        return False
    
    # Check LLM engine
    valid_engines = ['together', 'vllm', 'openai', 'mock']
    if args.llm_engine not in valid_engines:
        print(f"Error: Invalid LLM engine '{args.llm_engine}'. Available: {', '.join(valid_engines)}")
        return False
    
    return True


def create_llm_client(engine: str, model_name: str = None, model_path: str = None, **kwargs) -> UnifiedLLMClient:
    """Create LLM client based on engine type."""
    if engine == 'mock':
        return UnifiedLLMClient(engine='mock')
    
    # Default models for each engine
    default_models = {
        'together': 'meta-llama/Llama-2-7b-chat-hf',
        'vllm': 'meta-llama/Llama-2-7b-chat-hf', 
        'openai': 'gpt-3.5-turbo'
    }
    
    # Set model_name for API engines or model_path for vLLM
    if engine == 'vllm':
        if model_path is None:
            model_path = model_name or default_models[engine]
        return UnifiedLLMClient(engine=engine, model_path=model_path, **kwargs)
    else:
        if model_name is None:
            model_name = default_models.get(engine, 'meta-llama/Llama-2-7b-chat-hf')
        return UnifiedLLMClient(engine=engine, model_name=model_name, **kwargs)


def run_task(args) -> Dict[str, Any]:
    """Run the specified task with given arguments."""
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Create LLM client
    logging.info(f"Initializing LLM client: {args.llm_engine}")
    
    # Prepare engine-specific kwargs
    engine_kwargs = {}
    if args.llm_engine == 'vllm':
        if hasattr(args, 'tensor_parallel_size'):
            engine_kwargs['tensor_parallel_size'] = args.tensor_parallel_size
        if hasattr(args, 'gpu_memory_utilization'):
            engine_kwargs['gpu_memory_utilization'] = args.gpu_memory_utilization
        if hasattr(args, 'hf_cache_dir') and args.hf_cache_dir:
            engine_kwargs['download_dir'] = args.hf_cache_dir
    
    llm_client = create_llm_client(
        engine=args.llm_engine,
        model_name=args.model,
        model_path=args.model if args.llm_engine == 'vllm' else None,
        base_url=args.base_url,
        api_key=args.api_key,
        **engine_kwargs
    )
    
    # Create window manager  
    logging.info(f"Creating window manager: size={args.window_size}, step={args.step_size}, mode={args.mode}")
    
    # Map mode to window_type
    if args.mode == 'sliding':
        window_type = 'date'  # sliding mode uses date-based windows
    else:
        window_type = args.mode  # incremental stays as is
        
    window_manager = WindowManager(
        window_size=args.window_size,
        stride=args.step_size,  # Fixed: step_size -> stride
        window_type=window_type
    )
    
    # Get task class
    task_class = TASK_CLASSES[args.task]
    
    # Create task instance with all parameters
    logging.info(f"Initializing {args.task} task")
    # Common task parameters
    task_kwargs = {
        'data_file': args.data_file,
        'window_manager': window_manager,
        'llm_client': llm_client,
        'output_dir': args.output_dir,
        'random_seed': args.random_seed,
        'max_input_tokens': args.max_input_tokens,
        'texts_per_event': args.texts_per_event,
        'use_structured_data': args.use_structured_data,
        'save_results': args.save_results,
        'exclude_fields': args.exclude_fields
    }
    
    # Add task-specific parameters
    if args.task == 'clustering':
        task_kwargs.update({
            'clustering_method': getattr(args, 'clustering_method', 'bertopic'),
            'max_clusters_per_window': getattr(args, 'max_clusters', 10),
            'min_cluster_size': getattr(args, 'min_cluster_size', 2),
            'balance_clusters': getattr(args, 'balance_clusters', False),
            'batch_size': getattr(args, 'batch_size', 1)
        })
    elif args.task == 'summarization':
        task_kwargs.update({
            'summary_max_tokens': getattr(args, 'summary_max_tokens', 150),
            'temperature': getattr(args, 'temperature', 0.0),  # Changed to 0.0 for deterministic results
            'label_type': getattr(args, 'label_type', 'concat'),
            'batch_size': getattr(args, 'batch_size', 1)
        })
    elif args.task == 'temporal_qa':
        task_kwargs.update({
            'qa_max_tokens': getattr(args, 'qa_max_tokens', 10),
            'temperature': getattr(args, 'temperature', 0.0),  # Changed to 0.0 for deterministic results
            'max_qa_pairs_per_window': getattr(args, 'max_qa_pairs', 10),
            'window_mode': getattr(args, 'window_mode', 'topic-mix'),
            'batch_size': getattr(args, 'batch_size', 1)
        })
    
    # Create and run task
    task = task_class(**task_kwargs)
    
    # Run task
    logging.info(f"Starting {args.task} task execution")
    results = task.run()
    
    # Print results
    task.print_summary()
    
    # Note: Evaluation is skipped during generation to avoid GPU conflicts with vLLM
    # Run evaluate_existing_summaries.py separately after generation completes
    logging.info("Skipping evaluation (run evaluate_existing_summaries.py separately to compute metrics)")
    
    # Results are already saved by BaseTask.save_results()
    # No need for additional saving here to avoid duplication
    
    return results


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Streaming Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run clustering with sliding window
  python unified_pipeline.py --task clustering --data-file data.json --window-size 5

  # Run summarization with structured data
  python unified_pipeline.py --task summarization --data-file data.json --use-structured-data
  
  # Run summarization with abstract labels (GPT-4o)
  python unified_pipeline.py --task summarization --data-file data.json --label-type abstract

  # Run temporal QA with vLLM engine
  python unified_pipeline.py --task temporal_qa --data-file qa_data.json --llm-engine vllm

  # Run clustering in incremental mode with custom parameters
  python unified_pipeline.py --task clustering --mode incremental --window-size 3 --step-size 1
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--task', 
        choices=AVAILABLE_TASKS,
        required=True,
        help='Task to run'
    )
    
    parser.add_argument(
        '--data-file',
        required=True,
        help='Path to input data file (JSON format)'
    )
    
    # Window parameters
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Size of sliding window (default: 5)'
    )
    
    parser.add_argument(
        '--step-size',
        type=int,
        default=1,
        help='Step size for sliding window (default: 1)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['sliding', 'incremental'],
        default='sliding',
        help='Processing mode (default: sliding). Incremental only for clustering.'
    )
    
    # LLM parameters
    parser.add_argument(
        '--llm-engine',
        choices=['together', 'vllm', 'openai', 'mock'],
        default='mock',
        help='LLM engine to use (default: mock)'
    )
    
    parser.add_argument(
        '--model',
        help='LLM model name (uses engine default if not specified)'
    )
    
    parser.add_argument(
        '--base-url',
        help='Base URL for vLLM or custom API endpoints'
    )
    
    parser.add_argument(
        '--api-key',
        help='API key for LLM services (or set env variables)'
    )
    
    # vLLM specific parameters
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallel (vLLM only, default: 1)'
    )
    
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.9,
        help='GPU memory utilization ratio (vLLM only, default: 0.9)'
    )
    
    parser.add_argument(
        '--hf-cache-dir',
        type=str,
        help='Hugging Face cache directory path (vLLM only, defaults to system cache)'
    )
    
    # Data processing parameters
    parser.add_argument(
        '--max-input-tokens',
        type=int,
        default=4000,
        help='Maximum input tokens (default: 4000)'
    )
    
    parser.add_argument(
        '--texts-per-event',
        type=int,
        default=1,
        help='Number of texts to sample per event (default: 1)'
    )
    
    parser.add_argument(
        '--use-structured-data',
        action='store_true',
        help='Include structured data in processing'
    )
    
    parser.add_argument(
        '--exclude-fields',
        type=str,
        nargs='+',
        default=[],
        help='Structured data fields to exclude (for ablation). Options: People, Location, Result, "Event Attributes"'
    )
    
    # Task-specific parameters
    parser.add_argument(
        '--clustering-method',
        choices=['bertopic', 'llm'],
        default='bertopic',
        help='Clustering method (for clustering task)'
    )
    
    parser.add_argument(
        '--max-clusters',
        type=int,
        default=10,
        help='Maximum clusters per window (for clustering task)'
    )
    
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=2,
        help='Minimum cluster size (for clustering task)'
    )
    
    parser.add_argument(
        '--balance-clusters',
        action='store_true',
        help='Balance initial clusters in incremental mode (for clustering task)'
    )
    
    parser.add_argument(
        '--window-mode',
        choices=['topic', 'topic-mix'],
        default='topic-mix',
        help='Window mode: topic (per-topic) or topic-mix (mixed) for temporal QA task'
    )
    
    parser.add_argument(
        '--summary-max-tokens',
        type=int,
        default=150,
        help='Maximum tokens for summaries (for summarization task)'
    )
    
    parser.add_argument(
        '--label-type',
        type=str,
        default='concat',
        choices=['concat', 'abstract'],
        help='Label type for summarization: concat (default) or abstract (GPT-4o)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of windows to process in batch (1=sequential, >1=batch processing)'
    )
    
    parser.add_argument(
        '--qa-max-tokens',
        type=int,
        default=10,
        help='Maximum tokens for QA responses (for temporal QA task)'
    )
    
    parser.add_argument(
        '--max-qa-pairs',
        type=int,
        default=10,
        help='Maximum QA pairs per window (for temporal QA task)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM temperature (default: 0.0 for deterministic results)'
    )
    
    # Output parameters
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detailed results to JSON file'
    )
    
    # Misc parameters
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        help='Log file path (logs to console if not specified)'
    )
    
    return parser


def main():
    """Main function."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Print configuration
    logging.info("="*60)
    logging.info("UNIFIED STREAMING EVALUATION PIPELINE")
    logging.info("="*60)
    logging.info(f"Task: {args.task}")
    logging.info(f"Data file: {args.data_file}")
    logging.info(f"Window size: {args.window_size}")
    logging.info(f"Step size: {args.step_size}")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"LLM engine: {args.llm_engine}")
    logging.info(f"Model: {args.model or 'default'}")
    logging.info(f"Structured data: {args.use_structured_data}")
    logging.info(f"Exclude fields: {args.exclude_fields if args.exclude_fields else 'None'}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info("="*60)
    
    try:
        # Run task
        results = run_task(args)
        
        logging.info("="*60)
        logging.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        logging.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
