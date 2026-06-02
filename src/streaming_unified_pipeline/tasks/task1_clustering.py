"""
Task-1: Streaming Document Clustering
Implements incremental and sliding window clustering with LLM-based document assignment.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

try:
    # Force PyTorch backend to avoid TensorFlow/Keras issues
    import os
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError:
    BERTopic = None
    SentenceTransformer = None

from .base_task import BaseTask
from evaluation import ClusteringEvaluator
from evaluation.clustering_metrics import compute_all_metrics, print_metrics


class ClusteringTask(BaseTask):
    """
    Streaming document clustering task.
    Supports both incremental and sliding window modes.
    """
    
    def __init__(self, 
                 data_file: str,
                 window_manager,
                 llm_client,
                 output_dir: str = "results",
                 random_seed: int = 42,
                 max_input_tokens: int = 4000,
                 texts_per_event: int = 1,
                 use_structured_data: bool = False,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 clustering_method: str = "bertopic",  # Fixed: initial_cluster_method -> clustering_method
                 min_cluster_size: int = 2,
                 similarity_threshold: float = 0.7,
                 max_clusters_per_window: int = 10,
                 balance_clusters: bool = False,
                 use_llm_keywords: bool = True,
                 keyword_num: int = 10,
                 batch_size: int = 1,
                 **kwargs):
        """
        Initialize clustering task.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            initial_cluster_method: Method for initial clustering ("bertopic")
            min_cluster_size: Minimum cluster size for BERTopic
            similarity_threshold: Threshold for cluster assignment
            max_clusters_per_window: Maximum clusters to consider per window
            balance_clusters: Whether to balance initial clusters (incremental mode)
        """
        super().__init__(
            task_name="clustering",
            data_file=data_file,
            window_manager=window_manager,
            llm_client=llm_client,
            output_dir=output_dir,
            random_seed=random_seed,
            max_input_tokens=max_input_tokens,
            texts_per_event=texts_per_event,
            use_structured_data=use_structured_data,
            **kwargs
        )
        
        self.embedding_model = embedding_model
        self.initial_cluster_method = clustering_method or "bertopic"  # Fixed variable name
        self.min_cluster_size = min_cluster_size
        self.similarity_threshold = similarity_threshold
        self.max_clusters_per_window = max_clusters_per_window or 10
        self.balance_clusters = balance_clusters
        self.use_llm_keywords = use_llm_keywords
        self.keyword_num = keyword_num
        self.batch_size = batch_size
        
        # Lazy load sentence transformer only when needed (for BERTopic)
        self.encoder = None  # Will be initialized only if BERTopic is used
        
        # Set max_input_tokens based on model capacity and texts_per_event
        self.max_input_tokens = self._calculate_max_input_tokens(
            llm_client, 
            max_input_tokens, 
            texts_per_event
        )
        
        # Clustering state for incremental mode
        self.global_clusters = {}  # cluster_id -> {'keywords': [...], 'documents': [...], 'visible': bool, 'updated': int}
        self.cluster_counter = 0
        self.all_predictions = []  # For evaluation
        self.all_ground_truth = []
        self.mode = None  # Will be set by run method
        self.step_counter = 0  # For visibility management
        
        # Update config (use instance variables to avoid scope issues)
        self.config.update({
            'embedding_model': self.embedding_model,
            'initial_cluster_method': self.initial_cluster_method,
            'min_cluster_size': self.min_cluster_size,
            'similarity_threshold': self.similarity_threshold,
            'max_clusters_per_window': self.max_clusters_per_window,
            'balance_clusters': self.balance_clusters,
            'use_llm_keywords': self.use_llm_keywords,
            'keyword_num': self.keyword_num,
            'batch_size': self.batch_size
        })
        
        logging.info(f"Using max_input_tokens={self.max_input_tokens} for model with texts_per_event={texts_per_event}")
    
    def _calculate_max_input_tokens(self, llm_client, default_max_tokens: int, texts_per_event: int) -> int:
        """
        Calculate max_input_tokens based on model capacity and texts_per_event.
        
        Strategy: Scale with texts_per_event, but respect model context limits.
        
        Args:
            llm_client: LLM client instance
            default_max_tokens: Default max tokens from config
            texts_per_event: Number of texts per event
            
        Returns:
            Calculated max_input_tokens
        """
        # Model context limits (in tokens)
        MODEL_CONTEXT_LIMITS = {
            # Gemma models
            'gemma-2-2b-it': 8192,
            'gemma-2-9b-it': 8192,
            'gemma-2-27b-it': 8192,
            'gemma-3-1b-it': 8192,
            'gemma-3-4b-it': 8192,
            # Llama models
            'llama-3.1-8b': 131072,
            'llama-3.1-70b': 131072,
            'llama-3_1-8b-instruct': 131072,
            'llama-3_1-70b-instruct': 131072,
            'llama-3.2-1b': 131072,  # Llama 3.2 series
            'llama-3.2-3b': 131072,
            # Qwen models
            'qwen2.5-7b': 131072,
            'qwen2.5-72b': 131072,
            'qwen3-1.7b': 32768,
            'qwen3-4b': 32768,
            'qwen3-8b': 32768,
            # Mistral models
            'mistral-small': 32768,
            'mistral-7b': 32768,
            # OLMo models
            'olmo-2-1124-7b': 8192,   # OLMo-2 7B (November 2024)
            'olmo-2-0425-1b': 8192,   # OLMo-2 1B (April 2025)
            'olmo-7b': 4096,          # Original OLMo
            'olmo-1b': 2048,          # Original OLMo
            'olmo-2': 8192,           # OLMo-2 series default
        }
        
        # Get model name (normalize to lowercase)
        model_name = getattr(llm_client, 'model_name', '').lower()
        
        # Find matching model limit
        model_limit = None
        for key, limit in MODEL_CONTEXT_LIMITS.items():
            if key in model_name:
                model_limit = limit
                break
        
        # If model not found, use default
        if model_limit is None:
            logging.warning(f"Model '{model_name}' not in known models, using default limit {default_max_tokens}")
            model_limit = default_max_tokens
        
        # Scale factor based on texts_per_event
        # Progressive scaling to simulate information growth
        scale_factors = {
            1: 1.0,   # baseline
            5: 2.0,   # 2x
            10: 4.0,  # 4x
            20: 8.0   # 8x
        }
        
        scale = scale_factors.get(texts_per_event, 1.0)
        
        # Calculate desired tokens
        base_tokens = 8000  # Base amount
        desired_tokens = int(base_tokens * scale)
        
        # Cap at 80% of model limit (leave room for prompt overhead, structured data, and existing clusters)
        max_allowed = int(model_limit * 0.8)
        final_tokens = min(desired_tokens, max_allowed)
        
        logging.info(f"Model: {model_name}, Limit: {model_limit}, texts_per_event: {texts_per_event}, "
                    f"Scale: {scale}x, Desired: {desired_tokens}, Final: {final_tokens}")
        
        return final_tokens
    
    def _run_incremental_mode(self):
        """Run clustering in incremental mode."""
        self.mode = 'incremental'
        events = self.data_loader.get_all_events()
        windows = self.window_manager.create_windows(events)
        
        logging.info(f"Processing {len(windows)} windows in incremental mode")
        
        # Initialize with first window
        if windows:
            first_window = windows[0]
            logging.info("Initializing clusters with first window")
            
            # Process first window to create initial clusters
            initial_result = self._initialize_clusters(first_window, 0)
            self.results['windows'].append(initial_result)
            
            # Process remaining windows incrementally with progress bar
            for i, window in enumerate(tqdm(windows[1:], desc="Processing clustering windows incrementally"), 1):
                logging.info(f"Processing window {i+1}/{len(windows)} incrementally")
                
                try:
                    window_result = self._process_incremental_window(window, i)
                    window_result['window_index'] = i
                    window_result['window_info'] = {
                        'start_date': window.get('start_date'),
                        'end_date': window.get('end_date'),
                        'event_count': len(window.get('events', []))
                    }
                    
                    self.results['windows'].append(window_result)
                    
                except Exception as e:
                    logging.error(f"Error processing incremental window {i}: {e}")
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
    
    def process_window(self, window: Dict[str, Any], window_index: int) -> Dict[str, Any]:
        """
        Process a single window in sliding mode (Original Task-1 approach).
        Each window is processed incrementally: initialize with first document, 
        then assign remaining documents one by one using LLM.
        
        Args:
            window: Window data
            window_index: Window index
            
        Returns:
            Window processing results
        """
        # Set mode for evaluation
        self.mode = 'sliding'
        
        # Get clustering documents from window (uses texts_per_event)
        documents = self.data_loader.get_clustering_documents(window['events'])
        
        if not documents:
            return {
                'method': 'sliding_incremental',
                'cluster_assignments': [],
                'cluster_keywords': {},
                'num_clusters': 0,
                'num_documents': 0
            }
        
        logging.info(f"Window {window_index}: Processing {len(documents)} documents incrementally")
        
        # Initialize window clusters with first document (Original Task-1 approach)
        try:
            first_doc_text = documents[0]['text']
            
            if self.use_llm_keywords:
                # Use LLM to extract initial keywords from first document
                initial_keywords = self._get_initial_keywords_with_llm(first_doc_text)
                logging.info(f"Window {window_index}: Initialized with LLM keywords: {initial_keywords}")
            else:
                # Fallback to simple keyword extraction
                initial_keywords = self._extract_keywords_from_text(first_doc_text, self.keyword_num)
                logging.info(f"Window {window_index}: Initialized with simple keywords: {initial_keywords}")
            
            # Initialize window-specific clusters (reset for each window)
            window_clusters = {
                0: {
                    'keywords': initial_keywords,
                    'visible': True,
                    'updated': 0
                }
            }
            
            cluster_assignments = []
            ground_truth = []
            step = 0
            
            # Process each document incrementally within this window
            for doc in documents:
                try:
                    # Assign document to cluster using LLM
                    assigned_cluster, is_new_cluster = self._assign_to_cluster_with_llm_for_window(
                        doc, window_clusters, step
                    )
                    
                    cluster_assignments.append(assigned_cluster)
                    ground_truth.append(doc['true_topic'])
                    
                    if is_new_cluster:
                        logging.info(f"Window {window_index}, Step {step}: Created new cluster {assigned_cluster}")
                    
                    step += 1
                    
                except Exception as e:
                    logging.error(f"Error assigning document in window {window_index}, step {step}: {e}")
                    cluster_assignments.append(-1)  # Assign to noise cluster
                    ground_truth.append(doc['true_topic'])
                    step += 1
            
            # Store for evaluation
            self.all_predictions.extend(cluster_assignments)
            self.all_ground_truth.extend(ground_truth)
            
            # Extract cluster keywords for reporting
            cluster_keywords = {cid: info['keywords'] for cid, info in window_clusters.items()}
            
            return {
                'method': 'sliding_incremental_llm',
                'cluster_assignments': cluster_assignments,
                'cluster_keywords': cluster_keywords,
                'num_clusters': len([k for k in window_clusters.keys() if k != -1]),
                'num_documents': len(documents),
                'ground_truth': ground_truth,
                'window_clusters': window_clusters
            }
            
        except Exception as e:
            logging.error(f"Error processing window {window_index}: {e}")
            raise
    
    def _initialize_clusters(self, window: Dict[str, Any], window_index: int) -> Dict[str, Any]:
        """Initialize clusters using the first window."""
        window_data = self.data_loader.prepare_window_data(window['events'])
        documents = self.data_loader.get_clustering_documents(window['events'])
        
        if not documents:
            return {
                'method': 'incremental_init',
                'cluster_assignments': [],
                'cluster_keywords': {},
                'num_clusters': 0,
                'num_documents': 0,
                'window_index': window_index
            }
        
        # Option 1: LLM-based keyword extraction (New!)
        if self.use_llm_keywords and documents:
            logging.info("Using LLM for initial keyword extraction")
            first_doc_text = documents[0]['text']
            keywords = self._get_initial_keywords_with_llm(first_doc_text)
            
            # Initialize single cluster with LLM keywords
            cluster_assignments = [0] * len(documents)
            cluster_keywords = {0: keywords}
            
            # Set up global clusters for incremental mode
            self.global_clusters[0] = {
                "keywords": keywords,
                "documents": documents,
                "visible": True,
                "updated": 0
            }
            
            # Store for evaluation
            ground_truth = [doc['true_topic'] for doc in documents]
            self.all_predictions.extend(cluster_assignments)
            self.all_ground_truth.extend(ground_truth)
            
            return {
                'method': 'incremental_init_llm',
                'cluster_assignments': cluster_assignments,
                'cluster_keywords': cluster_keywords,
                'num_clusters': 1,
                'num_documents': len(documents),
                'global_clusters_count': len(self.global_clusters),
                'ground_truth': ground_truth,
                'window_index': window_index
            }
        
        # Apply balance clustering if enabled (original Task-1 feature)
        balanced_documents = documents
        if self.balance_clusters:
            balanced_documents = self._balance_initial_clusters(documents)
        
        # Perform initial clustering
        if self.initial_cluster_method == "bertopic":
            cluster_assignments, cluster_keywords = self._cluster_with_bertopic(balanced_documents)
        else:
            raise ValueError(f"Unsupported clustering method: {self.initial_cluster_method}")
        
        # Initialize global clusters
        self.global_clusters = {}
        self.cluster_counter = 0
        
        for cluster_id, keywords in cluster_keywords.items():
            if cluster_id != -1:  # Skip noise cluster
                global_id = self.cluster_counter
                self.global_clusters[global_id] = {
                    'keywords': keywords,
                    'documents': [],
                    'visible': True,
                    'updated': self.step_counter
                }
                self.cluster_counter += 1
        
        # Add documents to global clusters
        for doc, cluster_id in zip(documents, cluster_assignments):
            if cluster_id != -1:
                # Map local cluster ID to global cluster ID
                global_id = list(cluster_keywords.keys()).index(cluster_id) if cluster_id in cluster_keywords else None
                if global_id is not None and global_id < len(self.global_clusters):
                    self.global_clusters[global_id]['documents'].append(doc)
        
        # Store for evaluation
        ground_truth = [doc['true_topic'] for doc in documents]
        self.all_predictions.extend(cluster_assignments)
        self.all_ground_truth.extend(ground_truth)
        
        return {
            'method': 'incremental_init',
            'cluster_assignments': cluster_assignments,
            'cluster_keywords': cluster_keywords,
            'num_clusters': len([k for k in cluster_keywords.keys() if k != -1]),
            'num_documents': len(documents),
            'global_clusters_count': len(self.global_clusters),
            'ground_truth': ground_truth,
            'window_index': window_index
        }
    
    def _process_incremental_window(self, window: Dict[str, Any], window_index: int) -> Dict[str, Any]:
        """Process window in incremental mode."""
        window_data = self.data_loader.prepare_window_data(window['events'])
        documents = self.data_loader.get_clustering_documents(window['events'])
        
        if not documents:
            return {
                'method': 'incremental',
                'cluster_assignments': [],
                'new_clusters': 0,
                'num_documents': 0,
                'global_clusters_count': len(self.global_clusters)
            }
        
        cluster_assignments = []
        new_clusters = 0
        
        # Process each document
        for doc in documents:
            try:
                self.step_counter += 1  # Increment step counter
                
                # Use LLM to assign document to cluster or create new one
                assigned_cluster, is_new_cluster = self._assign_document_with_llm(doc)
                cluster_assignments.append(assigned_cluster)
                
                # Manage visibility for the assigned cluster
                if assigned_cluster != -1:
                    self._manage_visibility(assigned_cluster)
                
                if is_new_cluster:
                    new_clusters += 1
                
            except Exception as e:
                logging.error(f"Error assigning document: {e}")
                cluster_assignments.append(-1)  # Assign to noise cluster
        
        # Store for evaluation
        ground_truth = [doc['true_topic'] for doc in documents]
        self.all_predictions.extend(cluster_assignments)
        self.all_ground_truth.extend(ground_truth)
        
        # Note: No intermediate evaluation in incremental mode (original Task-1 behavior)
        
        return {
            'method': 'incremental',
            'cluster_assignments': cluster_assignments,
            'new_clusters': new_clusters,
            'num_documents': len(documents),
            'global_clusters_count': len(self.global_clusters),
            'ground_truth': ground_truth
        }
    
    def _cluster_with_bertopic(self, documents: List[Dict]) -> Tuple[List[int], Dict[int, List[str]]]:
        """Cluster documents using BERTopic."""
        if BERTopic is None:
            raise ImportError("bertopic required for clustering. Install with: pip install bertopic")
        
        # Lazy load encoder only when BERTopic is actually used
        if self.encoder is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers required for BERTopic. Install with: pip install sentence-transformers")
            logging.info(f"Loading sentence transformer on CPU: {self.embedding_model}")
            self.encoder = SentenceTransformer(self.embedding_model, device='cpu')
        
        # Extract texts
        texts = [doc['text'] for doc in documents]
        
        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=self.encoder,
            min_topic_size=self.min_cluster_size,
            verbose=False
        )
        
        # Fit and predict 
        topics, _ = topic_model.fit_transform(texts)
        
        # Get topic keywords
        cluster_keywords = {}
        topic_info = topic_model.get_topic_info()
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1:  # Skip outliers
                keywords = [word for word, _ in topic_model.get_topic(topic_id)[:5]]
                cluster_keywords[topic_id] = keywords
        
        return topics, cluster_keywords
    
    def _balance_initial_clusters(self, documents: List[Dict]) -> List[Dict]:
        """
        Balance initial clusters by sampling equal number of documents per true cluster.
        Original Task-1 implementation.
        """
        from collections import defaultdict
        import random
        
        # Group documents by true cluster
        cluster_docs = defaultdict(list)
        for doc in documents:
            true_cluster = doc.get('true_topic', doc.get('topic', -1))
            cluster_docs[true_cluster].append(doc)
        
        if not cluster_docs:
            return documents
            
        # Find minimum cluster size
        min_docs = min(len(docs) for docs in cluster_docs.values())
        
        if min_docs == 0:
            return documents
            
        # Sample equal number from each cluster
        balanced_docs = []
        random.seed(42)  # For reproducibility
        
        for cluster_id, docs in cluster_docs.items():
            if len(docs) >= min_docs:
                sampled = random.sample(docs, min_docs)
            else:
                sampled = docs
            balanced_docs.extend(sampled)
            
        logging.info(f"Balanced clustering: {len(documents)} -> {len(balanced_docs)} documents")
        logging.info(f"Min cluster size: {min_docs}, Clusters: {len(cluster_docs)}")
        
        return balanced_docs
    
    def _assign_to_cluster_with_llm_incremental(self, document: Dict) -> Tuple[int, bool]:
        """Assign document to existing cluster or create new one using LLM (for incremental mode)."""
        text = document['text']
        event_date = document.get('event_date')
        
        # Get structured data if available
        structured_data = None
        if self.use_structured_data and 'event' in document:
            structured_data = self.data_loader._extract_structured_data(document['event'])
        
        # Create prompt with existing clusters
        prompt = self._create_clustering_prompt(text, self.global_clusters, structured_data, event_date)
        
        try:
            # Get LLM response
            response = self.llm_client.generate(
                prompt,
                max_tokens=200,
                temperature=0.0  # Completely deterministic
            )
            
            # Parse response using regex (original method)
            cluster_id, keywords = self._parse_clustering_response_regex(response)
            
            # Check if cluster exists and is visible
            visible_clusters = [cid for cid in self.global_clusters if self.global_clusters[cid].get('visible', True)]
            is_new_cluster = cluster_id not in visible_clusters
            
            if is_new_cluster:
                # Create new cluster
                if cluster_id not in self.global_clusters:
                    # Completely new cluster
                    self.global_clusters[cluster_id] = {
                        'keywords': keywords,
                        'documents': [document],
                        'visible': True,
                        'updated': self.step_counter
                    }
                    if cluster_id >= self.cluster_counter:
                        self.cluster_counter = cluster_id + 1
                else:
                    # Reactivating existing cluster
                    self.global_clusters[cluster_id]['documents'].append(document)
                    self.global_clusters[cluster_id]['keywords'] = keywords
                return cluster_id, True
            else:
                # Add to existing visible cluster and update keywords
                if cluster_id in self.global_clusters:
                    self.global_clusters[cluster_id]['documents'].append(document)
                    self.global_clusters[cluster_id]['keywords'] = keywords  # Update keywords
                return cluster_id, False
                
        except Exception as e:
            logging.error(f"Error in LLM clustering: {e}")
            # Default to noise cluster
            return -1, False
    
    def _assign_to_cluster_with_llm_for_window(self, document: Dict, window_clusters: Dict, step: int) -> Tuple[int, bool]:
        """
        Assign document to existing cluster or create new one using LLM (for sliding mode).
        Similar to _assign_document_with_llm but uses window-specific clusters.
        
        Args:
            document: Document to assign
            window_clusters: Window-specific cluster dictionary
            step: Current step number
            
        Returns:
            Tuple of (cluster_id, is_new_cluster)
        """
        text = document['text']
        event_date = document.get('event_date')
        
        # Get structured data if available
        structured_data = None
        if self.use_structured_data and 'event' in document:
            structured_data = self.data_loader._extract_structured_data(document['event'])
        
        # Create prompt with window clusters
        prompt = self._create_clustering_prompt(text, window_clusters, structured_data, event_date)
        
        try:
            # Get LLM response
            response = self.llm_client.generate(
                prompt,
                max_tokens=200,
                temperature=0.0  # Completely deterministic
            )
            
            # Parse response using regex
            cluster_id, keywords = self._parse_clustering_response_regex(response)
            
            # Check if this is a new cluster
            is_new_cluster = cluster_id not in window_clusters
            
            if is_new_cluster:
                # Create new cluster in window
                window_clusters[cluster_id] = {
                    'keywords': keywords,
                    'visible': True,
                    'updated': step
                }
                logging.info(f"Created new cluster {cluster_id} with keywords: {keywords}")
            else:
                # Update existing cluster keywords
                window_clusters[cluster_id]['keywords'] = keywords
                window_clusters[cluster_id]['updated'] = step
                logging.debug(f"Updated cluster {cluster_id} with keywords: {keywords}")
            
            return cluster_id, is_new_cluster
            
        except Exception as e:
            logging.error(f"Error in LLM clustering for window: {e}")
            # Default to noise cluster
            return -1, False
    
    def _manage_visibility(self, cluster_id: int) -> None:
        """
        Manage cluster visibility based on max_clusters_per_window.
        Deactivate oldest clusters when limit is exceeded.
        """
        if cluster_id in self.global_clusters:
            if not self.global_clusters[cluster_id].get("visible", True):
                # Check if we need to make room for this cluster
                visible_clusters = [cid for cid in self.global_clusters if self.global_clusters[cid].get("visible", True)]
                if len(visible_clusters) >= self.max_clusters_per_window:
                    # Find oldest visible cluster
                    oldest = min(visible_clusters, key=lambda cid: self.global_clusters[cid].get("updated", 0))
                    if oldest != cluster_id:
                        self.global_clusters[oldest]["visible"] = False
                        logging.info(f"Deactivated cluster {oldest} (oldest)")
            # Update this cluster's visibility and timestamp
            self.global_clusters[cluster_id].update({"visible": True, "updated": self.step_counter})
        else:
            # New cluster - check if we need to make room
            visible_clusters = [cid for cid in self.global_clusters if self.global_clusters[cid].get("visible", True)]
            if len(visible_clusters) >= self.max_clusters_per_window:
                # Find oldest visible cluster to deactivate
                oldest = min(visible_clusters, key=lambda cid: self.global_clusters[cid].get("updated", 0))
                if oldest != cluster_id:
                    self.global_clusters[oldest]["visible"] = False
                    logging.info(f"Deactivated cluster {oldest} to make room for new cluster {cluster_id}")
            
            # Initialize new cluster with visibility
            if cluster_id not in self.global_clusters:
                self.global_clusters[cluster_id] = {
                    "keywords": [],
                    "documents": [],
                    "visible": True,
                    "updated": self.step_counter
                }
    
    def _create_clustering_prompt(self, text: str, clusters: Dict, structured_data: Dict = None, event_date: str = None) -> str:
        """Create prompt for LLM-based clustering. Improved version with better instructions."""
        # Get current visible cluster info
        visible_clusters = {k: v["keywords"] for k, v in clusters.items() if v.get("visible", True)}
        
        # Truncate text if too long (avoid token overflow)
        max_text_chars = 2000  # ~500 tokens
        if len(text) > max_text_chars:
            text = text[:max_text_chars] + "... [truncated]"
        
        # Build structured info string if available (same format as QA task)
        structured_str = None
        if self.use_structured_data and structured_data:
            # Use the same format_structured_data method as QA task
            structured_str = self.data_loader.format_structured_data(structured_data)
        
        # Format article text with date if available (like QA task)
        if event_date:
            formatted_text = f"[Date: {event_date}]\n{text}"
        else:
            formatted_text = text
            
        # Build prompt parts
        prompt_parts = [
            "Your task is to assign the following article to an existing topic cluster or create a new topic cluster.",
            "",
            "Article to cluster:",
            f"{formatted_text}",
        ]
        
        if structured_str:
            prompt_parts.append("")
            prompt_parts.append("Additional Context:")
            prompt_parts.append(structured_str)
            
        # Build keyword format string for prompt
        keyword_placeholders = ", ".join([f"keyword{i+1}" for i in range(self.keyword_num)])
        
        # Show existing clusters
        if visible_clusters:
            prompt_parts.append("")
            prompt_parts.append("Existing Clusters:")
            for cid, keywords in visible_clusters.items():
                prompt_parts.append(f"  Cluster {cid}: {keywords}")
        else:
            prompt_parts.append("")
            prompt_parts.append("Existing Clusters: (none - this is the first article, start with cluster 0)")
        
        prompt_parts.extend([
            "",
            "Guidelines:",
            "1. Articles about the SAME EVENT/TOPIC → assign to existing cluster",
            "2. Compare keywords: if 70%+ overlap with existing cluster → assign to that cluster",
            "3. Articles about CLEARLY DIFFERENT topics → create new cluster",
            "4. Same people/entities but different events → different clusters",
            "5. Different dates OK if same ongoing story (e.g., election campaign, investigation)",
        ])
        
        # Add instruction to use structured data if available
        if self.use_structured_data and structured_str:
            prompt_parts.extend([
                "6. Use both the article text AND additional context to identify the topic",
                f"7. Extract up to {self.keyword_num} keywords that capture the topic",
                "8. If assigning to existing cluster, update keywords to cover both old and new content",
            ])
        else:
            prompt_parts.extend([
                f"6. Extract up to {self.keyword_num} keywords that best represent the article's topic",
                "7. If assigning to existing cluster, update keywords to cover both old and new content",
            ])
        
        # Show existing clusters and instructions
        if visible_clusters:
            prompt_parts.append(f"- If article matches an existing cluster ({', '.join(map(str, visible_clusters.keys()))}), use that cluster ID")
            next_cluster_id = max(visible_clusters.keys()) + 1
            prompt_parts.append(f"- If article is a NEW different topic, use cluster ID {next_cluster_id}")
            prompt_parts.append("")
            prompt_parts.append(f"Example for existing: (0, ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10'])")
            prompt_parts.append(f"Example for NEW topic: ({next_cluster_id}, ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10'])")
        else:
            prompt_parts.append("- This is the first article, use cluster ID 0")
            prompt_parts.append("")
            prompt_parts.append("Example: (0, ['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5', 'keyword6', 'keyword7', 'keyword8', 'keyword9', 'keyword10'])")
        
        prompt_parts.extend([
            "",
            "IMPORTANT: Each DIFFERENT topic needs a DIFFERENT cluster number.",
            "Output ONLY one line - no explanation, no code, no extra text:",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_clustering_response_regex(self, response: str) -> Tuple[int, List[str]]:
        """
        Parse LLM response using regex to extract (cluster_id, [keywords]).
        Improved to handle code blocks and extra text.
        """
        import re
        
        logging.info(f"Parsing response: {response[:500]}...")  # Truncate for logging
        
        # Remove code blocks if present
        response_clean = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        response_clean = re.sub(r'`.*?`', '', response_clean)
        
        # Try to find the pattern: (number, [...])
        # More flexible regex that handles various formats
        patterns = [
            r'\((\d+),\s*\[(.*?)\]\)',  # Standard: (0, [...])
            r'\((\d+)\s*,\s*\[(.*?)\]\)',  # With spaces
            r'return\s*\((\d+),\s*\[(.*?)\]\)',  # With return statement
        ]
        
        match = None
        for pattern in patterns:
            match = re.search(pattern, response_clean, re.DOTALL)
            if match:
                break
        
        if not match:
            # Try to find just a number at the beginning
            logging.warning(f"Standard patterns failed, trying fallback parsing")
            # Look for any (number, [...]) pattern in original response
            match = re.search(r'\((\d+).*?\[(.+?)\]', response, re.DOTALL)
            
        if not match:
            logging.error(f"Failed to parse response: {response[:200]}...")
            raise ValueError(f"Cannot parse cluster assignment from response")
            
        cluster_id = int(match.group(1))
        keywords_str = match.group(2)
        
        # Extract keywords, handling quotes and various separators
        keywords = []
        for kw in re.split(r',|\n', keywords_str):
            kw_clean = kw.strip().strip('"\'\\')
            if kw_clean and not kw_clean.startswith('#'):
                keywords.append(kw_clean)
        
        # Limit to keyword_num and ensure we have at least some keywords
        keywords = keywords[:self.keyword_num]
        if not keywords:
            keywords = ['unknown']
        
        logging.info(f"Parsed cluster_id: {cluster_id}, keywords: {keywords}")
        return cluster_id, keywords
    
    def _parse_clustering_response(self, response: str) -> Tuple[int, bool]:
        """Parse LLM clustering response."""
        response = response.strip().upper()
        
        if "NEW_CLUSTER" in response:
            return self.cluster_counter, True
        elif "ASSIGN" in response:
            try:
                # Extract cluster ID
                parts = response.split()
                for i, part in enumerate(parts):
                    if part == "ASSIGN" and i + 1 < len(parts):
                        cluster_id = int(parts[i + 1])
                        if cluster_id in self.global_clusters:
                            return cluster_id, False
                        break
            except (ValueError, IndexError):
                pass
        
        # Default to new cluster if parsing fails
        return self.cluster_counter, True
    
    def _extract_keywords_from_text(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Simple keyword extraction - can be improved with more sophisticated methods
        words = text.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Get most frequent words
        word_counts = Counter(words)
        keywords = [word for word, _ in word_counts.most_common(max_keywords)]
        
        return keywords or ['unknown']
    
    def _get_initial_keywords_with_llm(self, text: str) -> List[str]:
        """
        Extract initial keywords using LLM.
        Based on the original Task-1 implementation.
        """
        import re
        import ast
        
        keyword_list_example = ", ".join([f"'keyword{i+1}'" for i in range(self.keyword_num)])
        prompt = (
            f"You are an expert at summarizing news articles using concise keywords.\n"
            f"Extract exactly {self.keyword_num} representative keywords from the following article.\n"
            f"The keywords should be concrete nouns or phrases that best summarize the topic.\n\n"
            f"Article:\n{text}\n\n"
            f"Respond ONLY with a Python list in this format:\n"
            f"[{keyword_list_example}]\n"
            f"No explanation. Only output the list."
        )
        
        logging.info("============ Initial Keyword Prompt ============")
        logging.info(prompt)
        
        try:
            # Get LLM response
            response = self.llm_client.generate(
                prompt,
                max_tokens=128,  # Increased for 10 keywords
                temperature=0.0  # Completely deterministic
            )
            
            logging.info("============ LLM Response ============")
            logging.info(response)
            
            # Try to parse as Python list
            try:
                keywords = ast.literal_eval(response.strip())
                if isinstance(keywords, list):
                    keywords = [kw.strip() for kw in keywords if isinstance(kw, str)]
                    return keywords[:self.keyword_num]
            except Exception as e:
                logging.warning(f"Failed to parse keywords using ast.literal_eval: {e}")
            
            # Fallback: regex extraction
            try:
                raw = re.findall(r"\[.*\]", response)
                if raw:
                    content = raw[0].strip("[]")
                    keywords = [kw.strip().strip("'\"") for kw in content.split(",")]
                    keywords = [kw for kw in keywords if kw]
                    return keywords[:self.keyword_num]
            except Exception as e:
                logging.warning(f"Failed to parse keywords using regex: {e}")
            
            # Last resort: extract simple words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', response)
            return words[:self.keyword_num] if words else ['unknown']
            
        except Exception as e:
            logging.error(f"Error in LLM keyword extraction: {e}")
            # Fallback to simple keyword extraction
            return self._extract_keywords_from_text(text, self.keyword_num)
    
    def _extract_llm_keywords_for_clusters(self, documents: List[Dict], cluster_assignments: List[int], 
                                          bertopic_keywords: Dict[int, List[str]]) -> Dict[int, List[str]]:
        """
        Extract LLM-based keywords for each cluster.
        Uses representative document(s) from each cluster.
        
        Args:
            documents: List of document dictionaries
            cluster_assignments: Cluster assignment for each document
            bertopic_keywords: Original BERTopic keywords (fallback)
            
        Returns:
            Dictionary mapping cluster_id to LLM-extracted keywords
        """
        from collections import defaultdict
        
        # Group documents by cluster
        cluster_docs = defaultdict(list)
        for doc, cluster_id in zip(documents, cluster_assignments):
            if cluster_id != -1:  # Skip noise cluster
                cluster_docs[cluster_id].append(doc)
        
        llm_keywords = {}
        
        for cluster_id, docs in cluster_docs.items():
            if not docs:
                # Use BERTopic keywords as fallback
                llm_keywords[cluster_id] = bertopic_keywords.get(cluster_id, ['unknown'])
                continue
            
            try:
                # Use the first document as representative (or could concatenate multiple)
                representative_text = docs[0]['text']
                
                # Extract keywords using LLM
                keywords = self._get_initial_keywords_with_llm(representative_text)
                llm_keywords[cluster_id] = keywords
                
                logging.info(f"Cluster {cluster_id}: Extracted LLM keywords: {keywords}")
                
            except Exception as e:
                logging.error(f"Error extracting LLM keywords for cluster {cluster_id}: {e}")
                # Fallback to BERTopic keywords
                llm_keywords[cluster_id] = bertopic_keywords.get(cluster_id, ['unknown'])
        
        # Handle noise cluster if it exists
        if -1 in bertopic_keywords:
            llm_keywords[-1] = bertopic_keywords[-1]
        
        return llm_keywords
    
    def evaluate_results(self) -> Dict[str, Any]:
        """Evaluate clustering results using original Task-1 metrics."""
        if not self.all_predictions or not self.all_ground_truth:
            logging.warning("No predictions or ground truth available for evaluation")
            return {'error': 'No data for evaluation'}
        
        try:
            # Different evaluation for incremental vs sliding window modes
            if hasattr(self, 'mode') and self.mode == 'sliding':
                return self._evaluate_sliding_window_results()
            else:
                return self._evaluate_incremental_results()
                
        except Exception as e:
            logging.error(f"Error in clustering evaluation: {e}")
            return {'error': str(e)}
    
    def _evaluate_incremental_results(self) -> Dict[str, Any]:
        """Evaluate incremental clustering results (original Task-1 behavior)."""
        # Use original Task-1 evaluation metrics
        original_metrics = compute_all_metrics(self.all_ground_truth, self.all_predictions)
        
        # Print metrics in original format
        print_metrics(original_metrics, "Clustering Eval (Incremental Mode)")
        
        # Also compute standard clustering metrics for comparison
        evaluator = ClusteringEvaluator()
        standard_metrics = evaluator.evaluate(
            self.all_predictions, 
            self.all_ground_truth
        )
        
        # Combine results
        combined_results = {
            'original_task1_metrics': original_metrics,
            'standard_metrics': standard_metrics,
            'num_predictions': len(self.all_predictions),
            'num_ground_truth': len(self.all_ground_truth),
            'num_clusters_predicted': len(set(self.all_predictions)),
            'num_clusters_true': len(set(self.all_ground_truth)),
            'mode': 'incremental'
        }
        
        return combined_results
    
    def _evaluate_sliding_window_results(self) -> Dict[str, Any]:
        """Evaluate sliding window results with per-window metrics and averaging (original Task-1 behavior)."""
        all_window_metrics = []
        
        # Evaluate each window separately
        for i, window_result in enumerate(self.results.get('windows', [])):
            if 'cluster_assignments' in window_result and 'ground_truth' in window_result:
                pred_labels = window_result['cluster_assignments']
                true_labels = window_result['ground_truth']
                
                if pred_labels and true_labels and len(pred_labels) == len(true_labels):
                    window_metrics = compute_all_metrics(true_labels, pred_labels)
                    all_window_metrics.append(window_metrics)
                    
                    # Print per-window evaluation (original Task-1 behavior)
                    print_metrics(window_metrics, f"Clustering Eval (Window {i})")
        
        if not all_window_metrics:
            logging.warning("No valid window metrics found")
            return {'error': 'No valid window metrics'}
        
        # Calculate average metrics across all windows (original Task-1 behavior)
        avg_metrics = {
            key: sum(metrics[key] for metrics in all_window_metrics) / len(all_window_metrics)
            for key in all_window_metrics[0].keys()
        }
        
        # Print average metrics
        print_metrics(avg_metrics, "Average Clustering Evaluation")
        
        # Also compute standard clustering metrics for comparison
        evaluator = ClusteringEvaluator()
        standard_metrics = evaluator.evaluate(
            self.all_predictions, 
            self.all_ground_truth
        )
        
        combined_results = {
            'original_task1_metrics': avg_metrics,
            'window_metrics': all_window_metrics,
            'standard_metrics': standard_metrics,
            'num_windows': len(all_window_metrics),
            'num_predictions': len(self.all_predictions),
            'num_ground_truth': len(self.all_ground_truth),
            'num_clusters_predicted': len(set(self.all_predictions)),
            'num_clusters_true': len(set(self.all_ground_truth)),
            'mode': 'sliding'
        }
        
        return combined_results
