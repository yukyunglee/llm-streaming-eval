"""
Implements sliding window summarization with LLM-based summary generation.
"""

import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

try:
    from rouge import Rouge
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from evaluate import load as load_metric
except ImportError:
    Rouge = None
    sentence_bleu = None
    SmoothingFunction = None
    load_metric = None

from .base_task import BaseTask
from core.llm_client import UnifiedLLMClient


class SummarizationTask(BaseTask):
    """
    Streaming summarization task using sliding windows.
    Supports structured data integration in summary generation.
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
                 summary_max_tokens: int = 150,
                 temperature: float = 0.0,
                 include_context: bool = True,
                 label_type: str = "concat",
                 batch_size: int = 1,
                 use_cache: bool = True,
                 cache_file: Optional[str] = None,
                 **kwargs):
        """
        Initialize summarization task.
        Args:
            summary_max_tokens: Maximum tokens for generated summaries
            temperature: LLM temperature for summary generation
            include_context: Whether to include context in summary prompts
            label_type: Label type for reference summaries ('concat' or 'abstract')
        """
        super().__init__(
            task_name="summarization",
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
        
        # Task-specific attributes
        self.summary_max_tokens = summary_max_tokens
        self.temperature = temperature
        self.include_context = include_context
        self.label_type = label_type
        self.batch_size = batch_size
        self.use_cache = use_cache
        
        # Set max_input_tokens based on model capacity and texts_per_event
        self.max_input_tokens = self._calculate_max_input_tokens(
            llm_client, 
            max_input_tokens, 
            texts_per_event
        )
        
        # Abstract label cache setup
        self.label_cache_dir = os.path.join(os.path.dirname(data_file), 'abstract_labels')
        os.makedirs(self.label_cache_dir, exist_ok=True)
        
        # Label file path based on data file and window settings
        data_basename = os.path.basename(data_file).replace('.json', '')
        self.label_cache_file = os.path.join(
            self.label_cache_dir,
            f"{data_basename}_abstract_labels_w{window_manager.window_size}_s{window_manager.stride}.json"
        )
        
        self.abstract_label_cache = self._load_label_cache_file()
        self.cache_hits = 0
        self.cache_misses = 0
        logging.info(f"Abstract label cache file: {self.label_cache_file}")
        
        # Storage for evaluation
        self.generated_summaries = []
        self.reference_summaries = []
        self.topics = []
        self.window_indices = []
        self.event_counts = []
        
        # Update config
        self.config.update({
            'summary_max_tokens': summary_max_tokens,
            'temperature': temperature,
            'include_context': include_context,
            'label_type': label_type,
            'batch_size': batch_size
        })
        
        logging.info(f"Initialized summarization task with max_tokens={summary_max_tokens}, batch_size={batch_size}")
        logging.info(f"Using max_input_tokens={self.max_input_tokens} for model with texts_per_event={texts_per_event}")
        if self.label_type == 'abstract':
            logging.info(f"Abstract label cache: {len(self.abstract_label_cache)} entries loaded")
    
    def _load_label_cache_file(self) -> Dict[str, Dict]:
        """
        Load abstract labels from cache file.
        
        Returns:
            Dictionary mapping window_index to label data
        """
        if not os.path.exists(self.label_cache_file):
            logging.info(f"No existing label cache file found. Will create: {self.label_cache_file}")
            return {}
        
        try:
            with open(self.label_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            logging.info(f"Loaded {len(cache_data)} abstract labels from cache: {self.label_cache_file}")
            return cache_data
            
        except Exception as e:
            logging.warning(f"Failed to load abstract label cache: {e}")
            return {}
    
    def _save_label_to_cache(self, window_index: int, reference_summaries: List[str], abstract_label: str):
        """
        Save abstract label to cache file.
        
        Args:
            window_index: Window index
            reference_summaries: Original reference summaries
            abstract_label: Generated abstract label
        """
        cache_key = str(window_index)
        self.abstract_label_cache[cache_key] = {
            'label': abstract_label,
            'reference_summaries': reference_summaries,
            'generated_at': datetime.now().isoformat()
        }
        
        try:
            with open(self.label_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.abstract_label_cache, f, indent=2, ensure_ascii=False)
            logging.debug(f"Saved abstract label for window {window_index} to cache")
        except Exception as e:
            logging.error(f"Failed to save abstract label to cache: {e}")
    
    
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
        
        # Cap at 90% of model limit (truncation logic handles the rest)
        max_allowed = int(model_limit * 0.9)
        final_tokens = min(desired_tokens, max_allowed)
        
        logging.info(f"Model: {model_name}, Limit: {model_limit}, texts_per_event: {texts_per_event}, "
                    f"Scale: {scale}x, Desired: {desired_tokens}, Final: {final_tokens}")
        
        return final_tokens
    
    def process_window(self, window: Dict[str, Any], window_index: int) -> Dict[str, Any]:
        """
        Process a single window to generate summary.
        
        Args:
            window: Window data
            window_index: Window index
            
        Returns:
            Window processing results
        """
        # Prepare window data with optional structured data
        window_data = self.data_loader.prepare_window_data(
            window['events'],
            include_structured_data=self.use_structured_data
        )
        
        input_texts = window_data['input_texts']
        reference_summaries = window_data['reference_summaries']
        structured_data_list = window_data.get('structured_data', [])
        
        # Apply truncation to both structured data and input texts
        # Token budget: max_input_tokens - prompt_overhead
        if input_texts:
            BASE_PROMPT_OVERHEAD = 200
            total_budget = self.max_input_tokens - BASE_PROMPT_OVERHEAD
            
            if self.use_structured_data and structured_data_list:
                # Allocate 30% for structured data, 70% for input texts
                structured_budget = int(total_budget * 0.3)
                text_budget = int(total_budget * 0.7)
                
                # Truncate structured data if needed
                structured_data_list = self._truncate_structured_data(
                    structured_data_list, structured_budget
                )
            else:
                text_budget = total_budget
            
            input_texts = self._truncate_texts_equally(input_texts, text_budget)
        
        if not input_texts:
            return {
                'generated_summary': '',
                'reference_summaries': reference_summaries,
                'num_input_texts': 0,
                'num_events': window_data['num_events'],
                'error': 'No input texts available'
            }
        
        try:
            # Create summarization prompt
            prompt = self._create_summarization_prompt(
                input_texts, 
                structured_data_list if self.use_structured_data else None
            )
            
            # Safety check: if prompt too long, truncate both structured data and input texts
            max_prompt_tokens = 7000  # Leave room for output tokens
            estimated_tokens = len(prompt) // 2  # Conservative: 2 chars per token
            while estimated_tokens > max_prompt_tokens:
                # Reduce both structured data and input texts proportionally
                reduction_ratio = max_prompt_tokens / estimated_tokens * 0.85
                
                # Truncate structured data
                if self.use_structured_data and structured_data_list:
                    sd_budget = int(sum(len(self.data_loader.format_structured_data(sd)) for sd in structured_data_list if sd) * reduction_ratio) // 2
                    structured_data_list = self._truncate_structured_data(structured_data_list, sd_budget)
                
                # Truncate input texts
                reduced_text_budget = int(len(''.join(input_texts)) * reduction_ratio) // 2
                input_texts = self._truncate_texts_equally(input_texts, max(reduced_text_budget, 500))
                
                # Recreate prompt and check again
                prompt = self._create_summarization_prompt(
                    input_texts,
                    structured_data_list if self.use_structured_data else None
                )
                new_estimated = len(prompt) // 2
                if new_estimated >= estimated_tokens:  # Prevent infinite loop
                    break
                estimated_tokens = new_estimated
            
            # Generate summary using LLM
            generated_summary = self.llm_client.generate(
                prompt,
                max_tokens=self.summary_max_tokens,
                temperature=self.temperature
            )
            
            # Clean generated summary (remove unnecessary newlines and whitespace)
            generated_summary = self._clean_summary(generated_summary)
            
            # Generate reference label based on label_type
            reference_label = self._generate_reference_label(reference_summaries, window_index)
            
            # Store for evaluation
            self.generated_summaries.append(generated_summary)
            self.reference_summaries.append(reference_label)
            # Store metadata
            topic = window.get('topic', 'unknown')
            self.topics.append(topic)
            self.window_indices.append(window_index)
            self.event_counts.append(window_data['num_events'])
            
            return {
                'generated_summary': generated_summary,
                'reference_summaries': reference_summaries,
                'num_input_texts': len(input_texts),
                'num_events': window_data['num_events'],
                'input_text_length': sum(len(text) for text in input_texts),
                'summary_length': len(generated_summary),
                'structured_data_used': bool(self.use_structured_data and structured_data_list)
            }
            
        except Exception as e:
            logging.error(f"Error generating summary for window {window_index}: {e}")
            return {
                'generated_summary': '',
                'reference_summaries': reference_summaries,
                'num_input_texts': len(input_texts),
                'num_events': window_data['num_events'],
                'error': str(e)
            }
    
    def _clean_summary(self, text: str) -> str:
        """
        Clean generated summary by removing unnecessary whitespace and newlines.
        
        Args:
            text: Raw summary text from LLM
            
        Returns:
            Cleaned summary text
        """
        if not text:
            return ""
        
        import re
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Replace multiple newlines with single space
        text = re.sub(r'\n+', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        
        # Final cleanup
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _create_summarization_prompt(self, input_texts: List[str], structured_data_list: Optional[List[Dict]] = None) -> str:
        """
        Create prompt for summarization.
        
        Args:
            input_texts: List of input texts to summarize
            structured_data_list: Optional list of structured data
            
        Returns:
            Formatted prompt for LLM
        """
        # Create base prompt
        prompt = "Your task is to create a concise summary that covers ALL major events from the following articles.\\n\\n"
        
        # Add articles to summarize FIRST (consistent with Task1 and Task3)
        prompt += "Articles:\\n"
        for i, text in enumerate(input_texts, 1):
            prompt += f"Article {i}:\\n{text}\\n\\n"
        
        # Add structured data AFTER texts if available (consistent with Task1 format)
        if self.use_structured_data and structured_data_list:
            prompt += "Additional Context:\\n"
            for i, structured_data in enumerate(structured_data_list):
                if structured_data:
                    prompt += f"Event {i+1}:\\n"
                    prompt += self.data_loader.format_structured_data(structured_data)
                    prompt += "\\n"
            prompt += "\\n"
        
        # Add instructions
        prompt += "Instructions:\\n"
        prompt += "1. Identify ALL distinct events/topics in the articles\\n"
        prompt += "2. Briefly mention each event - don't focus on only one\\n"
        prompt += "3. Keep the summary concise and balanced\\n"
        
        if self.use_structured_data and structured_data_list:
            prompt += "4. Use the additional context to identify key entities when helpful\\n"
        else:
            prompt += "4. Focus on the most important facts\\n"
        
        prompt += "\\nSummary:"
        
        return prompt
    
    def _generate_reference_label(self, reference_summaries: List[str], window_index: int) -> str:
        """
        Generate reference label based on label_type.
        
        Args:
            reference_summaries: List of reference summaries
            window_index: Window index for caching
            
        Returns:
            Generated reference label
        """
        if not reference_summaries:
            return ""
        
        if self.label_type == "concat":
            # Simple concatenation
            return " ".join(reference_summaries)
        
        elif self.label_type == "abstract":
            # Generate abstract label using GPT-4o with caching
            return self._generate_abstract_label(reference_summaries, window_index)
        
        else:
            logging.warning(f"Unknown label_type: {self.label_type}, using concat")
            return " ".join(reference_summaries)
    
    def _generate_abstract_label(self, reference_summaries: List[str], window_index: int) -> str:
        """
        Generate an abstract label from reference summaries using GPT-4o.
        Uses file-based cache to avoid redundant API calls.
        
        Args:
            reference_summaries: List of reference summaries
            window_index: Window index for caching
            
        Returns:
            Abstract label generated by GPT-4o or from cache
        """
        # Check cache first
        cache_key = str(window_index)
        if cache_key in self.abstract_label_cache:
            self.cache_hits += 1
            cached_label = self.abstract_label_cache[cache_key]['label']
            logging.info(f"Abstract label cache HIT for window {window_index} (total hits: {self.cache_hits})")
            return cached_label
        
        self.cache_misses += 1
        logging.info(f"Abstract label cache MISS for window {window_index} (total misses: {self.cache_misses}). Generating with GPT-4o...")
        
        # Combine reference summaries
        combined_text = ' '.join(reference_summaries)
        
        # Create prompt for abstract label generation
        prompt = f"""Please create a concise, coherent summary that integrates the following event summaries. 
Maintain all key information and important keywords, but create a single cohesive summary 
with minimal edits to the original content. The summary should read as one complete text.

EVENT SUMMARIES:
{combined_text}

INTEGRATED SUMMARY:"""
        
        try:
            # Try to use OpenAI GPT-4o for abstract labels
            import os
            openai_api_key = os.getenv('OPENAI_API_KEY')
            
            if openai_api_key:
                # Create a separate LLM client for GPT-4o
                label_client = UnifiedLLMClient(
                    engine="openai",
                    model_name="gpt-4o",
                    api_key=openai_api_key
                )
                
                # Generate abstract label with GPT-4o
                logging.info("Generating abstract label with GPT-4o")
                abstract_label = label_client.generate(
                    prompt,
                    max_tokens=300,
                    temperature=0.3
                )
                
                # Save to cache file
                self._save_label_to_cache(window_index, reference_summaries, abstract_label.strip())
                
                return abstract_label.strip()
            else:
                raise ValueError("OPENAI_API_KEY environment variable is required for abstract label generation")
            
        except Exception as e:
            logging.error(f"Error generating abstract label: {e}")
            raise  # Re-raise exception instead of fallback
    
    def evaluate_results(self) -> Dict[str, Any]:
        """
        Evaluate summarization results using comprehensive metrics including EMDS.
        
        Returns:
            Dictionary containing detailed evaluation metrics and summary statistics
        """
        from evaluation.summarization_metrics import SummaryEvaluator
        
        evaluator = SummaryEvaluator()
        return evaluator.evaluate_all(
            references=self.reference_summaries,
            hypotheses=self.generated_summaries,
            topics=self.topics,
            window_indices=self.window_indices,
            event_counts=self.event_counts,
            calculate_emds=True
        )
    
    def get_summaries(self) -> Dict[str, List[str]]:
        """
        Get generated and reference summaries.
        
        Returns:
            Dictionary containing generated and reference summaries
        """
        return {
            'generated_summaries': self.generated_summaries,
            'reference_summaries': self.reference_summaries
        }
    
    def save_summaries(self, filename: Optional[str] = None):
        """
        Save generated summaries to file.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"summaries_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        summaries_data = {
            'config': self._get_config_summary(),
            'summaries': []
        }
        
        for i, (gen, ref) in enumerate(zip(self.generated_summaries, self.reference_summaries)):
            summaries_data['summaries'].append({
                'window_index': i,
                'generated_summary': gen,
                'reference_summary': ref,
                'generated_length': len(gen.split()),
                'reference_length': len(ref.split()) if ref else 0
            })
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summaries_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Summaries saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving summaries: {e}")
    
    def print_sample_summaries(self, num_samples: int = 3):
        """Print sample summaries for inspection."""
        print(f"\\n{'='*60}")
        print("SAMPLE SUMMARIES")
        print(f"{'='*60}")
        
        num_samples = min(num_samples, len(self.generated_summaries))
        
        for i in range(num_samples):
            print(f"\\n--- Window {i+1} ---")
            print(f"Generated: {self.generated_summaries[i]}")
            if i < len(self.reference_summaries) and self.reference_summaries[i]:
                print(f"Reference: {self.reference_summaries[i]}")
            else:
                print("Reference: [Not available]")
            print("-" * 40)
    
    def process_window_batch(self, windows: List[Dict[str, Any]], window_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Process a batch of windows simultaneously for improved efficiency.
        
        Args:
            windows: List of window data
            window_indices: List of window indices
            
        Returns:
            List of window processing results
        """
        if not windows:
            return []
        
        batch_results = []
        batch_prompts = []
        batch_data = []
        
        # Prepare all prompts in the batch
        for window, window_index in zip(windows, window_indices):
            window_data = self.data_loader.prepare_window_data(
                window['events'],
                include_structured_data=self.use_structured_data
            )
            
            input_texts = window_data['input_texts']
            reference_summaries = window_data['reference_summaries']
            structured_data_list = window_data.get('structured_data', [])
            
            # Apply truncation to both structured data and input texts
            # Token budget: max_input_tokens - prompt_overhead
            if input_texts:
                BASE_PROMPT_OVERHEAD = 200
                total_budget = self.max_input_tokens - BASE_PROMPT_OVERHEAD
                
                if self.use_structured_data and structured_data_list:
                    # Allocate 30% for structured data, 70% for input texts
                    structured_budget = int(total_budget * 0.3)
                    text_budget = int(total_budget * 0.7)
                    
                    # Truncate structured data if needed
                    structured_data_list = self._truncate_structured_data(
                        structured_data_list, structured_budget
                    )
                else:
                    text_budget = total_budget
                
                input_texts = self._truncate_texts_equally(input_texts, text_budget)
            
            if not input_texts:
                batch_results.append({
                    'generated_summary': '',
                    'reference_summaries': reference_summaries,
                    'num_input_texts': 0,
                    'num_events': window_data['num_events'],
                    'error': 'No input texts available'
                })
                batch_prompts.append(None)
                batch_data.append(window_data)
                continue
            
            # Create summarization prompt
            prompt = self._create_summarization_prompt(
                input_texts, 
                structured_data_list if self.use_structured_data else None
            )
            
            # Safety check: if prompt too long, truncate both structured data and input texts
            max_prompt_tokens = 7000
            estimated_tokens = len(prompt) // 2  # Conservative: 2 chars per token
            while estimated_tokens > max_prompt_tokens:
                reduction_ratio = max_prompt_tokens / estimated_tokens * 0.85
                
                if self.use_structured_data and structured_data_list:
                    sd_budget = int(sum(len(self.data_loader.format_structured_data(sd)) for sd in structured_data_list if sd) * reduction_ratio) // 2
                    structured_data_list = self._truncate_structured_data(structured_data_list, sd_budget)
                
                reduced_text_budget = int(len(''.join(input_texts)) * reduction_ratio) // 2
                input_texts = self._truncate_texts_equally(input_texts, max(reduced_text_budget, 500))
                
                prompt = self._create_summarization_prompt(
                    input_texts,
                    structured_data_list if self.use_structured_data else None
                )
                new_estimated = len(prompt) // 2
                if new_estimated >= estimated_tokens:
                    break
                estimated_tokens = new_estimated
            
            batch_prompts.append(prompt)
            batch_data.append(window_data)
            
            # Placeholder for batch processing
            batch_results.append(None)
        
        # Filter out None prompts for batch processing
        valid_indices = [i for i, p in enumerate(batch_prompts) if p is not None]
        valid_prompts = [batch_prompts[i] for i in valid_indices]
        
        if valid_prompts:
            try:
                # Generate summaries in batch
                batch_summaries = self._generate_batch_summaries(valid_prompts)
                
                # Assign results back to the correct positions
                valid_idx = 0
                for i in valid_indices:
                    window_data = batch_data[i]
                    generated_summary = batch_summaries[valid_idx] if valid_idx < len(batch_summaries) else ''
                    valid_idx += 1
                    
                    # Clean generated summary
                    generated_summary = self._clean_summary(generated_summary)
                    
                    # Generate reference label
                    reference_label = self._generate_reference_label(window_data['reference_summaries'])
                    
                    batch_results[i] = {
                        'generated_summary': generated_summary,
                        'reference_summaries': window_data['reference_summaries'],
                        'reference_label': reference_label,
                        'num_input_texts': len(window_data['input_texts']),
                        'num_events': window_data['num_events'],
                        'topic': window_data.get('topic', 'unknown')
                    }
                    
                    # Store for evaluation
                    self.generated_summaries.append(generated_summary)
                    self.reference_summaries.append(reference_label)
                    self.topics.append(window_data.get('topic', 'unknown'))
                    self.window_indices.append(window_indices[i])
                    self.event_counts.append(window_data['num_events'])
                    
            except Exception as e:
                logging.error(f"Error in batch summary generation: {e}")
                # Fallback to individual processing for remaining valid prompts
                for i in valid_indices:
                    if batch_results[i] is None:
                        batch_results[i] = {
                            'generated_summary': '',
                            'reference_summaries': batch_data[i]['reference_summaries'],
                            'num_input_texts': len(batch_data[i]['input_texts']),
                            'num_events': batch_data[i]['num_events'],
                            'error': f'Batch processing failed: {e}'
                        }
        
        return batch_results
    
    def _generate_batch_summaries(self, prompts: List[str]) -> List[str]:
        """
        Generate summaries for a batch of prompts.
        
        Args:
            prompts: List of summarization prompts
            
        Returns:
            List of generated summaries
        """
        if not prompts:
            return []
        
        # For batch processing, we concatenate prompts with separators
        # and then split the response
        batch_prompt = self._create_batch_prompt(prompts)
        
        try:
            # Generate batch response
            response = self.llm_client.generate(
                prompt=batch_prompt,
                max_tokens=self.summary_max_tokens * len(prompts) + 100,  # Extra tokens for separators
                temperature=self.temperature
            )
            
            # Parse batch response
            summaries = self._parse_batch_response(response, len(prompts))
            
            return summaries
            
        except Exception as e:
            logging.error(f"Error in batch summary generation: {e}")
            raise  # Fail fast instead of fallback
    
    def _create_batch_prompt(self, prompts: List[str]) -> str:
        """
        Create a single prompt for batch processing multiple summarization requests.
        
        Args:
            prompts: List of individual summarization prompts
            
        Returns:
            Combined batch prompt
        """
        batch_prompt = "You will be given multiple summarization tasks. For each task, provide a concise summary and separate your responses with '---SUMMARY_SEPARATOR---'.\n\n"
        
        for i, prompt in enumerate(prompts, 1):
            batch_prompt += f"Task {i}:\n{prompt}\n\n"
        
        batch_prompt += "Please provide exactly one summary for each task, separated by '---SUMMARY_SEPARATOR---'."
        
        return batch_prompt
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse batch response into individual summaries.
        
        Args:
            response: Batch response from LLM
            expected_count: Expected number of summaries
            
        Returns:
            List of individual summaries
        """
        # Split by separator
        separator = '---SUMMARY_SEPARATOR---'
        parts = response.split(separator)
        
        summaries = []
        for part in parts:
            summary = part.strip()
            if summary:
                summaries.append(summary)
        
        # Ensure we have the expected number of summaries
        while len(summaries) < expected_count:
            summaries.append('')  # Empty summary for missing ones
        
        return summaries[:expected_count]
    
    def _generate_individual_summaries(self, prompts: List[str]) -> List[str]:
        """
        Fallback method to generate summaries individually.
        
        Args:
            prompts: List of summarization prompts
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for prompt in prompts:
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=self.summary_max_tokens,
                    temperature=self.temperature
                )
                summaries.append(response.strip())
            except Exception as e:
                logging.error(f"Error generating individual summary: {e}")
                raise  # Don't append empty string, just fail
        
        return summaries
    
    def run(self) -> Dict[str, Any]:
        """
        Run summarization task with optional batch processing.
        
        Returns:
            Task execution results
        """
        logging.info(f"Starting summarization task with batch_size={self.batch_size}")
        
        # Get all windows
        windows = list(self.window_manager.create_windows(self.data_loader.get_all_events()))
        total_windows = len(windows)
        
        logging.info(f"Processing {total_windows} windows with batch size {self.batch_size}")
        
        # Process windows in batches with progress bar
        for batch_start in tqdm(range(0, total_windows, self.batch_size), desc="Processing summarization windows"):
            batch_end = min(batch_start + self.batch_size, total_windows)
            batch_windows = windows[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            if self.batch_size == 1:
                # Use original single-window processing
                for window, window_index in zip(batch_windows, batch_indices):
                    result = self.process_window(window, window_index)
                    logging.info(f"Processed window {window_index + 1}/{total_windows}")
            else:
                # Use batch processing
                batch_results = self.process_window_batch(batch_windows, batch_indices)
                logging.info(f"Processed batch {batch_start//self.batch_size + 1} (windows {batch_start+1}-{batch_end})")
        
        # Compile results
        results = {
            'task': 'summarization',
            'total_windows': total_windows,
            'batch_size': self.batch_size,
            'total_summaries': len(self.generated_summaries),
            'config': self.config
        }
        
        # Save results if requested
        if self.save_results:
            self.save_summaries()
        
        logging.info(f"Summarization task completed: {len(self.generated_summaries)} summaries generated")
        return results
    
    def _truncate_texts_equally(self, texts: List[str], max_tokens: int) -> List[str]:
        """
        Truncate texts by distributing available tokens equally among all texts.
        This ensures all documents contribute some information (user's preferred approach).
        
        Args:
            texts: List of text strings
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            List of equally truncated texts
        """
        if not texts:
            return []
        
        # Estimate tokens (conservative: 2 chars ≈ 1 token for mixed content)
        def estimate_tokens(text):
            return len(text) // 2
        
        # Calculate total tokens
        total_tokens = sum(estimate_tokens(text) for text in texts)
        
        # If already under limit, return as is
        if total_tokens <= max_tokens:
            return texts
        
        # Distribute tokens equally among all texts
        tokens_per_text = max_tokens // len(texts)
        remaining_tokens = max_tokens % len(texts)
        
        result = []
        for i, text in enumerate(texts):
            # Give some texts one extra token from remainder
            allocated_tokens = tokens_per_text + (1 if i < remaining_tokens else 0)
            
            # Convert tokens to characters (2 chars ≈ 1 token)
            max_chars = allocated_tokens * 2
            
            if estimate_tokens(text) <= allocated_tokens:
                # Text fits within allocation
                result.append(text)
            else:
                # Truncate text to allocation
                truncated = self._smart_truncate(text, max_chars)
                result.append(truncated)
        
        return result
    
    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """
        Smartly truncate text to max_chars, trying to break at sentence or word boundaries.
        
        Args:
            text: Text to truncate
            max_chars: Maximum number of characters
            
        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text
        
        # Try to break at sentence boundary
        truncated = text[:max_chars]
        sentence_break = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        
        if sentence_break > max_chars * 0.7:  # If we found a good sentence break point
            return text[:sentence_break+1]
        
        # Try to break at word boundary
        word_break = truncated.rfind(' ')
        if word_break > 0:
            return text[:word_break]
        
        # Last resort: hard truncate
        return truncated + '...'
    
    def _truncate_structured_data(self, structured_data_list: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Truncate structured data to fit within token budget.
        Keeps essential fields but limits content size.
        
        Args:
            structured_data_list: List of structured data dicts
            max_tokens: Maximum tokens allowed for all structured data
            
        Returns:
            Truncated structured data list
        """
        if not structured_data_list:
            return []
        
        import json
        
        # Calculate current size
        total_chars = 0
        for sd in structured_data_list:
            if sd:
                total_chars += len(self.data_loader.format_structured_data(sd))
        
        current_tokens = total_chars // 2
        
        # If within budget, return as-is
        if current_tokens <= max_tokens:
            return structured_data_list
        
        # Need to truncate - limit chars per event
        max_chars_total = max_tokens * 2
        chars_per_event = max_chars_total // len(structured_data_list)
        
        truncated_list = []
        for sd in structured_data_list:
            if not sd:
                truncated_list.append(sd)
                continue
            
            # Truncate each field's content
            truncated_sd = {}
            for key, value in sd.items():
                if isinstance(value, str) and len(value) > chars_per_event // 4:
                    truncated_sd[key] = value[:chars_per_event // 4] + '...'
                elif isinstance(value, list) and len(str(value)) > chars_per_event // 4:
                    # Keep only first few items
                    truncated_sd[key] = value[:3] if len(value) > 3 else value
                else:
                    truncated_sd[key] = value
            
            truncated_list.append(truncated_sd)
        
        return truncated_list
