"""
Task-3: Temporal QA
Implements sliding window temporal question answering with structured data support.
"""

import json
import logging
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from .base_task import BaseTask
from core.data_loader import QADataLoader


class TemporalQATask(BaseTask):
    """
    Temporal question answering task using sliding windows.
    Supports structured data integration in QA prompts.
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
                 qa_max_tokens: int = 20,
                 temperature: float = 0.0,
                 sample_qa_pairs: bool = True,
                 max_qa_pairs_per_window: int = 10,
                 window_mode: str = "topic-mix",
                 batch_size: int = 1,
                 **kwargs):
        """
        Initialize temporal QA task.
        
        Args:
            qa_max_tokens: Maximum tokens for QA responses
            temperature: LLM temperature for QA
            sample_qa_pairs: Whether to sample QA pairs if too many
            max_qa_pairs_per_window: Maximum QA pairs to process per window
            window_mode: Window mode - 'topic' (per-topic) or 'topic-mix' (mixed topics)
        """
        super().__init__(
            task_name="temporal_qa",
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
        
        self.qa_max_tokens = qa_max_tokens
        self.temperature = temperature
        self.sample_qa_pairs = sample_qa_pairs
        self.max_qa_pairs_per_window = max_qa_pairs_per_window
        self.window_mode = window_mode
        self.batch_size = batch_size
        
        # Set max_input_tokens based on model capacity and texts_per_event
        self.max_input_tokens = self._calculate_max_input_tokens(
            llm_client, 
            max_input_tokens, 
            texts_per_event
        )
        
        # Initialize QA data loader
        self.qa_data_loader = QADataLoader(
            qa_file=data_file  # Fixed: json_file -> qa_file, removed unsupported random_seed
        )
        
        # Set random seed for reproducible sampling
        random.seed(self.random_seed)
        
        # Storage for evaluation
        self.qa_results = []
        self.correct_answers = 0
        self.total_questions = 0
        
        # Update config
        self.config.update({
            'qa_max_tokens': qa_max_tokens,
            'temperature': temperature,
            'sample_qa_pairs': sample_qa_pairs,
            'max_qa_pairs_per_window': max_qa_pairs_per_window,
            'batch_size': batch_size
        })
        
        logging.info(f"Initialized temporal QA task with {len(self.qa_data_loader.get_all_qa_pairs())} QA pairs")
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
        
        # Cap at 80% of model limit (leave room for structured data and overhead)
        max_allowed = int(model_limit * 0.8)
        final_tokens = min(desired_tokens, max_allowed)
        
        logging.info(f"Model: {model_name}, Limit: {model_limit}, texts_per_event: {texts_per_event}, "
                    f"Scale: {scale}x, Desired: {desired_tokens}, Final: {final_tokens}")
        
        return final_tokens
    
    def process_window(self, window: Dict[str, Any], window_index: int) -> Dict[str, Any]:
        """
        Process a single window for temporal QA.
        
        Args:
            window: Window data
            window_index: Window index
            
        Returns:
            Window processing results
        """
        # Get QA pairs and events for this window
        qa_pairs = self.qa_data_loader.get_qa_pairs_for_window(window['events'])
        
        if not qa_pairs:
            return {
                'qa_results': [],
                'accuracy': 0.0,
                'num_questions': 0,
                'num_events': len(window['events']),
                'error': 'No QA pairs available for this window'
            }
        
        # Sample QA pairs if too many
        if self.sample_qa_pairs and len(qa_pairs) > self.max_qa_pairs_per_window:
            qa_pairs = random.sample(qa_pairs, self.max_qa_pairs_per_window)
        
        window_results = []
        window_correct = 0
        
        for qa_pair in qa_pairs:
            try:
                # Get relevant articles for this QA pair
                articles = self._get_relevant_articles(qa_pair, window['events'])
                
                if not articles:
                    continue
                
                # Get structured data if enabled
                structured_data = None
                if self.use_structured_data:
                    structured_data = self._get_structured_data_for_qa(qa_pair, window['events'])
                
                # Create QA prompt
                prompt = self._create_qa_prompt(qa_pair, articles, structured_data)
                
                # Get LLM response
                response = self.llm_client.generate(
                    prompt,
                    max_tokens=self.qa_max_tokens,
                    temperature=self.temperature
                )
                
                # Parse and evaluate response
                predicted_answer = self._parse_qa_response(response)
                correct_answer = qa_pair.get('answer', '')
                is_correct = predicted_answer.upper() == correct_answer.upper()
                
                if is_correct:
                    window_correct += 1
                    self.correct_answers += 1
                
                self.total_questions += 1
                
                # Store result
                qa_result = {
                    'question': qa_pair.get('question', ''),
                    'choices': qa_pair.get('choices', {}),
                    'correct_answer': correct_answer,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'confidence': self._extract_confidence(response),
                    'response': response.strip(),
                    'structured_data_used': bool(self.use_structured_data and structured_data)
                }
                
                window_results.append(qa_result)
                self.qa_results.append(qa_result)
                
            except Exception as e:
                logging.error(f"Error processing QA pair in window {window_index}: {e}")
                continue
        
        # Calculate window accuracy
        window_accuracy = window_correct / len(window_results) if window_results else 0.0
        
        return {
            'qa_results': window_results,
            'accuracy': window_accuracy,
            'num_questions': len(window_results),
            'num_correct': window_correct,
            'num_events': len(window['events']),
            'structured_data_used': self.use_structured_data
        }
    
    def _get_relevant_articles(self, qa_pair: Dict, events: List[Dict]) -> List[str]:
        """
        Get relevant articles for a QA pair from window events.
        Only includes events up to and including the question date (no future leakage).
        
        Args:
            qa_pair: QA pair dictionary
            events: List of events in window
            
        Returns:
            List of relevant articles
        """
        articles = []
        
        # Get question date for temporal filtering
        qa_date = qa_pair.get('event_date', '')
        
        # Collect articles from events using texts_per_event sampling
        # Filter to only include events up to question date (no future leakage)
        for event in events:
            event_date = event.get('event_date', 'Unknown date')
            
            # Skip events after the question date to prevent temporal leakage
            if qa_date and event_date > qa_date:
                continue
            
            # Use data_loader to sample texts according to texts_per_event setting
            sampled_texts = self.data_loader.sample_event_texts(event, num_texts=self.texts_per_event)
            
            # Add date information to each sampled text
            for text in sampled_texts:
                # Prepend date to article text
                dated_text = f"[Date: {event_date}]\n{text}"
                articles.append(dated_text)
        
        # Apply equal distribution truncation to articles
        if articles:
            articles = self._truncate_texts_equally(articles, self.max_input_tokens)
        
        return articles
    
    def _get_structured_data_for_qa(self, qa_pair: Dict, events: List[Dict]) -> Optional[List[Dict]]:
        """
        Get structured data relevant to QA pair, organized by date.
        Only includes events up to and including the question date (no future leakage).
        
        Args:
            qa_pair: QA pair dictionary
            events: List of events in window
            
        Returns:
            List of date-organized structured data dictionaries or None
        """
        # Get question date for temporal filtering
        qa_date = qa_pair.get('event_date', '')
        
        # Collect structured data per date (preserve temporal information)
        dated_structured_data = []
        
        # Sort events by date for chronological ordering
        sorted_events = sorted(events, key=lambda x: x.get('event_date', ''))
        
        for event in sorted_events:
            event_date = event.get('event_date', '')
            
            # Skip events after the question date to prevent temporal leakage
            if qa_date and event_date > qa_date:
                continue
            
            structured_data = self.data_loader._extract_structured_data(event)
            if structured_data:
                dated_entry = {
                    'date': event_date,
                    'data': structured_data
                }
                dated_structured_data.append(dated_entry)
        
        return dated_structured_data if dated_structured_data else None
    
    def _format_dated_structured_data(self, dated_structured_data: List[Dict]) -> str:
        """
        Format date-organized structured data for inclusion in prompts.
        
        Args:
            dated_structured_data: List of {'date': ..., 'data': {...}} dictionaries
            
        Returns:
            Formatted string with structured data organized by date
        """
        formatted_parts = []
        
        for entry in dated_structured_data:
            date = entry.get('date', 'Unknown date')
            data = entry.get('data', {})
            
            part = f"[{date}]"
            
            # Format each field
            for field, value in data.items():
                if field == 'Event Attributes' and isinstance(value, dict):
                    # Format Event Attributes specially
                    attrs = ', '.join(f"{k}: {v}" for k, v in value.items() if v)
                    if attrs:
                        part += f"\n- {field}: {attrs}"
                elif field == 'Relations' and isinstance(value, list):
                    # Skip Relations for brevity (or format if needed)
                    continue
                elif isinstance(value, list):
                    if value:
                        part += f"\n- {field}: {', '.join(str(v) for v in value)}"
                else:
                    if value:
                        part += f"\n- {field}: {value}"
            
            formatted_parts.append(part)
        
        return "\n\n".join(formatted_parts)
    
    def _create_qa_prompt(self, qa_pair: Dict, articles: List[str], structured_data: Optional[List[Dict]] = None) -> str:
        """
        Create QA prompt with articles and optional structured data.
        
        Args:
            qa_pair: QA pair dictionary
            articles: List of relevant articles
            structured_data: Optional list of date-organized structured data
            
        Returns:
            Formatted QA prompt
        """
        prompt = "Your task is to answer the following question based on the provided articles.\\n\\n"
        
        # Add structured data if available (now organized by date)
        if self.use_structured_data and structured_data:
            prompt += "Additional Context (by date):\\n"
            prompt += self._format_dated_structured_data(structured_data)
            prompt += "\\n\\n"
        
        # Add articles
        prompt += "Articles:\\n"
        for i, article in enumerate(articles, 1):
            prompt += f"Article {i}:\\n{article}\\n\\n"
        
        # Add question and choices
        question = qa_pair.get('question', '')
        choices = qa_pair.get('choices', {})
        
        prompt += f"Question: {question}\\n\\n"
        
        if choices:
            prompt += "Choices:\\n"
            for choice_key, choice_text in choices.items():
                prompt += f"{choice_key}. {choice_text}\\n"
        
        prompt += "\\nInstructions:\\n"
        prompt += "1. Read all articles carefully\\n"
        prompt += "2. Use the provided information to answer the question\\n"
        
        if self.use_structured_data and structured_data:
            prompt += "3. Use the additional context when helpful\\n"
        
        prompt += "4. Choose one of the options above: a, b, c, or d\\n"
        prompt += "5. Respond with ONLY the letter (no numbers, no explanations)\\n"
        
        prompt += "\\nAnswer: "
        
        return prompt
    
    def _parse_qa_response(self, response: str) -> str:
        """
        Parse LLM response to extract answer.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed answer
        """
        response = response.strip().upper()
        
        # Look for answer at the beginning of response (most reliable)
        if response and response[0] in ['A', 'B', 'C', 'D']:
            return response[0]
        
        # Look for "Answer: X" pattern
        import re
        answer_match = re.search(r'ANSWER:\s*([ABCD])', response)
        if answer_match:
            return answer_match.group(1)
        
        # Look for standalone letter with word boundary  
        for char in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']:
            if re.search(rf'\b{char}\b', response):
                return char
        
        # Map numbers to letters (common model behavior)
        number_map = {'1': 'a', '2': 'b', '3': 'c', '4': 'd'}
        if response and response[0] in number_map:
            return number_map[response[0]]
        
        # Look for number patterns like "1)", "1.", "1:"
        number_match = re.search(r'^([1234])[.):\s]', response)
        if number_match:
            return number_map[number_match.group(1)]
        
        # If no clear choice found, return first character if it's valid
        if response and response[0] in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']:
            return response[0]
        
        # Final fallback
        return 'A'
    
    def _extract_confidence(self, response: str) -> Optional[float]:
        """
        Extract confidence score from response if available.
        
        Args:
            response: LLM response
            
        Returns:
            Confidence score or None
        """
        # Simple confidence extraction - can be enhanced
        # For now, return None as most responses won't have explicit confidence
        return None
    
    def evaluate_results(self) -> Dict[str, Any]:
        """
        Evaluate temporal QA results.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        from evaluation import QAEvaluator
        
        evaluator = QAEvaluator()
        return evaluator.evaluate(self.qa_results)
    
    def _classify_question_type(self, question: str) -> str:
        """
        Classify question type based on question text.
        
        Args:
            question: Question text
            
        Returns:
            Question type classification
        """
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['when', 'date', 'time', 'year']):
            return 'temporal'
        elif any(word in question_lower for word in ['who', 'person', 'people']):
            return 'person'
        elif any(word in question_lower for word in ['where', 'location', 'place']):
            return 'location'
        elif any(word in question_lower for word in ['what', 'which']):
            return 'factual'
        elif any(word in question_lower for word in ['why', 'how']):
            return 'causal'
        else:
            return 'other'
    
    def get_qa_results(self) -> List[Dict]:
        """Get all QA results."""
        return self.qa_results
    
    def save_qa_results(self, filename: Optional[str] = None):
        """
        Save QA results to file.
        
        Args:
            filename: Optional custom filename
        """
        if filename is None:
            filename = "qa_detailed_results.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        qa_data = {
            'config': self._get_config_summary(),
            'evaluation_metrics': self.evaluate_results(),
            'qa_results': self.qa_results
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"QA results saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving QA results: {e}")
    
    def print_sample_results(self, num_samples: int = 3):
        """Print sample QA results for inspection."""
        print(f"\\n{'='*80}")
        print("SAMPLE QA RESULTS")
        print(f"{'='*80}")
        
        num_samples = min(num_samples, len(self.qa_results))
        
        for i in range(num_samples):
            result = self.qa_results[i]
            print(f"\\n--- Question {i+1} ---")
            print(f"Question: {result['question']}")
            
            choices = result.get('choices', {})
            if choices:
                print("Choices:")
                for key, value in choices.items():
                    print(f"  {key}. {value}")
            
            print(f"Correct Answer: {result['correct_answer']}")
            print(f"Predicted Answer: {result['predicted_answer']}")
            print(f"Correct: {'✓' if result['is_correct'] else '✗'}")
            print(f"Structured Data Used: {'Yes' if result.get('structured_data_used', False) else 'No'}")
            print("-" * 60)
    
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
        batch_qa_data = []
        
        # Prepare all QA data in the batch
        for window, window_index in zip(windows, window_indices):
            # Get QA pairs and events for this window
            qa_pairs = self.qa_data_loader.get_qa_pairs_for_window(window['events'])
            
            if not qa_pairs:
                batch_results.append({
                    'qa_results': [],
                    'accuracy': 0.0,
                    'num_questions': 0,
                    'num_events': len(window['events']),
                    'error': 'No QA pairs available for this window'
                })
                batch_qa_data.append([])
                continue
            
            # Sample QA pairs if too many
            if self.sample_qa_pairs and len(qa_pairs) > self.max_qa_pairs_per_window:
                qa_pairs = random.sample(qa_pairs, self.max_qa_pairs_per_window)
            
            # Prepare QA data for batch processing
            window_qa_data = []
            for qa_pair in qa_pairs:
                # Get relevant articles
                articles = self._get_relevant_articles(qa_pair, window['events'])
                if articles:
                    articles = self._truncate_texts_equally(articles, self.max_input_tokens)
                
                # Get structured data if enabled
                structured_data = None
                if self.use_structured_data:
                    structured_data = self._get_structured_data_for_qa(qa_pair, window['events'])
                
                window_qa_data.append({
                    'qa_pair': qa_pair,
                    'articles': articles,
                    'structured_data': structured_data
                })
            
            batch_qa_data.append(window_qa_data)
            batch_results.append(None)  # Placeholder
        
        # Process each window's QA pairs in batch
        for window_idx, (window_qa_data, window_index) in enumerate(zip(batch_qa_data, window_indices)):
            if not window_qa_data:
                continue  # Already handled empty case above
            
            if self.batch_size == 1:
                # Use original single processing
                window_results = []
                window_correct = 0
                
                for qa_data in window_qa_data:
                    try:
                        result = self._process_single_qa(qa_data)
                        window_results.append(result)
                        if result['is_correct']:
                            window_correct += 1
                            self.correct_answers += 1
                        self.total_questions += 1
                        self.qa_results.append(result)
                    except Exception as e:
                        logging.error(f"Error processing QA pair: {e}")
                        error_result = {
                            'question': qa_data['qa_pair'].get('question', ''),
                            'predicted_answer': '',
                            'correct_answer': qa_data['qa_pair'].get('answer', ''),
                            'is_correct': False,
                            'structured_data_used': False,
                            'error': str(e)
                        }
                        window_results.append(error_result)
                        self.total_questions += 1
                        self.qa_results.append(error_result)
                
                batch_results[window_idx] = {
                    'qa_results': window_results,
                    'accuracy': window_correct / len(window_results) if window_results else 0.0,
                    'num_questions': len(window_results),
                    'num_events': len(windows[window_idx]['events'])
                }
            else:
                # Use batch processing for QA pairs
                try:
                    batch_answers = self._process_qa_batch(window_qa_data)
                    window_results = []
                    window_correct = 0
                    
                    for qa_data, predicted_answer in zip(window_qa_data, batch_answers):
                        qa_pair = qa_data['qa_pair']
                        correct_answer = qa_pair.get('answer', '')
                        is_correct = self._evaluate_answer(predicted_answer, correct_answer)
                        
                        result = {
                            'question': qa_pair.get('question', ''),
                            'predicted_answer': predicted_answer,
                            'correct_answer': correct_answer,
                            'is_correct': is_correct,
                            'structured_data_used': qa_data['structured_data'] is not None,
                            'choices': qa_pair.get('choices', {}),
                            'topic': qa_pair.get('topic', 'unknown'),
                            'date': qa_pair.get('date', '')
                        }
                        
                        window_results.append(result)
                        if is_correct:
                            window_correct += 1
                            self.correct_answers += 1
                        self.total_questions += 1
                        self.qa_results.append(result)
                    
                    batch_results[window_idx] = {
                        'qa_results': window_results,
                        'accuracy': window_correct / len(window_results) if window_results else 0.0,
                        'num_questions': len(window_results),
                        'num_events': len(windows[window_idx]['events'])
                    }
                    
                except Exception as e:
                    logging.error(f"Error in batch QA processing: {e}")
                    # Fallback to individual processing
                    batch_results[window_idx] = self.process_window(windows[window_idx], window_index)
        
        return batch_results
    
    def _process_single_qa(self, qa_data: Dict) -> Dict:
        """
        Process a single QA pair (used in both batch and individual processing).
        
        Args:
            qa_data: Dictionary containing qa_pair, articles, and structured_data
            
        Returns:
            QA result dictionary
        """
        qa_pair = qa_data['qa_pair']
        articles = qa_data['articles']
        structured_data = qa_data['structured_data']
        
        if not articles:
            return {
                'question': qa_pair.get('question', ''),
                'predicted_answer': '',
                'correct_answer': qa_pair.get('answer', ''),
                'is_correct': False,
                'structured_data_used': False,
                'error': 'No relevant articles found'
            }
        
        # Create QA prompt
        prompt = self._create_qa_prompt(qa_pair, articles, structured_data)
        
        # Generate answer
        response = self.llm_client.generate(
            prompt=prompt,
            max_tokens=self.qa_max_tokens,
            temperature=self.temperature
        )
        
        # Parse response
        predicted_answer = self._parse_qa_response(response)
        correct_answer = qa_pair.get('answer', '')
        is_correct = self._evaluate_answer(predicted_answer, correct_answer)
        
        return {
            'question': qa_pair.get('question', ''),
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'structured_data_used': structured_data is not None,
            'choices': qa_pair.get('choices', {}),
            'topic': qa_pair.get('topic', 'unknown'),
            'date': qa_pair.get('date', '')
        }
    
    def _process_qa_batch(self, qa_data_list: List[Dict]) -> List[str]:
        """
        Process a batch of QA pairs simultaneously.
        
        Args:
            qa_data_list: List of QA data dictionaries
            
        Returns:
            List of predicted answers
        """
        if not qa_data_list:
            return []
        
        # Create batch prompt
        batch_prompt = self._create_batch_qa_prompt(qa_data_list)
        
        try:
            # Generate batch response
            response = self.llm_client.generate(
                prompt=batch_prompt,
                max_tokens=self.qa_max_tokens * len(qa_data_list) + 100,
                temperature=self.temperature
            )
            
            # Parse batch response
            answers = self._parse_batch_qa_response(response, len(qa_data_list))
            
            return answers
            
        except Exception as e:
            logging.error(f"Error in batch QA generation: {e}")
            raise  # Fail fast instead of fallback
    
    def _create_batch_qa_prompt(self, qa_data_list: List[Dict]) -> str:
        """
        Create a single prompt for batch processing multiple QA requests.
        
        Args:
            qa_data_list: List of QA data dictionaries
            
        Returns:
            Combined batch prompt
        """
        batch_prompt = "You will be given multiple question-answering tasks. For each task, provide a short answer and separate your responses with '---QA_SEPARATOR---'.\n\n"
        
        for i, qa_data in enumerate(qa_data_list, 1):
            qa_pair = qa_data['qa_pair']
            articles = qa_data['articles']
            structured_data = qa_data['structured_data']
            
            # Create individual prompt
            individual_prompt = self._create_qa_prompt(qa_pair, articles, structured_data)
            
            batch_prompt += f"Task {i}:\n{individual_prompt}\n\n"
        
        batch_prompt += "Please provide exactly one answer for each task, separated by '---QA_SEPARATOR---'."
        
        return batch_prompt
    
    def _parse_batch_qa_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse batch response into individual answers.
        
        Args:
            response: Batch response from LLM
            expected_count: Expected number of answers
            
        Returns:
            List of individual answers
        """
        # Split by separator
        separator = '---QA_SEPARATOR---'
        parts = response.split(separator)
        
        answers = []
        for part in parts:
            answer = part.strip()
            if answer:
                # Parse the individual answer
                parsed_answer = self._parse_qa_response(answer)
                answers.append(parsed_answer)
        
        # Ensure we have the expected number of answers
        while len(answers) < expected_count:
            answers.append('')  # Empty answer for missing ones
        
        return answers[:expected_count]
    
    def _process_qa_individually(self, qa_data_list: List[Dict]) -> List[str]:
        """
        Fallback method to process QA pairs individually.
        
        Args:
            qa_data_list: List of QA data dictionaries
            
        Returns:
            List of predicted answers
        """
        answers = []
        
        for qa_data in qa_data_list:
            try:
                result = self._process_single_qa(qa_data)
                answers.append(result['predicted_answer'])
            except Exception as e:
                logging.error(f"Error processing individual QA: {e}")
                raise  # Don't append empty answer, just fail
        
        return answers
    
    def _evaluate_answer(self, predicted: str, correct: str) -> bool:
        """
        Evaluate if predicted answer matches correct answer.
        
        Args:
            predicted: Predicted answer
            correct: Correct answer
            
        Returns:
            True if answers match
        """
        if not predicted or not correct:
            return False
        
        # Simple exact match (can be enhanced with fuzzy matching)
        return predicted.strip().lower() == correct.strip().lower()
    
    def run(self) -> Dict[str, Any]:
        """
        Run temporal QA task with optional batch processing.
        
        Returns:
            Task execution results
        """
        logging.info(f"Starting temporal QA task with batch_size={self.batch_size}")
        logging.info(f"use_structured_data = {self.use_structured_data}")
        
        # Get all windows using QA data
        qa_events = self.qa_data_loader.get_qa_events()
        windows = list(self.window_manager.create_windows(qa_events))
        total_windows = len(windows)
        
        logging.info(f"Processing {total_windows} windows with batch size {self.batch_size}")
        
        # Process windows with progress bar
        window_progress = tqdm(total=total_windows, desc="Processing QA windows", unit="window")
        
        for batch_start in range(0, total_windows, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_windows)
            batch_windows = windows[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            if self.batch_size == 1:
                # Use original single-window processing
                for window, window_index in zip(batch_windows, batch_indices):
                    result = self.process_window(window, window_index)
                    window_progress.update(1)  # Update progress bar
                    window_progress.set_postfix(window=f"{window_index + 1}/{total_windows}")
            else:
                # Use batch processing
                batch_results = self.process_window_batch(batch_windows, batch_indices)
                window_progress.update(len(batch_windows))  # Update by number of windows processed
                window_progress.set_postfix(windows=f"{batch_end}/{total_windows}")
        
        window_progress.close()  # Close progress bar
        
        # Compile results
        accuracy = self.correct_answers / self.total_questions if self.total_questions > 0 else 0.0
        
        # Update self.results for BaseTask summary
        self.results.update({
            'execution_info': {
                'mode': 'sliding',
                'data_statistics': {
                    'total_events': len(qa_events),
                    'total_topics': 1,  # QA data doesn't have topic structure
                }
            },
            'windows': [{'window_index': i} for i in range(total_windows)],  # Add window info for summary
            'accuracy': accuracy,
            'total_questions': self.total_questions,
            'correct_answers': self.correct_answers
        })
        
        results = {
            'task': 'temporal_qa',
            'total_windows': total_windows,
            'batch_size': self.batch_size,
            'total_questions': self.total_questions,
            'correct_answers': self.correct_answers,
            'accuracy': accuracy,
            'config': self.config
        }
        
        # Save results if requested
        if self.save_results_flag:
            self.save_qa_results()
        
        logging.info(f"Temporal QA task completed: {self.correct_answers}/{self.total_questions} correct (accuracy: {accuracy:.3f})")
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
        
        # Estimate tokens (rough approximation: 4 chars ≈ 1 token)
        def estimate_tokens(text):
            return len(text) // 4
        
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
            
            # Convert tokens to characters (4 chars ≈ 1 token)
            max_chars = allocated_tokens * 4
            
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
