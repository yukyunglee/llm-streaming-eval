"""
Unified data loader for all tasks in the streaming evaluation pipeline.
Handles JSON loading, preprocessing, and data organization.
"""

import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict


class UnifiedDataLoader:
    """
    Unified data loader for Task-1, Task-2, and Task-3.
    Handles JSON loading, date parsing, text sampling, and preprocessing.
    """
    
    def __init__(self, 
                 json_file: str,
                 random_seed: int = 42,
                 max_input_tokens: int = 4000,
                 texts_per_event: int = 1,
                 structured_keys: Optional[List[str]] = None,
                 exclude_fields: Optional[List[str]] = None):
        """
        Initialize the unified data loader.
        
        Args:
            json_file: Path to JSON file containing event data
            random_seed: Random seed for reproducibility
            max_input_tokens: Maximum number of tokens for input text
            texts_per_event: Number of texts to sample from each event
            structured_keys: Keys to extract structured information
            exclude_fields: Fields to exclude from structured data (for ablation studies)
        """
        self.json_file = json_file
        self.random_seed = random_seed
        self.max_input_tokens = max_input_tokens
        self.texts_per_event = texts_per_event
        self.structured_keys = structured_keys or []
        self.exclude_fields = exclude_fields or []
        
        self.data = None
        self.topic_events = defaultdict(list)  # events organized by topic
        self.all_events = []  # all events mixed together
        
        # Set random seed
        random.seed(self.random_seed)
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and parse JSON data."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Process events
            self._process_events()
            
            logging.info(f"Data loading completed: {len(self.topic_events)} topics, "
                        f"{len(self.all_events)} total events")
                        
        except Exception as e:
            logging.error(f"Error loading data from {self.json_file}: {e}")
            raise
    
    def _process_events(self):
        """Process events from JSON data."""
        for topic in self.data.get('topics', []):
            topic_title = topic.get('topic_title', 'Unknown Topic')
            
            for event in topic.get('events', []):
                # Parse date
                try:
                    event_date = event.get('event_date', '')
                    parsed_date = datetime.strptime(event_date, '%Y-%m-%d')
                    event['parsed_date'] = parsed_date
                    event['date'] = event_date  # Keep original string format
                    
                    # Add topic information
                    event['topic_title'] = topic_title
                    
                    # Add to collections
                    self.topic_events[topic_title].append(event)
                    self.all_events.append(event)
                    
                except ValueError:
                    logging.warning(f"Date parsing failed for event: {event.get('event_date')}")
        
        # Sort events by date
        for topic, events in self.topic_events.items():
            self.topic_events[topic] = sorted(events, key=lambda x: x.get('parsed_date', datetime.min))
        
        self.all_events = sorted(self.all_events, key=lambda x: x.get('parsed_date', datetime.min))
    
    def get_events_by_topic(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get events organized by topic."""
        return dict(self.topic_events)
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all events mixed together, sorted by date."""
        return self.all_events
    
    def get_documents_for_clustering(self) -> Tuple[List[str], List[int], List[str], List[Any]]:
        """
        Get documents formatted for Task-1 clustering.
        
        Returns:
            Tuple of (documents, true_clusters, event_dates, structured_infos)
        """
        documents, true_clusters, event_dates, structured_infos = [], [], [], []
        event_id = 0
        
        for topic in self.data.get('topics', []):
            for event in topic.get('events', []):
                date = event.get('event_date', '')
                event_texts = event.get('event_text', [])
                
                for text in event_texts:
                    documents.append(text)
                    true_clusters.append(event_id)
                    event_dates.append(date)
                    
                    # Extract structured information if requested
                    if self.structured_keys:
                        structured_info = {
                            k: event.get(k, {} if k == "Event Attributes" else [])
                            for k in self.structured_keys
                        }
                        structured_infos.append(structured_info)
                    else:
                        structured_infos.append(None)
                
                event_id += 1
        
        # Sort by date
        combined = list(zip(documents, true_clusters, event_dates, structured_infos))
        combined.sort(key=lambda x: datetime.strptime(x[2], "%Y-%m-%d"))
        
        return zip(*combined)
    
    def sample_event_texts(self, event: Dict[str, Any], num_texts: Optional[int] = None) -> List[str]:
        """
        Sample texts from an event.
        
        Args:
            event: Event dictionary
            num_texts: Number of texts to sample (defaults to self.texts_per_event)
            
        Returns:
            List of sampled texts
        """
        if num_texts is None:
            num_texts = self.texts_per_event
            
        event_texts = event.get('event_text', [])
        
        if not event_texts:
            return []
        
        if num_texts >= len(event_texts):
            return event_texts
        
        return random.sample(event_texts, num_texts)
    
    def truncate_texts(self, texts: List[str], max_tokens: Optional[int] = None) -> List[str]:
        """
        Truncate texts to fit within token limit.
        
        Args:
            texts: List of text strings
            max_tokens: Maximum tokens (defaults to self.max_input_tokens)
            
        Returns:
            List of truncated texts
        """
        if max_tokens is None:
            max_tokens = self.max_input_tokens
            
        if not texts:
            return []
        
        def estimate_tokens(text: str) -> int:
            """Rough token estimation: ~4 chars per token."""
            return len(text) // 4
        
        # Try to preserve as many complete texts as possible
        total_tokens = 0
        truncated_texts = []
        
        for text in texts:
            text_tokens = estimate_tokens(text)
            
            if total_tokens + text_tokens <= max_tokens:
                # Add complete text
                truncated_texts.append(text)
                total_tokens += text_tokens
            else:
                # Truncate this text to fit remaining budget
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 0:
                    # Rough character limit based on remaining tokens
                    char_limit = remaining_tokens * 4
                    truncated_text = text[:char_limit]
                    if truncated_text:
                        truncated_texts.append(truncated_text)
                break
        
        return truncated_texts
    
    def prepare_window_data(self, events: List[Dict[str, Any]], include_structured_data: bool = False) -> Dict[str, Any]:
        """
        Prepare window data for processing.
        
        Args:
            events: List of event dictionaries
            include_structured_data: Whether to include structured data in preparation
            
        Returns:
            Dictionary containing input texts, reference summaries, and structured data
        """
        input_texts = []
        reference_summaries = []
        structured_data_list = []
        
        for event in events:
            # Sample texts from event
            event_texts = self.sample_event_texts(event)
            
            # Add date information to each text (like Task3)
            event_date = event.get('event_date') or event.get('date', 'Unknown date')
            dated_texts = [f"[Date: {event_date}]\\n{text}" for text in event_texts]
            input_texts.extend(dated_texts)
            
            # Add reference summary if available
            event_sum = event.get('event_sum', '')
            if event_sum:
                reference_summaries.append(event_sum)
            
            # Add structured data if requested (append None to maintain event order)
            if include_structured_data:
                structured_data = self._extract_structured_data(event)
                structured_data_list.append(structured_data)  # Include None to maintain alignment
        
        result = {
            'input_texts': input_texts,
            'reference_summaries': reference_summaries,
            'num_events': len(events)
        }
        
        if include_structured_data:
            result['structured_data'] = structured_data_list
            
        return result
    
    def _extract_structured_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from event (unified format with top-level fields).
        Respects exclude_fields for ablation studies.
        
        Args:
            event: Event dictionary
            
        Returns:
            Structured data dictionary or None
        """
        structured_data = {}
        structured_fields = [
            'People', 'Location', 'Result', 'Relations', 'Event Attributes'
        ]
        
        has_data = False
        for field in structured_fields:
            # Skip fields that are in exclude_fields (for ablation)
            if field in self.exclude_fields:
                continue
            if field in event and event[field]:
                structured_data[field] = event[field]
                has_data = True
        
        return structured_data if has_data else None
    
    def format_structured_data(self, structured_data: Dict[str, Any]) -> str:
        """
        Format structured data for inclusion in prompts.
        Filters to essential fields only: People, Location, Result, Event Attributes
        Respects exclude_fields for ablation studies.
        
        Args:
            structured_data: Structured data dictionary
            
        Returns:
            Formatted structured data string (JSON)
        """
        import json
        
        # Filter to essential fields only (remove Relations and other noise)
        essential_fields = ['People', 'Location', 'Result', 'Event Attributes']
        
        filtered_data = {}
        for field in essential_fields:
            # Skip fields that are in exclude_fields (for ablation)
            if field in self.exclude_fields:
                continue
            if field in structured_data and structured_data[field]:
                filtered_data[field] = structured_data[field]
        
        return json.dumps(filtered_data, indent=2, ensure_ascii=False)
    
    def get_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the overall date range of all events.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        if not self.all_events:
            return None, None
        
        dates = [event.get('parsed_date') for event in self.all_events if event.get('parsed_date')]
        
        if not dates:
            return None, None
        
        return min(dates), max(dates)
    
    def get_clustering_documents(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get documents formatted for clustering from events.
        Uses texts_per_event to sample texts from each event.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of document dictionaries with text and metadata
        """
        documents = []
        
        for event in events:
            # Sample texts from event using texts_per_event parameter
            event_texts = self.sample_event_texts(event)
            event_date = event.get('event_date') or event.get('date', '')
            true_topic = event.get('topic_title', event.get('topic', 'unknown'))
            
            for text in event_texts:
                doc = {
                    'text': text,
                    'true_topic': true_topic,
                    'event_date': event_date,
                    'event': event  # Keep reference to original event
                }
                documents.append(doc)
        
        return documents
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data statistics."""
        total_events = len(self.all_events)
        total_topics = len(self.topic_events)
        
        # Calculate texts per event distribution
        text_counts = []
        for event in self.all_events:
            text_count = len(event.get('event_text', []))
            text_counts.append(text_count)
        
        avg_texts_per_event = sum(text_counts) / len(text_counts) if text_counts else 0
        
        start_date, end_date = self.get_date_range()
        
        return {
            'total_events': total_events,
            'total_topics': total_topics,
            'avg_texts_per_event': avg_texts_per_event,
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d') if start_date else None,
                'end': end_date.strftime('%Y-%m-%d') if end_date else None
            },
            'events_per_topic': {topic: len(events) for topic, events in self.topic_events.items()}
        }


class QADataLoader:
    """
    Specialized data loader for Task-3 QA datasets.
    """
    
    def __init__(self, qa_file: str, texts_per_event: int = 1, random_seed: int = 42):
        """
        Initialize QA data loader.
        
        Args:
            qa_file: Path to QA JSON file
            texts_per_event: Number of texts to sample per event
            random_seed: Random seed for reproducibility
        """
        self.qa_file = qa_file
        self.texts_per_event = texts_per_event
        self.random_seed = random_seed
        self.qa_data = None
        
        # Set random seed
        random.seed(self.random_seed)
        
        self._load_qa_data()
    
    def _load_qa_data(self):
        """Load QA dataset."""
        try:
            with open(self.qa_file, 'r', encoding='utf-8') as f:
                self.qa_data = json.load(f)
            
            logging.info(f"Loaded QA dataset: {len(self.qa_data.get('events', []))} QA events")
            
        except Exception as e:
            logging.error(f"Error loading QA data from {self.qa_file}: {e}")
            raise
    
    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        """Get QA pairs sorted by date."""
        qa_pairs = self.qa_data.get('qa_pairs', [])
        return sorted(qa_pairs, key=lambda x: x.get('event_date', ''))
    
    def get_qa_events(self) -> List[Dict[str, Any]]:
        """Get QA events."""
        return self.qa_data.get('events', [])
    
    def get_qa_pairs_for_window(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get QA pairs that match the events in a window.
        
        Args:
            events: List of events in the window
            
        Returns:
            List of QA pairs matching the window events
        """
        window_dates = set()
        for event in events:
            event_date = event.get('event_date') or event.get('date')
            if event_date:
                window_dates.add(event_date)
        
        qa_pairs = []
        
        # Each event in our dataset has qa_pairs embedded in it
        # We need to extract qa_pairs from events that match our window dates
        for event in self.qa_data.get('events', []):
            event_date = event.get('event_date')
            if event_date in window_dates:
                event_qa_pairs = event.get('qa_pairs', [])
                
                # Add event metadata to each QA pair
                for qa_pair in event_qa_pairs:
                    enriched_qa_pair = qa_pair.copy()
                    enriched_qa_pair['event_date'] = event_date
                    enriched_qa_pair['referenced_article'] = qa_pair.get('referenced_article', '')
                    
                    # Get the first event text if no referenced article
                    if not enriched_qa_pair['referenced_article']:
                        event_texts = event.get('event_text', [])
                        if event_texts:
                            enriched_qa_pair['referenced_article'] = event_texts[0]
                    
                    qa_pairs.append(enriched_qa_pair)
        
        return qa_pairs
    
    def get_all_qa_pairs(self) -> List[Dict[str, Any]]:
        """Get all QA pairs from all events."""
        all_qa_pairs = []
        
        for event in self.qa_data.get('events', []):
            event_date = event.get('event_date')
            event_qa_pairs = event.get('qa_pairs', [])
            
            for qa_pair in event_qa_pairs:
                enriched_qa_pair = qa_pair.copy()
                enriched_qa_pair['event_date'] = event_date
                enriched_qa_pair['referenced_article'] = qa_pair.get('referenced_article', '')
                
                # Get the first event text if no referenced article
                if not enriched_qa_pair['referenced_article']:
                    event_texts = event.get('event_text', [])
                    if event_texts:
                        enriched_qa_pair['referenced_article'] = event_texts[0]
                
                all_qa_pairs.append(enriched_qa_pair)
        
        return all_qa_pairs
    
    def sample_event_texts(self, event: Dict[str, Any], num_texts: Optional[int] = None) -> List[str]:
        """
        Sample texts from an event.
        
        Args:
            event: Event dictionary
            num_texts: Number of texts to sample (defaults to self.texts_per_event)
            
        Returns:
            List of sampled texts
        """
        if num_texts is None:
            num_texts = self.texts_per_event
            
        event_texts = event.get('event_text', [])
        
        if not event_texts:
            return []
        
        if num_texts >= len(event_texts):
            return event_texts
        
        return random.sample(event_texts, num_texts)
    
    def get_clustering_documents(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get documents formatted for clustering from events.
        Uses texts_per_event to sample texts from each event.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of document dictionaries with text and metadata
        """
        documents = []
        
        for event in events:
            # Sample texts from event using texts_per_event parameter
            event_texts = self.sample_event_texts(event)
            event_date = event.get('event_date') or event.get('date', '')
            true_topic = event.get('topic_title', event.get('topic', 'unknown'))
            
            for text in event_texts:
                doc = {
                    'text': text,
                    'true_topic': true_topic,
                    'event_date': event_date,
                    'event': event  # Keep reference to original event
                }
                documents.append(doc)
        
        return documents
