#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Metrics for Streaming Summarization

This module provides metrics for evaluating summarization quality,
including ROUGE, BLEU, BERTScore, METEOR, BLEURT, and EMDS metrics.
"""

import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Callable, Optional
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer, AutoModel
import evaluate
import numpy as np

try:
    from transformers import AutoModelForSequenceClassification
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False
    logging.warning("BLEURT not available. Install transformers for BLEURT support.")

# Default BERT model configuration
BERT_MODEL = "bert-base-uncased"

class SummaryEvaluator:
    """
    Evaluator for summarization quality using various metrics.
    """
    
    def __init__(self):
        """
        Initialize the summary evaluator with necessary models and scorers.
        """
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Determine device based on CUDA_VISIBLE_DEVICES
        import os
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logging.info(f"Using GPU: CUDA_VISIBLE_DEVICES={cuda_visible} (PyTorch sees as cuda:0)")
            logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            
            # Enable memory optimization
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        else:
            self.device = torch.device('cpu')
            logging.info("Using CPU for evaluation")
        
        # Initialize BERT model and tokenizer for BERTScore
        logging.info(f"Loading BERT model: {BERT_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.model = AutoModel.from_pretrained(BERT_MODEL).to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
        """
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logging.error(f"Error calculating ROUGE scores: {e}")
            return {'rouge1_f': 0.0, 'rouge2_f': 0.0, 'rougeL_f': 0.0}
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            BLEU score between 0 and 1
        """
        try:
            ref_tokens = [reference.lower().split()]
            hyp_tokens = hypothesis.lower().split()
            
            smoothing = SmoothingFunction().method1
            return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothing)
        except Exception as e:
            logging.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Calculate METEOR score.
        
        Args:
            reference: Reference summary
            hypothesis: Generated summary
            
        Returns:
            METEOR score between 0 and 1
        """
        try:
            ref_tokens = reference.lower().split()
            hyp_tokens = hypothesis.lower().split()
            return meteor_score([ref_tokens], hyp_tokens)
        except Exception as e:
            logging.error(f"Error calculating METEOR score: {e}")
            return 0.0
    
    def calculate_bertscore(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """
        Calculate BERTScore using batch processing with memory cleanup.
        
        Args:
            references: List of reference summaries
            hypotheses: List of generated summaries
            
        Returns:
            List of BERTScore F1 scores
        """
        try:
            bertscore = evaluate.load("bertscore")
            device_str = 'cuda' if self.device.type == 'cuda' else None
            results = bertscore.compute(
                predictions=hypotheses, 
                references=references, 
                lang="en",
                device=device_str
            )
            scores = results["f1"]
            
            # Clean up
            del bertscore, results
            torch.cuda.empty_cache()
            
            return scores
        except Exception as e:
            logging.error(f"Error calculating BERTScore: {e}")
            torch.cuda.empty_cache()
            return [0.0] * len(references)
    
    def calculate_bleurt(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """
        Calculate BLEURT scores using batch processing.
        
        Args:
            references: List of reference summaries
            hypotheses: List of generated summaries
            
        Returns:
            List of BLEURT scores
        """
        if not BLEURT_AVAILABLE:
            logging.warning("BLEURT not available")
            return [0.0] * len(references)
        
        try:
            # Load BLEURT without device parameter (incompatible in newer versions)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bleurt = evaluate.load("bleurt", module_type="metric")
            
            # Compute without device parameter
            results = bleurt.compute(
                predictions=hypotheses, 
                references=references
            )
            return results["scores"]
        except Exception as e:
            logging.error(f"Error calculating BLEURT: {e}")
            return [0.0] * len(references)
    
    def bert_score_single(self, hypothesis: str, reference: str) -> float:
        """
        Calculate BERTScore for a single hypothesis-reference pair using cached model.
        
        Args:
            hypothesis: Generated summary
            reference: Reference summary
            
        Returns:
            BERTScore F1 score between 0 and 1
        """
        try:
            # Tokenize
            hyp_tokens = self.tokenizer(hypothesis, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            ref_tokens = self.tokenizer(reference, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                hyp_output = self.model(**hyp_tokens)
                ref_output = self.model(**ref_tokens)
                
                # Use [CLS] token embeddings
                hyp_embedding = hyp_output.last_hidden_state[:, 0, :]
                ref_embedding = ref_output.last_hidden_state[:, 0, :]
                
                # Compute cosine similarity
                cosine_sim = F.cosine_similarity(hyp_embedding, ref_embedding, dim=1)
                score = cosine_sim.item()
                
                # Clean up
                del hyp_tokens, ref_tokens, hyp_output, ref_output, hyp_embedding, ref_embedding
                torch.cuda.empty_cache()
                
                return max(0.0, min(1.0, score))
        except Exception as e:
            logging.error(f"Error calculating single BERTScore: {e}")
            torch.cuda.empty_cache()
            return 0.0
    
    def rouge_f_single(self, hypothesis: str, reference: str, rouge_type: str = "rougeL") -> float:
        """
        Calculate ROUGE F1 score for a single hypothesis-reference pair.
        
        Args:
            hypothesis: Generated summary
            reference: Reference summary
            rouge_type: Type of ROUGE (rouge1, rouge2, rougeL)
            
        Returns:
            ROUGE F1 score between 0 and 1
        """
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return scores[rouge_type].fmeasure
        except Exception as e:
            logging.error(f"Error calculating single ROUGE score: {e}")
            return 0.0
    
    def emds_relevance(self, metric_fn: Callable, current_summary: str, reference_summary: str) -> float:
        """
        Calculate EMDS Relevance metric - measures how relevant the current summary is to the reference.
        
        Args:
            metric_fn: Similarity function that takes (hypothesis, reference) and returns a score
            current_summary: Current system-generated summary
            reference_summary: Gold/reference summary for the same topic
            
        Returns:
            Relevance score between 0 and 1, higher means more relevant
        """
        # Simple direct application of the similarity metric
        return metric_fn(current_summary, reference_summary)
    
    def emds_novelty(self, metric_fn: Callable, current_summary: str, previous_summary: str, reference_summary: str) -> float:
        """
        Calculate EMDS Novelty metric - measures how much new relevant information is in the current summary
        compared to the previous summary.
        
        Args:
            metric_fn: Similarity function that takes (hypothesis, reference) and returns a score
            current_summary: Current system-generated summary
            previous_summary: Previous system-generated summary for the same topic
            reference_summary: Gold/reference summary for the same topic
            
        Returns:
            Novelty score between 0 and 1, higher means more novel relevant content
        """
        # Extract novel tokens (tokens in current summary that weren't in previous summary)
        novel_tokens = list(set(current_summary.split()) - set(previous_summary.split()))
        novel_text = ' '.join(novel_tokens)
        
        # If there are no novel tokens, return 0
        if not novel_text.strip():
            return 0.0
            
        # Calculate similarity between the novel content and the reference summary
        return metric_fn(novel_text, reference_summary)
    
    def emds_distinctiveness(self, metric_fn: Callable, current_summary: str, current_reference: str, other_references: List[str]) -> float:
        """
        Calculate EMDS Distinctiveness metric - measures how well the current summary distinguishes its topic
        from other topics. A good summary should be similar to its own reference but different from references
        of other topics.
        
        Args:
            metric_fn: Similarity function that takes (hypothesis, reference) and returns a score
            current_summary: Current system-generated summary
            current_reference: Gold/reference summary for the current topic
            other_references: List of gold/reference summaries for other topics
            
        Returns:
            Distinctiveness score, higher means better distinction between topics
        """
        if not other_references:
            return 1.0  # Perfect distinctiveness if no other topics to compare with
        
        # Calculate similarity to the target reference
        target_similarity = metric_fn(current_summary, current_reference)
        
        # Calculate dissimilarity to other references (1 - similarity for each)
        # Higher dissimilarity means better distinction between topics
        dissimilarity_to_others = sum([1 - metric_fn(current_summary, other_ref) for other_ref in other_references])
        
        # Normalize by the number of other references and the dissimilarity to current reference
        # Small epsilon (1e-8) added to avoid division by zero
        denominator = len(other_references) * (1 - target_similarity + 1e-8)
        
        return dissimilarity_to_others / denominator
    
    def evaluate_all(self, references: List[str], hypotheses: List[str], 
                    topics: Optional[List[str]] = None, 
                    window_indices: Optional[List[int]] = None,
                    event_counts: Optional[List[int]] = None,
                    calculate_emds: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of summarization results.
        
        Args:
            references: List of reference summaries
            hypotheses: List of generated summaries
            topics: Optional list of topic names for each summary
            window_indices: Optional list of window indices for temporal analysis
            event_counts: Optional list of event counts per window
            calculate_emds: Whether to calculate EMDS metrics
            
        Returns:
            Dictionary containing detailed results and summary statistics
        """
        if len(references) != len(hypotheses):
            raise ValueError(f"Number of references ({len(references)}) must match number of hypotheses ({len(hypotheses)})")
        
        logging.info("Starting evaluation")
        
        # Store evaluation results
        results = {
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'bleu': [],
            'bertscore': [],
            'meteor': [],
            'emds_relevance_bert': [],
            'emds_novelty_bert': [],
            'emds_distinctiveness_bert': [],
            'emds_relevance_rouge': [],
            'emds_novelty_rouge': [],
            'emds_distinctiveness_rouge': []
        }
        
        # Add metadata if provided
        if topics is not None:
            results['topic'] = topics
        if window_indices is not None:
            results['window_index'] = window_indices
        if event_counts is not None:
            results['event_count'] = event_counts
        
        # Calculate ROUGE, BLEU, and METEOR for each pair
        print(f"  [1/6] ROUGE scores (fast)...", end='', flush=True)
        for i, (reference, hypothesis) in enumerate(zip(references, hypotheses)):
            # ROUGE
            rouge_scores = self.calculate_rouge(reference, hypothesis)
            for key, value in rouge_scores.items():
                results[key].append(value)
            
            # BLEU
            bleu_score = self.calculate_bleu(reference, hypothesis)
            results['bleu'].append(bleu_score)
            
            # METEOR
            meteor_score_value = self.calculate_meteor(reference, hypothesis)
            results['meteor'].append(meteor_score_value)
        print(f" ✓")
        
        # Calculate BERTScore (batch processing)
        print(f"  [2/6] BERTScore (GPU, slow)...", end='', flush=True)
        try:
            bert_scores = self.calculate_bertscore(references, hypotheses)
            results['bertscore'] = bert_scores
            print(f" ✓")
        except Exception as e:
            logging.error(f"Error during BERTScore calculation: {e}")
            results['bertscore'] = [0.0] * len(references)
            print(f" ✗")
        
        # BLEURT is completely removed due to frequent compatibility problems
        # Skipping metric [3/7]
        
        # Calculate EMDS metrics if requested
        if calculate_emds and len(references) > 0:
            # Initialize EMDS results
            results['emds_relevance_bert'] = []
            results['emds_novelty_bert'] = []
            results['emds_distinctiveness_bert'] = []
            results['emds_relevance_rouge'] = []
            results['emds_novelty_rouge'] = []
            results['emds_distinctiveness_rouge'] = []
            
            # Define similarity functions
            bert_fn = self.bert_score_single
            rouge_fn = lambda hyp, ref: self.rouge_f_single(hyp, ref, rouge_type="rougeL")
            
            # Group references by topic
            topic_to_refs = {}
            for i, topic in enumerate(topics if topics else ['unknown'] * len(references)):
                if topic not in topic_to_refs:
                    topic_to_refs[topic] = []
                topic_to_refs[topic].append(references[i])
            
            # Calculate EMDS metrics for each summary
            print(f"  [3/6] EMDS BERTScore (very slow, {len(hypotheses)} windows)...", end='', flush=True)
            for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
                topic = topics[i] if topics else 'unknown'
                window_idx = window_indices[i] if window_indices else 0
                
                # Get other topics' references
                other_refs = []
                for other_topic, refs in topic_to_refs.items():
                    if other_topic != topic:
                        other_refs.extend(refs)
                
                # Get previous summary for the same topic (if available)
                prev_hyp = ""
                for j in range(i):
                    if (topics[j] if topics else 'unknown') == topic and \
                       (window_indices[j] if window_indices else 0) < window_idx:
                        prev_hyp = hypotheses[j]
                        break
                
                # If no previous summary, use empty string
                if not prev_hyp:
                    prev_hyp = ""
                
                # Calculate BERTScore-based EMDS metrics
                relevance_bert = self.emds_relevance(bert_fn, hyp, ref)
                novelty_bert = self.emds_novelty(bert_fn, hyp, prev_hyp, ref)
                distinctiveness_bert = self.emds_distinctiveness(bert_fn, hyp, ref, other_refs)
                
                # Store results
                results['emds_relevance_bert'].append(relevance_bert)
                results['emds_novelty_bert'].append(novelty_bert)
                results['emds_distinctiveness_bert'].append(distinctiveness_bert)
                
                # Periodic memory cleanup (every 10 windows)
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            print(f" ✓")
            
            # Final cleanup after EMDS BERTScore
            torch.cuda.empty_cache()
            
            # Calculate ROUGE-L-based EMDS metrics
            print(f"  [4/6] EMDS ROUGE-L (fast, {len(hypotheses)} windows)...", end='', flush=True)
            for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
                topic = topics[i] if topics else 'unknown'
                window_idx = window_indices[i] if window_indices else 0
                
                # Get other topics' references
                other_refs = []
                for other_topic, refs in topic_to_refs.items():
                    if other_topic != topic:
                        other_refs.extend(refs)
                
                # Get previous summary for the same topic
                prev_hyp = ""
                for j in range(i):
                    if (topics[j] if topics else 'unknown') == topic and \
                       (window_indices[j] if window_indices else 0) < window_idx:
                        prev_hyp = hypotheses[j]
                        break
                
                if not prev_hyp:
                    prev_hyp = ""
                
                # Calculate ROUGE-L-based EMDS metrics
                relevance_rouge = self.emds_relevance(rouge_fn, hyp, ref)
                novelty_rouge = self.emds_novelty(rouge_fn, hyp, prev_hyp, ref)
                distinctiveness_rouge = self.emds_distinctiveness(rouge_fn, hyp, ref, other_refs)
                
                # Store results
                results['emds_relevance_rouge'].append(relevance_rouge)
                results['emds_novelty_rouge'].append(novelty_rouge)
                results['emds_distinctiveness_rouge'].append(distinctiveness_rouge)
            print(f" ✓")
        
        # Summarize results
        summary = {
            'rouge1_f_avg': sum(results['rouge1_f']) / len(results['rouge1_f']) if results['rouge1_f'] else 0,
            'rouge2_f_avg': sum(results['rouge2_f']) / len(results['rouge2_f']) if results['rouge2_f'] else 0,
            'rougeL_f_avg': sum(results['rougeL_f']) / len(results['rougeL_f']) if results['rougeL_f'] else 0,
            'bleu_avg': sum(results['bleu']) / len(results['bleu']) if results['bleu'] else 0,
            'bertscore_avg': sum(results['bertscore']) / len(results['bertscore']) if results['bertscore'] else 0,
            'meteor_avg': sum(results['meteor']) / len(results['meteor']) if results['meteor'] else 0,
            'sample_count': len(references)
        }
        
        # Add EMDS metrics to summary if available
        if calculate_emds and len(references) > 0:
            summary['emds_relevance_bert'] = sum(results['emds_relevance_bert']) / len(results['emds_relevance_bert']) if results['emds_relevance_bert'] else 0
            summary['emds_novelty_bert'] = sum(results['emds_novelty_bert']) / len(results['emds_novelty_bert']) if results['emds_novelty_bert'] else 0
            summary['emds_distinctiveness_bert'] = sum(results['emds_distinctiveness_bert']) / len(results['emds_distinctiveness_bert']) if results['emds_distinctiveness_bert'] else 0
            summary['emds_relevance_rouge'] = sum(results['emds_relevance_rouge']) / len(results['emds_relevance_rouge']) if results['emds_relevance_rouge'] else 0
            summary['emds_novelty_rouge'] = sum(results['emds_novelty_rouge']) / len(results['emds_novelty_rouge']) if results['emds_novelty_rouge'] else 0
            summary['emds_distinctiveness_rouge'] = sum(results['emds_distinctiveness_rouge']) / len(results['emds_distinctiveness_rouge']) if results['emds_distinctiveness_rouge'] else 0
        
        # Log basic metrics
        logging.info(f"Evaluation completed: ROUGE-1={summary['rouge1_f_avg']:.4f}, ROUGE-2={summary['rouge2_f_avg']:.4f}, "
                   f"ROUGE-L={summary['rougeL_f_avg']:.4f}, BLEU={summary['bleu_avg']:.4f}, BERTScore={summary['bertscore_avg']:.4f}, "
                   f"METEOR={summary['meteor_avg']:.4f}")
        
        # Log EMDS metrics if available
        if calculate_emds and len(references) > 0:
            logging.info(f"EMDS (BERTScore): Relevance={summary['emds_relevance_bert']:.4f}, "
                       f"Novelty={summary['emds_novelty_bert']:.4f}, "
                       f"Distinctiveness={summary['emds_distinctiveness_bert']:.4f}")
            logging.info(f"EMDS (ROUGE-L): Relevance={summary['emds_relevance_rouge']:.4f}, "
                       f"Novelty={summary['emds_novelty_rouge']:.4f}, "
                       f"Distinctiveness={summary['emds_distinctiveness_rouge']:.4f}")
        
        return {
            'detailed': results,
            'summary': summary
        }
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of summarization metrics."""
        return {
            'rouge1_f': 'ROUGE-1 F1 Score - unigram overlap F1',
            'rouge2_f': 'ROUGE-2 F1 Score - bigram overlap F1',
            'rougeL_f': 'ROUGE-L F1 Score - longest common subsequence F1',
            'bleu': 'BLEU Score - modified precision with brevity penalty',
            'bertscore': 'BERTScore F1 - BERT-based semantic F1',
            'meteor': 'METEOR Score - alignment-based metric with synonyms',
            'emds_relevance_bert': 'EMDS Relevance (BERTScore) - relevance to reference summary',
            'emds_novelty_bert': 'EMDS Novelty (BERTScore) - novel information vs previous summaries',
            'emds_distinctiveness_bert': 'EMDS Distinctiveness (BERTScore) - distinction from other topics',
            'emds_relevance_rouge': 'EMDS Relevance (ROUGE) - relevance to reference summary',
            'emds_novelty_rouge': 'EMDS Novelty (ROUGE) - novel information vs previous summaries',
            'emds_distinctiveness_rouge': 'EMDS Distinctiveness (ROUGE) - distinction from other topics',
            'sample_count': 'Number of summaries evaluated'
        }

# Export the main class
__all__ = ['SummaryEvaluator']
