"""
# Example usage:
python temporal_qa/llm_evaluation.py \
  --data_json data_qa/temporal_qa_1.json \
  --out_dir ./data_qa_with_llmeval \
  --provider openai \
  --model gpt-4o

  
QA evaluator for temporal QA

# Input
- temporal_qa_{story_num}.json
    - datafreame with temporal QA data
    - there is a list of events where each event has a list of qa pairs with distractors

# Evaluation criteria:
Question Quality
    - Answer Concealment : Question should not include the answer target.
    - Temporal Validity: Question must target time-varying attributes or causal relationships and be explicitly tied to a temporal reference (e.g., currently, after ~, etc.).

Answer Quality
    - Contextual Groundedness: Answer should be grounded in the Event Context.
    - Specificity Alignment: The specificity of the answer should be aligned to the specificity of the reported information.

Choice Quality
    - Contextual Plausibility: Choices must be contextually relevant and match the specificity level of the answer.
    - Mutual Exclusivity: Choices should be mutually exclusive, ensuring no redundancy or semantic overlap.

# Output
- temporal_qa_{story_num}_with_llmeval.json
"""

import os
import json
import argparse
import random
from typing import Optional, Dict, List, Any
from pathlib import Path
from tqdm import tqdm

class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float,
        openai_api_key: Optional[str] = None, together_api_key: Optional[str] = None):

        self.provider = provider.lower()
        self.model = model
        self.temperature = float(temperature)

        if self.provider == "openai":
            # Use the passed API key or fall back to environment variable
            openai_api_key = ""
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided either as parameter or via OPENAI_API_KEY environment variable")
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "together":
            # Use the passed API key or fall back to environment variable
            api_key = together_api_key or os.getenv("TOGETHER_API_KEY")
            if not api_key:
                raise ValueError("Together API key must be provided either as parameter or via TOGETHER_API_KEY environment variable")
            from together import Together  # type: ignore
            self.client = Together(api_key=api_key)
        else:
            raise ValueError("provider must be one of: openai, together")

    def chat(self, system: str, user: str) -> str:
        base_params = {
            "model": self.model,
            "temperature": 0.0,           # No randomness
            "top_p": 1.0,                 # Do not truncate token distribution
            "frequency_penalty": 0.0,     # No penalty for repeated tokens
            "presence_penalty": 0.0,      # No encouragement for new topics
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        }
        # openai api
        if self.provider == "openai":
            resp = self.client.chat.completions.create(**base_params)
            return (resp.choices[0].message.content or "").strip()
        
        # together ai api
        resp = self.client.chat.completions.create(**base_params)
        try:
            return (resp.choices[0].message["content"] or "").strip()
        except Exception:
            return (resp.choices[0].message.content or "").strip()

class QAEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        
    def evaluate_question_quality(self, question: str, answer: str, choices: Dict[str, str], event_text: List[str]) -> Dict[str, Any]:
        """
        Evaluate question quality based on two criteria:
        1. Answer Concealment: Question should not include the answer target.
        2. Temporal Validity: Question must target time-varying attributes or causal relationships and be explicitly tied to a temporal reference.
        
        Returns score 0-2 (1 point per criterion satisfied).
        """
        system_prompt = """You are an expert evaluator of question quality for temporal QA datasets. 
        Evaluate the given question based on two criteria:
        
        Criterion 1 - Answer Concealment: Question should not include the answer target.
        Criterion 2 - Temporal Validity: Question must target time-varying attributes or causal relationships and be explicitly tied to a temporal reference (e.g., currently, after ~, etc.).
        
        For each criterion, provide:
        - is_satisfied: true if satisfied, false if not satisfied
        - reason: brief explanation of your assessment

        Score: 1 point per criterion satisfied (total 0-2).
        
        Return your evaluation as a JSON object with the following structure:
        {
            "answer_concealment": {"is_satisfied": True, "reason": "explanation"},
            "temporal_validity": {"is_satisfied": True, "reason": "explanation"},
            "score": 2
        }
        
        Be strict but fair in your evaluation."""
        
        user_prompt = f"""Question: {question}
        Correct Answer: {answer}
        All Choices: {json.dumps(choices, indent=2)}
        
        Evaluate the question quality based on the two criteria above."""
        
        try:
            response = self.llm_client.chat(system_prompt, user_prompt)
            
            # Debug: Print the actual response
            print(f"LLM Response (question quality): '{response}'")
            
            # Check if response is empty
            if not response or response.strip() == "":
                print("Warning: Empty response from LLM")
                return {
                    "answer_concealment": {"is_satisfied": False, "reason": "Empty LLM response"},
                    "temporal_validity": {"is_satisfied": False, "reason": "Empty LLM response"},
                    "score": 0
                }
            
            # Extract JSON from markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.replace('```', '').strip()
            
            # Parse JSON response
            evaluation = json.loads(cleaned_response)
            
            # Calculate score from criteria
            score = 0
            if evaluation.get("answer_concealment", {}).get("is_satisfied", False):
                score += 1
            if evaluation.get("temporal_validity", {}).get("is_satisfied", False):
                score += 1
            
            evaluation["score"] = score
            return evaluation
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: '{response}'")
            return {
                "answer_concealment": {"is_satisfied": False, "reason": "Invalid JSON response"},
                "temporal_validity": {"is_satisfied": False, "reason": "Invalid JSON response"},
                "score": 0
            }
        except Exception as e:
            print(f"Error evaluating question quality: {e}")
            return {
                "answer_concealment": {"is_satisfied": False, "reason": "Evaluation failed"},
                "temporal_validity": {"is_satisfied": False, "reason": "Evaluation failed"},
                "score": 0
            }
    
    def evaluate_answer_quality(self, question: str, answer: str, choices: Dict[str, str], event_text: List[str], qa_pair: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate answer quality based on two criteria:
        1. Contextual Groundedness: Answer should be grounded in the Event Context.
        2. Specificity Alignment: The specificity of the answer should be aligned to the specificity of the reported information.
        
        Returns score 0-2 (1 point per criterion satisfied).
        """
        system_prompt = """You are an expert evaluator of answer quality for temporal QA datasets.
        Evaluate the given answer based on two criteria:
        
        Criterion 1 - Contextual Groundedness: Answer should be grounded in the Event Context.
        Criterion 2 - Specificity Alignment: The specificity of the answer should be aligned to the specificity of the reported information.
        
        For each criterion, provide:
        - is_satisfied: true if satisfied, false if not satisfied
        - reason: brief explanation of your assessment

        Score: 1 point per criterion satisfied (total 0-2).
        
        Return your evaluation as a JSON object with the following structure:
        {
            "contextual_groundedness": {"is_satisfied": True, "reason": "explanation"},
            "specificity_alignment": {"is_satisfied": True, "reason": "explanation"},
            "score": 2
        }
        
        Be strict but fair in your evaluation."""
        
        # Get referenced article if available
        if qa_pair and "referenced_article" in qa_pair:
            referenced_article = qa_pair["referenced_article"]
        else:
            referenced_article = "No referenced article available"

        answer_text = choices.get(answer, "")
        
        user_prompt = f"""Question: {question}
        Answer: {answer_text} (label: {answer})
        All Choices: {json.dumps(choices, indent=2)}
        Referenced Article: {referenced_article}
        
        Evaluate the answer quality based on the two criteria above."""
        
        try:
            response = self.llm_client.chat(system_prompt, user_prompt)
            
            # Debug: Print the actual response
            print(f"LLM Response (answer quality): '{response}'")
            
            # Check if response is empty
            if not response or response.strip() == "":
                print("Warning: Empty response from LLM for answer quality")
                return {
                    "contextual_groundedness": {"is_satisfied": False, "reason": "Empty LLM response"},
                    "specificity_alignment": {"is_satisfied": False, "reason": "Empty LLM response"},
                    "score": 0
                }
            
            # Extract JSON from markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.replace('```', '').strip()
            
            evaluation = json.loads(cleaned_response)
            
            # Calculate score from criteria
            score = 0
            if evaluation.get("contextual_groundedness", {}).get("is_satisfied", False):
                score += 1
            if evaluation.get("specificity_alignment", {}).get("is_satisfied", False):
                score += 1
            
            evaluation["score"] = score
            return evaluation
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in answer quality: {e}")
            print(f"Raw response: '{response}'")
            return {
                "contextual_groundedness": {"is_satisfied": False, "reason": "Invalid JSON response"},
                "specificity_alignment": {"is_satisfied": False, "reason": "Invalid JSON response"},
                "score": 0
            }
        except Exception as e:
            print(f"Error evaluating answer quality: {e}")
            return {
                "contextual_groundedness": {"is_satisfied": False, "reason": "Evaluation failed"},
                "specificity_alignment": {"is_satisfied": False, "reason": "Evaluation failed"},
                "score": 0
            }
    
    def evaluate_choice_quality(self, question_type: str, question: str, 
                              choices: Dict[str, str], answer: str, 
                              event_text: List[str], qa_pair: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate choice quality based on two criteria:
        1. Contextual Plausibility: Choices must be contextually relevant and match the specificity level of the answer.
        2. Mutual Exclusivity: Choices should be mutually exclusive, ensuring no redundancy or semantic overlap.
        
        Returns score 0-2 (1 point per criterion satisfied).
        """
        system_prompt = """You are an expert evaluator of choice quality for temporal QA datasets.
        Evaluate the given choices based on two criteria:
        
        Criterion 1 - Contextual Plausibility: Choices must be contextually relevant and match the specificity level of the answer.
        Criterion 2 - Mutual Exclusivity: Choices should be mutually exclusive, ensuring no redundancy or semantic overlap.
        
        For each criterion, provide:
        - is_satisfied: true if satisfied, false if not satisfied
        - reason: brief explanation of your assessment

        Score: 1 point per criterion satisfied (total 0-2).
        
        Return your evaluation as a JSON object with the following structure:
        {
            "contextual_plausibility": {"is_satisfied": True, "reason": "explanation"},
            "mutual_exclusivity": {"is_satisfied": True, "reason": "explanation"},
            "score": 2
        }
        
        Be strict but fair in your evaluation."""
        
        # Get referenced article if available
        if qa_pair and "referenced_article" in qa_pair:
            referenced_article = qa_pair["referenced_article"]
        else:
            referenced_article = "No referenced article available"
        
        user_prompt = f"""Question Type: {question_type}
        Question: {question}
        All Choices: {json.dumps(choices, indent=2)}
        Correct Answer Label: {answer}
        Referenced Article: {referenced_article}
        
        Evaluate the choice quality based on the two criteria above."""
        
        try:
            response = self.llm_client.chat(system_prompt, user_prompt)
            
            # Debug: Print the actual response
            print(f"LLM Response (choice quality): '{response}'")
            
            # Check if response is empty
            if not response or response.strip() == "":
                print("Warning: Empty response from LLM for choice quality")
                return {
                    "contextual_plausibility": {"is_satisfied": False, "reason": "Empty LLM response"},
                    "mutual_exclusivity": {"is_satisfied": False, "reason": "Empty LLM response"},
                    "score": 0
                }
            
            # Extract JSON from markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.replace('```json', '').replace('```', '').strip()
            elif cleaned_response.startswith('```'):
                cleaned_response = cleaned_response.replace('```', '').strip()
            
            evaluation = json.loads(cleaned_response)
            
            # Calculate score from criteria
            score = 0
            if evaluation.get("contextual_plausibility", {}).get("is_satisfied", False):
                score += 1
            if evaluation.get("mutual_exclusivity", {}).get("is_satisfied", False):
                score += 1
            
            evaluation["score"] = score
            return evaluation
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in choice quality: {e}")
            print(f"Raw response: '{response}'")
            return {
                "contextual_plausibility": {"is_satisfied": False, "reason": "Invalid JSON response"},
                "mutual_exclusivity": {"is_satisfied": False, "reason": "Invalid JSON response"},
                "score": 0
            }
        except Exception as e:
            print(f"Error evaluating choice quality: {e}")
            return {
                "contextual_plausibility": {"is_satisfied": False, "reason": "Evaluation failed"},
                "mutual_exclusivity": {"is_satisfied": False, "reason": "Evaluation failed"},
                "score": 0
            }
    
    def filter_invalid_qa_pairs(self, qa_pairs: List[Dict[str, Any]], redundancy_check: Dict[str, Any], sanity_check: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Filter QA pairs based on redundancy check and validity. For each question type with redundancy=True,
        randomly select 1 QA pair instead of keeping all. Only include QA pairs where is_valid=True.
        
        Args:
            qa_pairs: List of QA pairs for the event
            redundancy_check: Redundancy check results for the event
            sanity_check: Sanity check results from the event (contains qa_checks with is_valid)
            
        Returns:
            Filtered list of QA pairs
        """
        # First filter by validity using sanity_check
        valid_qa_pairs = []
        if sanity_check and "qa_checks" in sanity_check:
            qa_checks = sanity_check["qa_checks"]
            for idx, qa_pair in enumerate(qa_pairs):
                if idx < len(qa_checks) and qa_checks[idx].get("is_valid", False):
                    valid_qa_pairs.append(qa_pair)
        else:
            # If no sanity_check, assume all are valid (for backward compatibility)
            valid_qa_pairs = qa_pairs
        
        if not valid_qa_pairs:
            print("No valid QA pairs found after validation filtering")
            return []
        
        if not redundancy_check:
            # Filter by validity only
            return valid_qa_pairs
        
        filtered_pairs = []
        
        # Group valid QA pairs by question type
        qa_by_type = {}
        for qa_pair in valid_qa_pairs:
            question_type = qa_pair["question_type"]
            if question_type not in qa_by_type:
                qa_by_type[question_type] = []
            qa_by_type[question_type].append(qa_pair)
        
        # Process each question type
        for question_type, type_qa_pairs in qa_by_type.items():
            # Normalize question type to match redundancy check keys
            # Convert "Result Recognition" -> "result_recognition", "Entity Tracking" -> "entity_tracking"
            normalized_type = question_type.lower().replace(" ", "_")
            
            # Check if this question type has redundancy check
            if normalized_type in redundancy_check:
                redundancy_info = redundancy_check[normalized_type]
                is_redundant = redundancy_info.get("is_redundant", False)
                
                if is_redundant:
                    # If redundant, randomly select 1 QA pair
                    selected_pair = random.choice(type_qa_pairs)
                    filtered_pairs.append(selected_pair)
                    print(f"Redundancy detected for {question_type}: randomly selected 1 QA pair out of {len(type_qa_pairs)}")
                else:
                    # If not redundant, keep all QA pairs
                    filtered_pairs.extend(type_qa_pairs)
                    print(f"No redundancy for {question_type}: keeping all {len(type_qa_pairs)} QA pairs")
            else:
                # If no redundancy check info, keep all QA pairs
                filtered_pairs.extend(type_qa_pairs)
                print(f"No redundancy check info for {question_type}: keeping all {len(type_qa_pairs)} QA pairs")
        
        return filtered_pairs
    
    def evaluate_qa_pair(self, qa_pair: Dict[str, Any], event_text: List[str]) -> Dict[str, Any]:
        """
        Evaluate a single QA pair
        """
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")
        choices = qa_pair.get("choices", {})
        question_type = qa_pair.get("question_type", "")
        
        # Evaluate question quality
        question_eval = self.evaluate_question_quality(question, answer, choices, event_text)
        question_score = question_eval["score"]
        
        # Evaluate answer quality
        answer_eval = self.evaluate_answer_quality(question, answer, choices, event_text, qa_pair)
        answer_score = answer_eval["score"]
        
        # Evaluate choice quality
        choice_eval = self.evaluate_choice_quality(question_type, question, choices, answer, event_text, qa_pair)
        choice_score = choice_eval["score"]
        
        return {
            "question_quality": question_eval,
            "answer_quality": answer_eval,
            "choice_quality": choice_eval,
            "question_score": question_score,
            "answer_score": answer_score,
            "choice_score": choice_score
        }
    
    def evaluate_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all QA pairs in an event, with redundancy filtering
        """
        event_text = event.get("event_text", [])
        qa_pairs = event.get("qa_pairs", [])
        redundancy_check = event.get("redundancy_check", {})
        sanity_check = event.get("sanity_check", {})
        
        # Filter QA pairs based on redundancy check and sanity check
        filtered_qa_pairs = self.filter_invalid_qa_pairs(qa_pairs, redundancy_check, sanity_check)
        
        print(f"Event: {len(qa_pairs)} original QA pairs -> {len(filtered_qa_pairs)} after redundancy filtering")
        
        evaluated_qa_pairs = []
        for qa_pair in filtered_qa_pairs:  # Use filtered_qa_pairs instead of qa_pairs
            evaluation = self.evaluate_qa_pair(qa_pair, event_text)
            qa_pair["llm_evaluation"] = evaluation
            evaluated_qa_pairs.append(qa_pair)
        
        return {
            **event,
            "qa_pairs": evaluated_qa_pairs,
            "original_qa_count": len(qa_pairs),
            "filtered_qa_count": len(filtered_qa_pairs)
        }
    
    def compute_structural_statistics(self, data: Dict[str, Any], evaluated_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute structural statistics for the dataset
        """
        # Check if data has topics structure
        has_topics = "topics" in data
        
        if has_topics:
            topics = data.get("topics", [])
            total_topics = len(topics)
        else:
            # If no topics, treat each event as its own topic
            total_topics = len(evaluated_events)
        
        total_events = len(evaluated_events)
        total_articles = sum(len(event.get("event_text", [])) for event in evaluated_events)
        total_qa_pairs = sum(len(event.get("qa_pairs", [])) for event in evaluated_events)
        
        avg_qa_per_topic = total_qa_pairs / total_topics if total_topics > 0 else 0
        avg_qa_per_event = total_qa_pairs / total_events if total_events > 0 else 0
        
        # Per question type statistics
        result_recognition_count = 0
        entity_tracking_count = 0
        
        for event in evaluated_events:
            for qa_pair in event.get("qa_pairs", []):
                question_type = qa_pair.get("question_type", "")
                if question_type == "Result Recognition":
                    result_recognition_count += 1
                elif question_type == "Entity Tracking":
                    entity_tracking_count += 1
        
        total_by_type = result_recognition_count + entity_tracking_count
        result_recognition_ratio = result_recognition_count / total_by_type if total_by_type > 0 else 0
        entity_tracking_ratio = entity_tracking_count / total_by_type if total_by_type > 0 else 0
        
        return {
            "total_topics": total_topics,
            "total_events": total_events,
            "total_articles": total_articles,
            "total_qa_pairs": total_qa_pairs,
            "avg_qa_per_topic": round(avg_qa_per_topic, 2),
            "avg_qa_per_event": round(avg_qa_per_event, 2),
            "per_question_type": {
                "result_recognition_count": result_recognition_count,
                "entity_tracking_count": entity_tracking_count,
                "ratio": {
                    "result_recognition": round(result_recognition_ratio, 3),
                    "entity_tracking": round(entity_tracking_ratio, 3)
                }
            }
        }
    
    def compute_quality_statistics(self, evaluated_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute quality statistics for the dataset
        """
        question_scores = []
        answer_scores = []
        choice_scores = []
        
        for event in evaluated_events:
            for qa_pair in event.get("qa_pairs", []):
                llm_eval = qa_pair.get("llm_evaluation", {})
                question_scores.append(llm_eval.get("question_score", 0))
                answer_scores.append(llm_eval.get("answer_score", 0))
                choice_scores.append(llm_eval.get("choice_score", 0))
        
        def compute_score_distribution(scores: List[int]) -> Dict[str, float]:
            total = len(scores) if scores else 1
            return {
                "score_2_percent": round(len([s for s in scores if s == 2]) / total * 100, 2),
                "score_1_percent": round(len([s for s in scores if s == 1]) / total * 100, 2),
                "score_0_percent": round(len([s for s in scores if s == 0]) / total * 100, 2)
            }
        
        return {
            "question_scores": compute_score_distribution(question_scores),
            "answer_scores": compute_score_distribution(answer_scores),
            "choice_scores": compute_score_distribution(choice_scores)
        }
    
    def evaluate_dataset(self, data: Dict[str, Any], limit_events: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate the entire dataset
        """
        # Handle both "events" and "topics" structures
        if "events" in data:
            events = data["events"]
        elif "topics" in data:
            # Flatten events from topics
            events = []
            for topic in data["topics"]:
                events.extend(topic.get("events", []))
        else:
            raise ValueError("Data must contain either 'events' or 'topics' key")
        
        if limit_events:
            events = events[:limit_events]
        
        evaluated_events = []
        for event in tqdm(events, desc="Evaluating events"):
            evaluated_event = self.evaluate_event(event)
            evaluated_events.append(evaluated_event)
        
        # Compute structural statistics
        structural_stats = self.compute_structural_statistics(data, evaluated_events)
        
        # Compute quality statistics
        quality_stats = self.compute_quality_statistics(evaluated_events)
        
        # Combine statistics
        dataset_evaluation = {
            "structural_statistics": structural_stats,
            "quality_statistics": quality_stats
        }
        
        # Update data structure
        result_data = {**data}
        if "events" in result_data:
            result_data["events"] = evaluated_events
        elif "topics" in result_data:
            # Update events within topics
            event_idx = 0
            for topic in result_data["topics"]:
                topic_events = topic.get("events", [])
                topic["events"] = evaluated_events[event_idx:event_idx + len(topic_events)]
                event_idx += len(topic_events)
        
        return {
            **result_data,
            "llm_evaluation": dataset_evaluation
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA dataset quality using LLM")
    parser.add_argument("--data_json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "together"], help="LLM provider")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for LLM")
    parser.add_argument("--limit_events", type=int, help="Limit number of events to evaluate (for testing)")
    parser.add_argument("--openai_api_key", type=str, help="OpenAI API key")
    parser.add_argument("--together_api_key", type=str, help="Together API key")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_json}")
    with open(args.data_json, 'r') as f:
        data = json.load(f)
    
    # Initialize LLM client
    print(f"Initializing LLM client with {args.provider}:{args.model}")
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        openai_api_key=args.openai_api_key,
        together_api_key=args.together_api_key
    )
    
    # Initialize evaluator
    evaluator = QAEvaluator(llm_client)
    
    # Evaluate dataset
    print("Starting evaluation...")
    evaluated_data = evaluator.evaluate_dataset(data, limit_events=args.limit_events)
    
    # Save results
    input_filename = Path(args.data_json).stem
    output_filename = f"{input_filename}_with_llmeval.json"
    output_path = Path(args.out_dir) / output_filename
    
    print(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(evaluated_data, f, indent=2)
    
    # Print summary
    eval_summary = evaluated_data["llm_evaluation"]
    structural_stats = eval_summary["structural_statistics"]
    quality_stats = eval_summary["quality_statistics"]
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print("\n--- STRUCTURAL STATISTICS ---")
    print(f"Total Topics: {structural_stats['total_topics']}")
    print(f"Total Events: {structural_stats['total_events']}")
    print(f"Total Articles: {structural_stats['total_articles']}")
    print(f"Total QA Pairs: {structural_stats['total_qa_pairs']}")
    print(f"Average QA per Topic: {structural_stats['avg_qa_per_topic']}")
    print(f"Average QA per Event: {structural_stats['avg_qa_per_event']}")
    
    print("\n--- PER QUESTION TYPE STATISTICS ---")
    per_type = structural_stats['per_question_type']
    print(f"Result Recognition QAs: {per_type['result_recognition_count']}")
    print(f"Entity Tracking QAs: {per_type['entity_tracking_count']}")
    print(f"Ratio (Result Recognition : Entity Tracking): {per_type['ratio']['result_recognition']:.3f} : {per_type['ratio']['entity_tracking']:.3f}")
    
    print("\n--- QUALITY STATISTICS ---")
    print("\nQuestion Scores:")
    q_scores = quality_stats['question_scores']
    print(f"  Score 2: {q_scores['score_2_percent']}%")
    print(f"  Score 1: {q_scores['score_1_percent']}%")
    print(f"  Score 0: {q_scores['score_0_percent']}%")
    
    print("\nAnswer Scores:")
    a_scores = quality_stats['answer_scores']
    print(f"  Score 2: {a_scores['score_2_percent']}%")
    print(f"  Score 1: {a_scores['score_1_percent']}%")
    print(f"  Score 0: {a_scores['score_0_percent']}%")
    
    print("\nChoice Scores:")
    c_scores = quality_stats['choice_scores']
    print(f"  Score 2: {c_scores['score_2_percent']}%")
    print(f"  Score 1: {c_scores['score_1_percent']}%")
    print(f"  Score 0: {c_scores['score_0_percent']}%")
    
    print("="*70)

if __name__ == "__main__":
    main()