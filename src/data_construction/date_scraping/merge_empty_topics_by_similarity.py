#!/usr/bin/env python3
"""
Merge events from topics with empty topic_title into topics with non-empty topic_title
based on semantic similarity of event_sum texts.
"""

import json
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from pathlib import Path
import torch


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def merge_empty_topics_by_similarity(
    input_path: str,
    output_path: str,
    device: str = None,
    model_name: str = 'Alibaba-NLP/gte-large-en-v1.5'
):
    """
    Merge events from topics with empty topic_title into topics with non-empty topic_title
    based on semantic similarity.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        model_name: Sentence transformer model name (default: 'Alibaba-NLP/gte-large-en-v1.5')
    """
    # Load data
    data = load_json(input_path)
    topics = data.get("topics", [])
    
    # Separate topics with and without topic_title
    topics_with_title = []
    topics_without_title = []
    
    for topic in topics:
        if topic.get("topic_title", "").strip():
            topics_with_title.append(topic)
        else:
            topics_without_title.append(topic)
    
    print(f"Found {len(topics_with_title)} topics with title")
    print(f"Found {len(topics_without_title)} topics without title")
    
    # Collect all event_sums from topics with title
    topic_texts = []  # List of combined event_sums for each topic
    topic_indices = []  # Map from index to topic
    
    for topic in topics_with_title:
        # Combine all event_sums from this topic into one text
        event_sums = [event.get("event_sum", "") for event in topic.get("events", [])]
        combined_text = " ".join(event_sums)
        topic_texts.append(combined_text)
        topic_indices.append(topic)
    
    if not topic_texts:
        print("No topics with title found. Nothing to merge.")
        save_json(data, output_path)
        return
    
    # Collect event_sums from topics without title
    orphan_events = []  # List of (event, original_topic_index)
    for topic_idx, topic in enumerate(topics_without_title):
        for event in topic.get("events", []):
            orphan_events.append((event, topic_idx))
    
    print(f"Found {len(orphan_events)} orphan events to merge")
    
    if not orphan_events:
        print("No orphan events found. Nothing to merge.")
        save_json(data, output_path)
        return
    
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("No GPU detected, using CPU")
    else:
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load sentence transformer model
    print(f"Loading sentence transformer model: {model_name}...")
    
    # Disable HuggingFace token requirement for public models
    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
    
    model = None
    
    try:
        print(f"Attempting to load: {model_name}")
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        print(f"Successfully loaded {model_name} on {device}")
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Failed to load sentence transformer model.")
        print("="*80)
        print(f"\nModel: {model_name}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Try manually downloading the model first:")
        print(f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{model_name}')\"")
        print("3. If you have HuggingFace authentication issues, try:")
        print("   huggingface-cli login")
        print("   or set HF_TOKEN environment variable")
        print("\nError details:")
        print(f"  {str(e)[:500]}")
        raise RuntimeError(f"Failed to load sentence transformer model '{model_name}'. See error messages above.")
    
    if model is None:
        raise RuntimeError("Failed to initialize sentence transformer model.")
    
    # Combine topic texts and orphan event texts
    all_texts = topic_texts + [event.get("event_sum", "") for event, _ in orphan_events]
    
    # Generate sentence embeddings
    print(f"Generating embeddings for {len(all_texts)} texts...")
    all_vectors = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Split vectors
    topic_vectors = all_vectors[:len(topic_texts)]
    orphan_vectors = all_vectors[len(topic_texts):]
    
    # Calculate similarity for each orphan event
    similarity_matrix = cosine_similarity(orphan_vectors, topic_vectors)
    
    # Assign each orphan event to the most similar topic
    assignments = {}  # topic_index -> list of events
    for orphan_idx, (event, original_topic_idx) in enumerate(orphan_events):
        # Find most similar topic
        similarities = similarity_matrix[orphan_idx]
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]
        
        if most_similar_idx not in assignments:
            assignments[most_similar_idx] = []
        
        assignments[most_similar_idx].append((event, similarity_score))
        
        print(f"Event '{event.get('event_sum', '')[:50]}...' -> Topic '{topic_indices[most_similar_idx].get('topic_title', '')}' (similarity: {similarity_score:.4f})")
    
    # Merge events into topics
    for topic_idx, events_with_scores in assignments.items():
        topic = topic_indices[topic_idx]
        events_to_add = [event for event, _ in events_with_scores]
        
        # Add events to the topic
        topic["events"].extend(events_to_add)
        
        # Update start_date and last_date if needed
        all_dates = []
        for event in topic["events"]:
            if event.get("event_date"):
                all_dates.append(event["event_date"])
        
        if all_dates:
            # Parse dates and find min/max
            from datetime import datetime
            parsed_dates = []
            for date_str in all_dates:
                try:
                    parsed_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                except (ValueError, TypeError):
                    pass
            
            if parsed_dates:
                topic["start_date"] = min(parsed_dates).strftime("%Y-%m-%d")
                topic["last_date"] = max(parsed_dates).strftime("%Y-%m-%d")
        
        print(f"Added {len(events_to_add)} events to topic '{topic.get('topic_title', '')}'")
    
    # Remove topics without title (they should now be empty or we keep them if they still have events)
    # Actually, we should remove the topics that had all their events merged
    final_topics = []
    
    # Add all topics with title
    final_topics.extend(topics_with_title)
    
    # Check topics without title - keep only if they still have events that weren't merged
    # (This shouldn't happen if we merged all events, but just in case)
    for topic in topics_without_title:
        # Check if this topic still has events that weren't merged
        remaining_events = []
        for event in topic.get("events", []):
            # Check if this event was merged
            was_merged = False
            for topic_idx, events_with_scores in assignments.items():
                if event in [e for e, _ in events_with_scores]:
                    was_merged = True
                    break
            
            if not was_merged:
                remaining_events.append(event)
        
        if remaining_events:
            # Keep the topic but update its events
            topic["events"] = remaining_events
            final_topics.append(topic)
            print(f"Kept topic without title with {len(remaining_events)} unmerged events")
    
    # Update data
    data["topics"] = final_topics
    
    # Save result
    save_json(data, output_path)
    
    print(f"\n=== Merge Statistics ===")
    print(f"Total topics after merge: {len(final_topics)}")
    print(f"Topics with title: {len(topics_with_title)}")
    print(f"Topics without title (kept): {len([t for t in final_topics if not t.get('topic_title', '').strip()])}")
    print(f"Total events merged: {len(orphan_events)}")
    print(f"\nSaved merged data to: {output_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge empty topics by semantic similarity")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help="Device to use: 'cuda' for GPU, 'cpu' for CPU, 'auto' for auto-detection (default: auto)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='Alibaba-NLP/gte-large-en-v1.5',
        help="Sentence transformer model name (default: Alibaba-NLP/gte-large-en-v1.5)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = None  # Will be auto-detected in merge_empty_topics_by_similarity
    else:
        device = args.device
    
    # Use relative path from script directory if not absolute
    script_dir = Path(__file__).parent
    if Path(args.input).is_absolute():
        input_path = args.input
    else:
        input_path = str(script_dir / args.input)
    
    if Path(args.output).is_absolute():
        output_path = args.output
    else:
        output_path = str(script_dir / args.output)
    
    merge_empty_topics_by_similarity(input_path, output_path, device=device, model_name=args.model)


if __name__ == "__main__":
    main()

