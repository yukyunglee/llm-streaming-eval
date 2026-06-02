#!/usr/bin/env python3
"""
Integrate original articles from original folder into merged processed events file.
Uses semantic similarity to assign original articles to existing topics.
"""

import json
import os
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import torch


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def convert_date_format(date_str: str) -> str:
    """Convert date from '2025_01_03' to '2025-01-03'."""
    if not date_str:
        return ""
    try:
        # Handle format like "2025_01_03"
        if "_" in date_str:
            parts = date_str.split("_")
            if len(parts) >= 3:
                return f"{parts[0]}-{parts[1]}-{parts[2]}"
        # If already in correct format, return as is
        return date_str
    except Exception:
        return date_str


def integrate_original_articles(
    merged_path: str,
    original_path: str,
    output_path: str,
    device: str = None,
    model_name: str = 'Alibaba-NLP/gte-large-en-v1.5'
):
    """
    Integrate original articles into merged processed events file.
    
    Args:
        merged_path: Path to merged processed events JSON file
        original_path: Path to original articles JSON file
        output_path: Path to output integrated JSON file
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        model_name: Sentence transformer model name (default: 'Alibaba-NLP/gte-large-en-v1.5')
    """
    # Load data
    print(f"Loading merged file: {merged_path}")
    merged_data = load_json(merged_path)
    topics = merged_data.get("topics", [])
    
    print(f"Loading original file: {original_path}")
    original_articles = load_json(original_path)
    
    if not isinstance(original_articles, list):
        print("Error: Original file should be a list of articles")
        return
    
    print(f"Found {len(topics)} topics in merged file")
    print(f"Found {len(original_articles)} articles in original file")
    
    # Collect all event_sums from each topic to create topic vectors
    topic_texts = []  # List of combined event_sums for each topic
    topic_indices = []  # Map from index to topic
    
    for topic in topics:
        # Combine all event_sums from this topic into one text
        event_sums = [event.get("event_sum", "") for event in topic.get("events", [])]
        combined_text = " ".join(event_sums)
        topic_texts.append(combined_text)
        topic_indices.append(topic)
    
    if not topic_texts:
        print("No topics found in merged file. Nothing to integrate.")
        save_json(merged_data, output_path)
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
    os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
    
    try:
        print(f"Attempting to load: {model_name}")
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        print(f"Successfully loaded {model_name} on {device}")
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Failed to load sentence transformer model.")
        print("="*80)
        print(f"\nModel: {model_name}")
        print("\nError details:")
        print(f"  {str(e)[:500]}")
        raise RuntimeError(f"Failed to load sentence transformer model '{model_name}'. See error messages above.")
    
    # Collect summaries from original articles
    original_summaries = []
    original_article_data = []  # Store full article data for each summary
    
    for article in original_articles:
        summary = article.get("summary", "")
        if summary:
            original_summaries.append(summary)
            original_article_data.append(article)
    
    print(f"Found {len(original_summaries)} articles with summaries to integrate")
    
    if not original_summaries:
        print("No articles with summaries found. Nothing to integrate.")
        save_json(merged_data, output_path)
        return
    
    # Generate embeddings for topic texts and original summaries
    all_texts = topic_texts + original_summaries
    print(f"Generating embeddings for {len(all_texts)} texts...")
    
    embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=32)
    
    # Split embeddings
    topic_embeddings = embeddings[:len(topic_texts)]
    original_embeddings = embeddings[len(topic_texts):]
    
    # Calculate similarity between original summaries and topics
    print("Calculating similarities...")
    similarity_matrix = cosine_similarity(original_embeddings, topic_embeddings)
    
    # Assign each original article to the most similar topic
    assignments = []
    for i, (article, similarities) in enumerate(zip(original_article_data, similarity_matrix)):
        best_topic_idx = np.argmax(similarities)
        best_similarity = similarities[best_topic_idx]
        assignments.append((i, best_topic_idx, best_similarity, article))
        
        topic_title = topic_indices[best_topic_idx].get("topic_title", "")
        summary_preview = article.get("summary", "")[:50] + "..." if len(article.get("summary", "")) > 50 else article.get("summary", "")
        print(f"Article '{summary_preview}' -> Topic '{topic_title}' (similarity: {best_similarity:.4f})")
    
    # Add original articles to assigned topics
    print("\nIntegrating articles into topics...")
    
    for article_idx, topic_idx, similarity, article in assignments:
        topic = topic_indices[topic_idx]
        
        # Extract data from original article
        date_str = article.get("date", "")
        event_date = convert_date_format(date_str)
        summary = article.get("summary", "")
        
        # Extract URLs and texts from articles dictionary
        articles_dict = article.get("articles", {})
        event_urls = []
        event_texts = []
        
        for source_name, article_data in articles_dict.items():
            if isinstance(article_data, dict):
                url = article_data.get("url", "")
                text = article_data.get("text", "")
                if url:
                    event_urls.append(url)
                if text:
                    event_texts.append(text)
            elif isinstance(article_data, str):
                # If article_data is just a URL string
                event_urls.append(article_data)
        
        # Create new event
        new_event = {
            "event_date": event_date,
            "event_sum": summary,
            "event_urls": event_urls,
            "event_text": event_texts
        }
        
        # Add event to topic
        if "events" not in topic:
            topic["events"] = []
        topic["events"].append(new_event)
        
        # Update start_date and last_date
        if event_date:
            try:
                event_date_obj = datetime.strptime(event_date, "%Y-%m-%d")
                start_date = topic.get("start_date", "")
                last_date = topic.get("last_date", "")
                
                if start_date:
                    try:
                        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                        if event_date_obj < start_date_obj:
                            topic["start_date"] = event_date
                    except:
                        topic["start_date"] = event_date
                else:
                    topic["start_date"] = event_date
                
                if last_date:
                    try:
                        last_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
                        if event_date_obj > last_date_obj:
                            topic["last_date"] = event_date
                    except:
                        topic["last_date"] = event_date
                else:
                    topic["last_date"] = event_date
            except:
                pass
    
    # Count statistics
    total_events_added = len(assignments)
    events_per_topic = {}
    for article_idx, topic_idx, similarity, article in assignments:
        topic_title = topic_indices[topic_idx].get("topic_title", "")
        events_per_topic[topic_title] = events_per_topic.get(topic_title, 0) + 1
    
    print("\n" + "="*80)
    print("=== Integration Statistics ===")
    print(f"Total articles integrated: {total_events_added}")
    print(f"Topics updated: {len(events_per_topic)}")
    print("\nEvents added per topic:")
    for topic_title, count in sorted(events_per_topic.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic_title}: {count} events")
    print("="*80)
    
    # Save integrated data
    print(f"\nSaving integrated data to: {output_path}")
    save_json(merged_data, output_path)
    print("Integration complete!")


def main():
    """Main function."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Integrate original articles into merged processed events")
    parser.add_argument(
        "--merged",
        type=str,
        required=True,
        help="Path to merged processed events JSON file"
    )
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original articles JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output integrated JSON file"
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
        device = None  # Will be auto-detected in integrate_original_articles
    else:
        device = args.device
    
    # Use relative path from script directory if not absolute
    script_dir = Path(__file__).parent
    
    if Path(args.merged).is_absolute():
        merged_path = args.merged
    else:
        merged_path = str(script_dir / args.merged)
    
    if Path(args.original).is_absolute():
        original_path = args.original
    else:
        original_path = str(script_dir / args.original)
    
    if Path(args.output).is_absolute():
        output_path = args.output
    else:
        output_path = str(script_dir / args.output)
    
    integrate_original_articles(merged_path, original_path, output_path, device=device, model_name=args.model)


if __name__ == "__main__":
    main()

