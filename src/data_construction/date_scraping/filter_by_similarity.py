#!/usr/bin/env python3
"""
Filter events in _dedup.json files based on similarity between event_text and event_sum.
Uses semantic similarity to remove event_urls and event_texts that don't match the event_sum.
"""

import json
import os
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import torch


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_events_batch(
    events: List[Dict[str, Any]],
    model: SentenceTransformer,
    similarity_threshold: float = 0.6
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter multiple events in batch for better performance.
    
    Args:
        events: List of event dictionaries
        model: SentenceTransformer model for embeddings
        similarity_threshold: Minimum similarity threshold
    
    Returns:
        Tuple of (filtered_events, total_stats)
    """
    total_stats = {
        "total_urls": 0,
        "total_texts": 0,
        "filtered_urls": 0,
        "filtered_texts": 0
    }
    
    # Collect all texts that need embedding
    event_data_list = []  # Store metadata for each event
    all_texts = []  # All texts to embed
    text_indices = []  # Map from text index to (event_idx, text_idx, is_sum)
    
    for event_idx, event in enumerate(events):
        event_sum = event.get("event_sum", "")
        event_urls = event.get("event_urls", [])
        event_texts = event.get("event_text", [])
        
        # Initialize stats
        stats = {
            "total_urls": len(event_urls),
            "total_texts": len(event_texts),
            "filtered_urls": 0,
            "filtered_texts": 0
        }
        total_stats["total_urls"] += stats["total_urls"]
        total_stats["total_texts"] += stats["total_texts"]
        
        # Ensure event_texts and event_urls have the same length
        min_length = min(len(event_urls), len(event_texts))
        event_urls = event_urls[:min_length]
        event_texts = event_texts[:min_length]
        
        # Store event data
        event_data = {
            "event": event,
            "event_sum": event_sum,
            "event_urls": event_urls,
            "event_texts": event_texts,
            "stats": stats,
            "sum_embedding_idx": None,
            "text_embedding_indices": []
        }
        
        # Add event_sum to embedding list if it exists
        if event_sum and event_sum.strip():
            event_data["sum_embedding_idx"] = len(all_texts)
            all_texts.append(event_sum)
        
        # Add event_texts to embedding list
        for text_idx, text in enumerate(event_texts):
            if text and text.strip():
                event_data["text_embedding_indices"].append(len(all_texts))
                all_texts.append(text)
            else:
                event_data["text_embedding_indices"].append(None)
        
        event_data_list.append(event_data)
    
    # Generate all embeddings in one batch
    if all_texts:
        print(f"    Generating embeddings for {len(all_texts)} texts in batch...")
        embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=64)
    else:
        embeddings = []
    
    # Process each event with pre-computed embeddings
    filtered_events = []
    
    for event_data in event_data_list:
        event = event_data["event"]
        event_sum = event_data["event_sum"]
        event_urls = event_data["event_urls"]
        event_texts = event_data["event_texts"]
        stats = event_data["stats"]
        
        # If no event_sum, keep everything
        if not event_sum or not event_sum.strip():
            filtered_events.append(event)
            continue
        
        # If no texts or URLs, return as is
        if not event_texts or not event_urls:
            filtered_events.append(event)
            continue
        
        # Get event_sum embedding
        sum_embedding_idx = event_data["sum_embedding_idx"]
        if sum_embedding_idx is None:
            filtered_events.append(event)
            continue
        
        event_sum_embedding = embeddings[sum_embedding_idx:sum_embedding_idx+1]
        
        # Get event_text embeddings and calculate similarities
        filtered_urls = []
        filtered_texts = []
        
        for text_idx, (url, text) in enumerate(zip(event_urls, event_texts)):
            text_embedding_idx = event_data["text_embedding_indices"][text_idx]
            
            if text_embedding_idx is None:
                # Empty text, filter it out
                stats["filtered_urls"] += 1
                stats["filtered_texts"] += 1
                total_stats["filtered_urls"] += 1
                total_stats["filtered_texts"] += 1
                continue
            
            text_embedding = embeddings[text_embedding_idx:text_embedding_idx+1]
            
            # Calculate similarity
            similarity = cosine_similarity(event_sum_embedding, text_embedding)[0][0]
            
            if similarity >= similarity_threshold:
                filtered_urls.append(url)
                filtered_texts.append(text)
            else:
                stats["filtered_urls"] += 1
                stats["filtered_texts"] += 1
                total_stats["filtered_urls"] += 1
                total_stats["filtered_texts"] += 1
        
        # Create filtered event
        filtered_event = event.copy()
        filtered_event["event_urls"] = filtered_urls
        filtered_event["event_text"] = filtered_texts
        filtered_events.append(filtered_event)
        
        # Store event stats for reporting
        event_data["filtered_event"] = filtered_event
        event_data["final_text_count"] = len(filtered_texts)
        event_data["filtered_text_count"] = stats["filtered_texts"]
    
    return filtered_events, total_stats, event_data_list


def filter_dedup_file(
    input_path: str,
    output_path: str,
    similarity_threshold: float = 0.6,
    device: str = None,
    model_name: str = 'Alibaba-NLP/gte-large-en-v1.5'
):
    """
    Filter a _dedup.json file by similarity.
    
    Args:
        input_path: Path to input _dedup.json file
        output_path: Path to output filtered JSON file
        similarity_threshold: Minimum similarity threshold (default: 0.6)
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        model_name: Sentence transformer model name (default: 'Alibaba-NLP/gte-large-en-v1.5')
    """
    print(f"\n{'='*80}")
    print(f"Processing: {input_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading file...")
    data = load_json(input_path)
    topics = data.get("topics", [])
    
    print(f"Found {len(topics)} topics")
    
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
    
    # Process each topic
    total_stats = {
        "total_events": 0,
        "total_urls_before": 0,
        "total_urls_after": 0,
        "total_texts_before": 0,
        "total_texts_after": 0,
        "filtered_urls": 0,
        "filtered_texts": 0
    }
    
    print("\nProcessing topics...")
    for topic_idx, topic in enumerate(topics):
        topic_title = topic.get("topic_title", f"Topic {topic_idx}")
        events = topic.get("events", [])
        
        print(f"\n  Topic {topic_idx + 1}/{len(topics)}: {topic_title}")
        print(f"    Events: {len(events)}")
        
        # Count before
        for event in events:
            total_stats["total_events"] += 1
            total_stats["total_urls_before"] += len(event.get("event_urls", []))
            total_stats["total_texts_before"] += len(event.get("event_text", []))
        
        # Filter all events in this topic in batch
        filtered_events, topic_stats, event_data_list = filter_events_batch(
            events, model, similarity_threshold
        )
        
        # Count after and print per-event details
        for event_idx, (event, event_data) in enumerate(zip(filtered_events, event_data_list)):
            total_stats["total_urls_after"] += len(event.get("event_urls", []))
            total_stats["total_texts_after"] += len(event.get("event_text", []))
            
            # Print event details if any texts were filtered
            initial_text_count = len(event_data["event_texts"])
            filtered_text_count = event_data["filtered_text_count"]
            final_text_count = event_data["final_text_count"]
            
            if filtered_text_count > 0:
                print(f"    Event {event_idx + 1}: {filtered_text_count}/{initial_text_count} texts filtered → {final_text_count} texts remaining")
        
        total_stats["filtered_urls"] += topic_stats["filtered_urls"]
        total_stats["filtered_texts"] += topic_stats["filtered_texts"]
        
        # Update topic with filtered events
        topic["events"] = filtered_events
        
        if topic_stats["filtered_urls"] > 0:
            print(f"    Topic summary: {topic_stats['filtered_texts']} texts filtered, {topic_stats['total_texts'] - topic_stats['filtered_texts']} texts remaining")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("=== Filtering Statistics ===")
    print(f"Total events processed: {total_stats['total_events']}")
    print(f"\nURLs:")
    print(f"  Before: {total_stats['total_urls_before']}")
    print(f"  After: {total_stats['total_urls_after']}")
    print(f"  Filtered: {total_stats['filtered_urls']} ({total_stats['filtered_urls']/max(total_stats['total_urls_before'], 1)*100:.1f}%)")
    print(f"\nEvent Texts:")
    print(f"  Before: {total_stats['total_texts_before']}")
    print(f"  After: {total_stats['total_texts_after']}")
    print(f"  Filtered: {total_stats['filtered_texts']} ({total_stats['filtered_texts']/max(total_stats['total_texts_before'], 1)*100:.1f}%)")
    print(f"  Final count: {total_stats['total_texts_after']} event_texts remaining")
    print("="*80)
    
    # Save filtered data
    print(f"\nSaving filtered data to: {output_path}")
    save_json(data, output_path)
    print("Filtering complete!")


def main():
    """Main function to process files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter events by semantic similarity")
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file path (or use --pattern for multiple files)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: adds _filtered_{threshold} suffix)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="result/processed/*_integrated_topic_final.json",
        help="File pattern to match (default: result/processed/*_integrated_topic_final.json)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity threshold (default: 0.6)"
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
        device = None  # Will be auto-detected in filter_dedup_file
    else:
        device = args.device
    
    scraping_dir = Path(__file__).parent
    
    if args.input:
        # Process single file
        if Path(args.input).is_absolute():
            input_path = args.input
        else:
            input_path = str(scraping_dir / args.input)
        
        if args.output:
            if Path(args.output).is_absolute():
                output_path = args.output
            else:
                output_path = str(scraping_dir / args.output)
        else:
            # Add _filtered suffix
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_filtered_{args.threshold}{input_file.suffix}")
        
        filter_dedup_file(input_path, output_path, similarity_threshold=args.threshold, device=device, model_name=args.model)
    else:
        # Process multiple files by pattern
        if Path(args.pattern).is_absolute():
            pattern_path = Path(args.pattern)
        else:
            pattern_path = scraping_dir / args.pattern
        
        # Extract directory and pattern
        pattern_dir = pattern_path.parent
        pattern = pattern_path.name
        
        dedup_files = list(pattern_dir.glob(pattern))
        
        if not dedup_files:
            print(f"No files found matching pattern: {args.pattern}")
            return
        
        print(f"Found {len(dedup_files)} files:")
        for f in dedup_files:
            print(f"  - {f.name}")

        for dedup_file in dedup_files:
            output_file = dedup_file.parent / f"{dedup_file.stem}_filtered_{args.threshold}.json"
            
            try:
                filter_dedup_file(
                    str(dedup_file),
                    str(output_file),
                    similarity_threshold=args.threshold,
                    device=device,
                    model_name=args.model
                )
            except Exception as e:
                print(f"\nERROR processing {dedup_file.name}:")
                print(f"  {str(e)}")
                continue
        
        print("\n" + "="*80)
        print("✅ All files processed!")
        print("="*80)


if __name__ == "__main__":
    main()

