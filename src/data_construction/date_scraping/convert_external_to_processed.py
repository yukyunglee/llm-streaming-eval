#!/usr/bin/env python3
"""
Convert external articles files to processed_events format.
Groups events by event_title (which becomes topic_title).
"""

import json
import glob
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def get_most_common_date(articles: List[Dict]) -> str:
    """
    Get the most common date from articles.
    If there's a tie, return the first date that appears.
    """
    dates = []
    for article in articles:
        if article.get("date"):
            dates.append(article["date"])
    
    if not dates:
        return None
    
    # Count occurrences
    date_counts = Counter(dates)
    max_count = max(date_counts.values())
    
    # Get all dates with max count
    most_common_dates = [date for date, count in date_counts.items() if count == max_count]
    
    # Return the first date that appears in the original list
    for date in dates:
        if date in most_common_dates:
            return date
    
    return most_common_dates[0] if most_common_dates else None


def get_all_dates_from_articles(articles: List[Dict]) -> List[str]:
    """Extract all valid dates from articles."""
    dates = []
    for article in articles:
        if article.get("date"):
            dates.append(article["date"])
    return dates


def convert_external_to_processed(external_files: List[str], output_path: str):
    """
    Convert external articles files to processed_events format.
    
    Args:
        external_files: List of paths to external article JSON files
        output_path: Path to save the converted JSON file
    """
    # Load all external files
    all_events = []
    for file_path in external_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_events.extend(data.get("events", []))
    
    print(f"Loaded {len(all_events)} total events from {len(external_files)} files")
    
    # Group events by event_title (empty string is treated as a single group)
    events_by_title = defaultdict(list)
    for event in all_events:
        event_title = event.get("event_title", "")
        events_by_title[event_title].append(event)
    
    print(f"Found {len(events_by_title)} unique event titles")
    
    # Convert to processed format
    topics = []
    
    for topic_title, events in events_by_title.items():
        # Calculate event_date for each event
        processed_events = []
        all_topic_dates = []
        
        for event in events:
            articles = event.get("articles", [])
            
            # Get most common date for this event
            event_date = get_most_common_date(articles)
            
            # Filter articles to only include those with the same date as event_date
            if event_date:
                filtered_articles = [article for article in articles if article.get("date") == event_date]
            else:
                # If no event_date, include only articles without dates
                filtered_articles = [article for article in articles if not article.get("date")]
            
            # Collect all dates for start_date/last_date calculation (from filtered articles)
            event_dates = get_all_dates_from_articles(filtered_articles)
            all_topic_dates.extend(event_dates)
            
            # Create processed event with only filtered articles
            processed_event = {
                "event_date": event_date,
                "event_sum": event.get("summary", ""),
                "event_urls": [article.get("url", "") for article in filtered_articles if article.get("url")],
                "event_text": [article.get("text", "") for article in filtered_articles if article.get("text")],
            }
            
            processed_events.append(processed_event)
        
        # Calculate start_date and last_date
        if all_topic_dates:
            parsed_dates = [parse_date(d) for d in all_topic_dates if parse_date(d)]
            if parsed_dates:
                start_date = min(parsed_dates).strftime("%Y-%m-%d")
                last_date = max(parsed_dates).strftime("%Y-%m-%d")
            else:
                start_date = None
                last_date = None
        else:
            start_date = None
            last_date = None
        
        # Create topic
        topic = {
            "topic_title": topic_title if topic_title else "",  # Empty string for empty titles
            "topic_category": "",
            "topic_url": "",
            "start_date": start_date,
            "last_date": last_date,
            "events": processed_events
        }
        
        topics.append(topic)
    
    # Create final structure
    result = {
        "keywords": [],
        "topics": topics
    }
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    # Print statistics
    print(f"\n=== Conversion Statistics ===")
    print(f"Total topics: {len(topics)}")
    
    event_counts = [len(topic["events"]) for topic in topics]
    if event_counts:
        avg_events = sum(event_counts) / len(event_counts)
        print(f"Average events per topic: {avg_events:.2f}")
        print(f"Min events per topic: {min(event_counts)}")
        print(f"Max events per topic: {max(event_counts)}")
    else:
        print("No events found")
    
    print(f"\nSaved converted data to: {output_path}")
    
    return result


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert external articles to processed events format")
    parser.add_argument(
        "--external-dir",
        type=str,
        default="result/external",
        help="Directory containing external article JSON files (default: result/external)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="File pattern to match (e.g., 'external_articles_*.json')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    external_dir = script_dir / args.external_dir
    
    if not external_dir.exists():
        print(f"Error: External directory not found: {external_dir}")
        return
    
    external_files = sorted(external_dir.glob(args.pattern))
    
    if not external_files:
        print(f"No files found matching pattern: {args.pattern} in {external_dir}")
        return
    
    print(f"Found {len(external_files)} external files:")
    for f in external_files:
        print(f"  - {f.name}")
    
    # Output path (relative to script directory if not absolute)
    if Path(args.output).is_absolute():
        output_path = args.output
    else:
        output_path = str(script_dir / args.output)
    
    # Convert
    convert_external_to_processed([str(f) for f in external_files], output_path)


if __name__ == "__main__":
    main()

