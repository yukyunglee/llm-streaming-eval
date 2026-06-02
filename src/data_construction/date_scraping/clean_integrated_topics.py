#!/usr/bin/env python3
"""
Clean integrated events files:
1. Sort events by date within each topic
2. Filter out events with start_date before 2024-01-01
3. Remove events with empty/null event_date, event_urls, or event_text
4. Remove events with Korean text in event_text
"""

import json
import re
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    """Save JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def contains_korean(text: str) -> bool:
    """Check if text contains Korean characters."""
    if not text:
        return False
    # Korean Unicode range: \uAC00-\uD7A3 (Hangul syllables)
    korean_pattern = re.compile(r'[\uAC00-\uD7A3]')
    return bool(korean_pattern.search(text))


def is_valid_date(date_str: str) -> bool:
    """Check if date string is valid and not empty."""
    if not date_str or date_str.strip() == "":
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except:
        return False


def clean_integrated_events(input_path: str, output_path: str):
    """
    Clean integrated events file.
    
    Args:
        input_path: Path to input integrated JSON file
        output_path: Path to output cleaned JSON file
    """
    print(f"Loading file: {input_path}")
    data = load_json(input_path)
    topics = data.get("topics", [])
    
    print(f"Found {len(topics)} topics")
    
    total_events_before = 0
    total_events_after = 0
    removed_by_date = 0
    removed_by_empty = 0
    removed_by_korean = 0
    
    # Process each topic
    for topic_idx, topic in enumerate(topics):
        events = topic.get("events", [])
        total_events_before += len(events)
        
        cleaned_events = []
        
        for event in events:
            # Check if event_date is valid and not before 2024-01-01
            event_date = event.get("event_date", "")
            if not is_valid_date(event_date):
                removed_by_empty += 1
                continue
            
            try:
                event_date_obj = datetime.strptime(event_date, "%Y-%m-%d")
                cutoff_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
                if event_date_obj < cutoff_date:
                    removed_by_date += 1
                    continue
            except:
                removed_by_empty += 1
                continue
            
            # Check if event_urls is empty or null
            event_urls = event.get("event_urls", [])
            if not event_urls or len(event_urls) == 0:
                removed_by_empty += 1
                continue
            
            # Check if event_text is empty or null
            event_text = event.get("event_text", [])
            if not event_text or len(event_text) == 0:
                removed_by_empty += 1
                continue
            
            # Check if event_text contains Korean
            has_korean = False
            for text in event_text:
                if contains_korean(text):
                    has_korean = True
                    break
            
            if has_korean:
                removed_by_korean += 1
                continue
            
            # All checks passed, keep the event
            cleaned_events.append(event)
        
        # Sort events by event_date
        cleaned_events.sort(key=lambda x: x.get("event_date", ""))
        
        # Update topic events
        topic["events"] = cleaned_events
        total_events_after += len(cleaned_events)
        
        # Update start_date and last_date based on cleaned events
        if cleaned_events:
            dates = [e.get("event_date", "") for e in cleaned_events if is_valid_date(e.get("event_date", ""))]
            if dates:
                dates.sort()
                topic["start_date"] = dates[0]
                topic["last_date"] = dates[-1]
            else:
                # If no valid dates, set to empty
                topic["start_date"] = ""
                topic["last_date"] = ""
        else:
            # If no events left, set to empty
            topic["start_date"] = ""
            topic["last_date"] = ""
    
    # Remove topics with no events
    topics = [t for t in topics if len(t.get("events", [])) > 0]
    data["topics"] = topics
    
    print("\n" + "="*80)
    print("=== Cleaning Statistics ===")
    print(f"Total events before cleaning: {total_events_before}")
    print(f"Total events after cleaning: {total_events_after}")
    print(f"Removed by date (< 2024-01-01): {removed_by_date}")
    print(f"Removed by empty/null fields: {removed_by_empty}")
    print(f"Removed by Korean text: {removed_by_korean}")
    print(f"Topics before cleaning: {len(data.get('topics', []))}")
    print(f"Topics after cleaning: {len(topics)}")
    print("="*80)
    
    # Save cleaned data
    print(f"\nSaving cleaned data to: {output_path}")
    save_json(data, output_path)
    print("Cleaning complete!")


def main():
    """Main function."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Clean integrated events files")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file path (or use --batch for multiple files)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: overwrites input file)"
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Process multiple files matching pattern (e.g., 'result/processed/*_integrated.json')"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    
    if args.batch:
        # Process multiple files
        if Path(args.batch).is_absolute():
            pattern_path = Path(args.batch)
        else:
            pattern_path = script_dir / args.batch
        
        # Extract directory and pattern
        pattern_dir = pattern_path.parent
        pattern = pattern_path.name
        
        files_to_clean = []
        for file_path in pattern_dir.glob(pattern):
            files_to_clean.append({
                "input": str(file_path),
                "output": str(file_path)  # Overwrite by default
            })
        
        if not files_to_clean:
            print(f"No files found matching pattern: {args.batch}")
            return
        
        for file_info in files_to_clean:
            try:
                print(f"\n{'='*80}")
                print(f"Processing: {file_info['input']}")
                print('='*80)
                clean_integrated_events(file_info['input'], file_info['output'])
            except FileNotFoundError:
                print(f"File not found: {file_info['input']}, skipping...")
            except Exception as e:
                print(f"Error processing {file_info['input']}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Process single file
        if Path(args.input).is_absolute():
            input_path = args.input
        else:
            input_path = str(script_dir / args.input)
        
        if args.output:
            if Path(args.output).is_absolute():
                output_path = args.output
            else:
                output_path = str(script_dir / args.output)
        else:
            output_path = input_path  # Overwrite input
        
        clean_integrated_events(input_path, output_path)


if __name__ == "__main__":
    main()

