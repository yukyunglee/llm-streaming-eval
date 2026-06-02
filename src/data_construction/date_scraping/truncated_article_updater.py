#!/usr/bin/env python3
"""
Truncated Article Updater
=========================
Update truncated article texts (ending with '...') with full content using newspaper3k.

This script is designed to be used after newsapi_enricher.py, which adds articles
with truncated body text (first 200 chars + '...'). This script fetches the full
article content from the URLs.

Pipeline:
    1. newsapi_enricher.py  → Adds articles with truncated text (200 chars + '...')
    2. truncated_article_updater.py → Replaces truncated text with full article

Input: processed_events JSON file with structure:
    {
        "topics": [
            {
                "events": [
                    {
                        "event_urls": ["url1", "url2", ...],
                        "event_text": ["truncated...", "truncated...", ...]
                    }
                ]
            }
        ]
    }

Usage:
    python truncated_article_updater.py --input enriched_events.json
    python truncated_article_updater.py --input events.json --timeout 15 --delay 3
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from newspaper import Article


# =============================================================================
# Article Scraper
# =============================================================================

def scrape_article(
    url: str,
    timeout: int = 10,
    retry_attempts: int = 1,
    retry_delay: int = 5,
    attempt: int = 1
) -> Tuple[Optional[str], Optional[str]]:
    """
    Scrape article with retry logic.
    
    Returns:
        Tuple of (article_text, error_message)
        - On success: (text, None)
        - On failure: (None, error_message)
    """
    try:
        article = Article(url, request_timeout=timeout)
        article.download()
        article.parse()
        return article.text, None
    except Exception as e:
        if attempt < retry_attempts:
            print(f"  Attempt {attempt} failed: {str(e)[:80]} - Retrying in {retry_delay}s")
            time.sleep(retry_delay)
            return scrape_article(url, timeout, retry_attempts, retry_delay, attempt + 1)
        else:
            error_msg = str(e)[:100]
            print(f"  Failed after {retry_attempts} attempts: {error_msg}")
            return None, error_msg


# =============================================================================
# Main Processing
# =============================================================================

def process_data(
    data: Dict,
    timeout: int = 10,
    retry_attempts: int = 1,
    retry_delay: int = 5,
    success_delay: int = 2
) -> Dict:
    """
    Process all events and update truncated texts.
    
    Returns statistics dict.
    """
    stats = {
        "total_events": 0,
        "total_texts": 0,
        "ellipsis_found": 0,
        "scrape_success": 0,
        "scrape_failed": 0,
        "no_url": 0,
    }

    topics = data.get("topics", [])
    total_topics = len(topics)

    # Process each topic
    for topic_idx, topic in enumerate(topics, 1):
        topic_title = topic.get("topic_title", "Unknown")
        events = topic.get("events", [])

        print(f"\n[{topic_idx}/{total_topics}] Topic: {topic_title}")
        print(f"  Events: {len(events)}")

        # Process each event
        for event in events:
            stats["total_events"] += 1

            event_text_list = event.get("event_text", [])
            event_url_list = event.get("event_urls", [])

            stats["total_texts"] += len(event_text_list)

            # Process each text in the event
            for text_idx in range(len(event_text_list)):
                text = event_text_list[text_idx]

                # Check if text ends with '...'
                if text.endswith("..."):
                    stats["ellipsis_found"] += 1

                    # Get URL from same index
                    if text_idx < len(event_url_list) and event_url_list[text_idx]:
                        url = event_url_list[text_idx]
                        print(f"\n  [{stats['ellipsis_found']}] Event {stats['total_events']}, Text {text_idx}")
                        print(f"    URL: {url[:70]}...")

                        # Scrape article
                        scraped_text, error = scrape_article(
                            url, timeout, retry_attempts, retry_delay
                        )

                        if scraped_text:
                            event_text_list[text_idx] = scraped_text
                            stats["scrape_success"] += 1
                            print(f"    ✓ Success ({len(scraped_text)} chars)")
                            time.sleep(success_delay)
                        else:
                            stats["scrape_failed"] += 1
                            print(f"    ✗ Kept original text")
                    else:
                        stats["no_url"] += 1
                        print(f"\n  [{stats['ellipsis_found']}] Event {stats['total_events']}, Text {text_idx}: No URL available")

    return stats


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Update truncated article texts with full content using newspaper3k",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (after newsapi_enricher.py)
  python truncated_article_updater.py --input events_enriched_dedup.json

  # With custom parameters
  python truncated_article_updater.py --input events.json \\
      --timeout 15 --retry-attempts 3 --delay 3

  # Faster processing (less polite to servers)
  python truncated_article_updater.py --input events.json --delay 1
        """
    )
    parser.add_argument("--input", required=True,
                        help="Input JSON file (processed_events format)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: {input}_final.json)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Request timeout in seconds (default: 10)")
    parser.add_argument("--retry-attempts", type=int, default=1,
                        help="Number of retry attempts on failure (default: 1)")
    parser.add_argument("--retry-delay", type=int, default=5,
                        help="Delay between retries in seconds (default: 5)")
    parser.add_argument("--delay", type=int, default=2,
                        help="Delay after successful scrape in seconds (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set output file
    if args.output:
        output_file = args.output
    else:
        # Remove .json and add suffix
        base = args.input.rsplit(".json", 1)[0]
        output_file = f"{base}_final.json"

    print("=" * 70)
    print("Truncated Article Updater")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {output_file}")
    print(f"Timeout: {args.timeout}s")
    print(f"Retry attempts: {args.retry_attempts}")
    print(f"Retry delay: {args.retry_delay}s")
    print(f"Success delay: {args.delay}s")
    print("=" * 70)

    # Load data
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.input}: {e}")
        return

    # Process data
    stats = process_data(
        data=data,
        timeout=args.timeout,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
        success_delay=args.delay
    )

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total events: {stats['total_events']}")
    print(f"Total texts: {stats['total_texts']}")
    print(f"Truncated texts found: {stats['ellipsis_found']}")
    print(f"Successfully scraped: {stats['scrape_success']}")
    print(f"Failed to scrape: {stats['scrape_failed']}")
    print(f"No URL available: {stats['no_url']}")
    if stats["ellipsis_found"] > 0:
        success_rate = stats["scrape_success"] / stats["ellipsis_found"] * 100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
