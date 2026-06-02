#!/usr/bin/env python3
"""
NewsAPI Enricher
================
Enrich event data with articles from NewsAPI (EventRegistry) and deduplicate.

This script performs a two-phase pipeline:
1. ENRICH: Fetch related articles for each event using keyword extraction
2. DEDUPLICATE: Remove duplicate articles based on Jaccard similarity

Input: processed_events JSON file with structure:
    {
        "topics": [
            {
                "topic_title": "...",
                "events": [
                    {
                        "event_date": "YYYY-MM-DD",
                        "event_sum": "...",
                        "event_urls": [...],
                        "event_text": [...]
                    }
                ]
            }
        ]
    }

Usage:
    python newsapi_enricher.py --input processed_events.json --api-key "YOUR_API_KEY"
    
    # With fallback keywords
    python newsapi_enricher.py --input processed_events.json --api-key "KEY" \\
        --fallback "California AND (wildfire OR fire)"
"""

import argparse
import json
import time
from collections import Counter
from typing import Dict, List, Optional

import requests
import spacy


# =============================================================================
# Setup
# =============================================================================

def load_spacy_model():
    """Load spaCy model for keyword extraction."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


# =============================================================================
# NewsAPI Functions
# =============================================================================

BASE_URL = "https://eventregistry.org/api/v1/article/getArticles"


def newsapi_post(payload: Dict, api_key: str, max_retries: int = 3) -> Optional[Dict]:
    """NewsAPI POST request with retry logic."""
    payload["apiKey"] = api_key
    sleep_time = 60

    for attempt in range(max_retries):
        try:
            r = requests.post(BASE_URL, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                return None
            if r.status_code in (429, 502, 503, 504):
                print(f"    Rate limit/server error, retrying... (attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
                continue
            r.raise_for_status()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_time)

    return None


def extract_keywords(nlp, text: str, top_n: int = 5) -> List[str]:
    """Extract top N proper nouns (PROPN) from text."""
    doc = nlp(text)

    # Count proper nouns
    propn_freq = Counter()
    for token in doc:
        if token.pos_ == "PROPN" and not token.is_stop and len(token.text) > 2:
            propn_freq[token.text] += 1

    # Get top N
    return [word for word, freq in propn_freq.most_common(top_n)]


def fetch_articles_for_event(
    keywords: List[str],
    event_date: str,
    api_key: str,
    fallback_keywords: str = None,
    max_articles: int = 30
) -> List[Dict]:
    """Fetch articles for specific keywords and date."""
    if not keywords and not fallback_keywords:
        return []

    # Create keyword query
    if keywords and fallback_keywords:
        keyword_query = " AND ".join(keywords) + " AND (" + fallback_keywords + ")"
    elif keywords:
        keyword_query = " AND ".join(keywords)
    else:
        keyword_query = fallback_keywords

    payload = {
        "query": {
            "$query": {
                "$and": [
                    {
                        "keyword": keyword_query,
                        "keywordSearchMode": "exact"
                    },
                    {
                        "dateStart": event_date,
                        "dateEnd": event_date,
                        "lang": "eng"
                    }
                ]
            }
        },
        "resultType": "articles",
        "articlesSortBy": "rel",
        "articlesCount": max_articles,
        "includeArticleEventUri": False,
        "articleBodyLen": -1,
    }

    response = newsapi_post(payload, api_key)

    if not response:
        return []

    articles_data = response.get("articles", {})
    results = articles_data.get("results", [])

    return results


# =============================================================================
# Enrich Function
# =============================================================================

def enrich_event_data(
    data: Dict,
    api_key: str,
    nlp,
    fallback_keywords: str = None,
    top_n_keywords: int = 5,
    max_articles_per_event: int = 30,
    delay: float = 0.2
) -> Dict:
    """
    Enrich event data with additional articles from NewsAPI.
    
    Returns statistics dict.
    """
    topics = data.get("topics", [])

    # Statistics
    stats = {
        "total_topics": len(topics),
        "total_events": 0,
        "total_new_articles": 0,
        "used_fallback_count": 0,
        "articles_per_event": [],
    }

    print(f"Total topics: {stats['total_topics']}")
    if fallback_keywords:
        print(f"Fallback keywords: {fallback_keywords}")

    # Process each topic
    for topic_idx, topic in enumerate(topics, 1):
        topic_title = topic.get("topic_title", "Unknown")
        events = topic.get("events", [])

        print(f"\n[{topic_idx}/{stats['total_topics']}] Topic: {topic_title}")
        print(f"  Events: {len(events)}")

        stats["total_events"] += len(events)

        # Process each event
        for event_idx, event in enumerate(events, 1):
            event_date = event.get("event_date", "")
            event_sum = event.get("event_sum", "")
            existing_urls = set(event.get("event_urls", []))

            # Initialize lists if not exist
            if "event_urls" not in event:
                event["event_urls"] = []
            if "event_text" not in event:
                event["event_text"] = []

            print(f"  [{event_idx}/{len(events)}] Date: {event_date}", end="")

            # Extract keywords
            keywords = extract_keywords(nlp, event_sum, top_n_keywords)

            # Decide whether to use fallback
            use_fallback = fallback_keywords and len(keywords) < top_n_keywords

            if not keywords and not fallback_keywords:
                print(" - No keywords, skipping")
                stats["articles_per_event"].append(0)
                continue

            if use_fallback:
                print(f" - Keywords ({len(keywords)}): {', '.join(keywords)} + FALLBACK")
                stats["used_fallback_count"] += 1
            else:
                print(f" - Keywords ({len(keywords)}): {', '.join(keywords)}")

            # Fetch articles
            articles = fetch_articles_for_event(
                keywords,
                event_date,
                api_key,
                fallback_keywords if use_fallback else None,
                max_articles_per_event
            )

            if not articles:
                print(f"    No articles found")
                stats["articles_per_event"].append(0)
                continue

            # Add new articles (skip duplicates)
            new_count = 0
            for article in articles:
                url = article.get("url", "")

                if url and url not in existing_urls:
                    event["event_urls"].append(url)

                    # Add body text (truncated preview)
                    body = article.get("body", "")
                    preview = body[:200] + "..." if len(body) > 200 else body
                    event["event_text"].append(preview)

                    existing_urls.add(url)
                    new_count += 1

            print(f"    Added {new_count} new articles (total found: {len(articles)})")
            stats["articles_per_event"].append(new_count)
            stats["total_new_articles"] += new_count

            time.sleep(delay)

    # Calculate average
    if stats["articles_per_event"]:
        stats["avg_articles_per_event"] = sum(stats["articles_per_event"]) / len(stats["articles_per_event"])
    else:
        stats["avg_articles_per_event"] = 0

    return stats


# =============================================================================
# Deduplication Functions
# =============================================================================

def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    if union == 0:
        return 0.0

    return intersection / union


def deduplicate_event_texts(
    event_texts: List[str],
    event_urls: List[str],
    threshold: float = 0.9
) -> tuple:
    """Remove duplicate texts based on Jaccard similarity."""
    if not event_texts or len(event_texts) != len(event_urls):
        return event_texts, event_urls, 0

    # Keep track of unique texts
    unique_indices = []

    for i, text in enumerate(event_texts):
        is_unique = True

        # Compare with already selected unique texts
        for j in unique_indices:
            similarity = jaccard_similarity(text, event_texts[j])

            if similarity >= threshold:
                is_unique = False
                break

        if is_unique:
            unique_indices.append(i)

    # Extract unique texts and URLs
    unique_texts = [event_texts[i] for i in unique_indices]
    unique_urls = [event_urls[i] for i in unique_indices]

    removed_count = len(event_texts) - len(unique_texts)

    return unique_texts, unique_urls, removed_count


def deduplicate_data(data: Dict, threshold: float = 0.9) -> Dict:
    """
    Deduplicate all events in data.
    
    Returns statistics dict.
    """
    topics = data.get("topics", [])

    # Statistics
    stats = {
        "total_topics": len(topics),
        "total_events": 0,
        "total_original_articles": 0,
        "total_unique_articles": 0,
        "total_removed": 0,
        "top_events": [],
    }

    # Process each topic
    for topic in topics:
        topic_title = topic.get("topic_title", "Unknown")
        events = topic.get("events", [])

        stats["total_events"] += len(events)

        # Process each event
        for event in events:
            event_date = event.get("event_date", "")
            event_texts = event.get("event_text", [])
            event_urls = event.get("event_urls", [])

            original_count = len(event_texts)
            stats["total_original_articles"] += original_count

            # Deduplicate
            unique_texts, unique_urls, removed = deduplicate_event_texts(
                event_texts, event_urls, threshold
            )

            # Update event
            event["event_text"] = unique_texts
            event["event_urls"] = unique_urls

            unique_count = len(unique_texts)
            stats["total_unique_articles"] += unique_count
            stats["total_removed"] += removed

            if removed > 0:
                stats["top_events"].append({
                    "topic": topic_title,
                    "date": event_date,
                    "original": original_count,
                    "unique": unique_count,
                    "removed": removed
                })

    # Sort top events by removed count
    stats["top_events"] = sorted(
        stats["top_events"],
        key=lambda x: x["removed"],
        reverse=True
    )[:5]

    return stats


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enrich event data with NewsAPI articles and deduplicate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python newsapi_enricher.py --input events.json --api-key "YOUR_KEY"

  # With fallback keywords
  python newsapi_enricher.py --input events.json --api-key "KEY" \\
      --fallback "California AND (wildfire OR fire)"

  # Custom parameters
  python newsapi_enricher.py --input events.json --api-key "KEY" \\
      --top-n-keywords 3 --max-articles 50 --threshold 0.85
        """
    )
    parser.add_argument("--input", required=True,
                        help="Input JSON file (processed_events format)")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: {input}_enriched_dedup.json)")
    parser.add_argument("--api-key", required=True,
                        help="NewsAPI (EventRegistry) API key")
    parser.add_argument("--fallback", default=None,
                        help="Fallback keywords for search (e.g., 'California AND wildfire')")
    parser.add_argument("--top-n-keywords", type=int, default=5,
                        help="Number of keywords to extract (default: 5)")
    parser.add_argument("--max-articles", type=int, default=30,
                        help="Max articles per event (default: 30)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Jaccard similarity threshold for dedup (default: 0.9)")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Delay between API calls in seconds (default: 0.2)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set output file
    if args.output:
        output_file = args.output
    else:
        # Remove .json and add suffix
        base = args.input.rsplit(".json", 1)[0]
        output_file = f"{base}_enriched_dedup.json"

    print("=" * 70)
    print("NewsAPI Enricher Pipeline")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {output_file}")
    print(f"Fallback: {args.fallback or 'None'}")
    print(f"Top-N Keywords: {args.top_n_keywords}")
    print(f"Max Articles: {args.max_articles}")
    print(f"Dedup Threshold: {args.threshold}")
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

    # Load spaCy model
    print("\nLoading spaCy model...")
    nlp = load_spacy_model()
    print("Model loaded!")

    # =========================================================================
    # Phase 1: ENRICH
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1: ENRICH")
    print("=" * 70)

    enrich_stats = enrich_event_data(
        data=data,
        api_key=args.api_key,
        nlp=nlp,
        fallback_keywords=args.fallback,
        top_n_keywords=args.top_n_keywords,
        max_articles_per_event=args.max_articles,
        delay=args.delay
    )

    print("\n" + "-" * 70)
    print("Enrich Statistics:")
    print(f"  Total topics: {enrich_stats['total_topics']}")
    print(f"  Total events: {enrich_stats['total_events']}")
    print(f"  New articles added: {enrich_stats['total_new_articles']}")
    print(f"  Fallback used: {enrich_stats['used_fallback_count']} times")
    print(f"  Avg articles per event: {enrich_stats['avg_articles_per_event']:.2f}")

    # =========================================================================
    # Phase 2: DEDUPLICATE
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 2: DEDUPLICATE")
    print("=" * 70)

    dedup_stats = deduplicate_data(data, threshold=args.threshold)

    print("\nDedup Statistics:")
    print(f"  Total topics: {dedup_stats['total_topics']}")
    print(f"  Total events: {dedup_stats['total_events']}")
    print(f"  Original articles: {dedup_stats['total_original_articles']}")
    print(f"  Unique articles: {dedup_stats['total_unique_articles']}")
    print(f"  Removed duplicates: {dedup_stats['total_removed']}")
    if dedup_stats["total_original_articles"] > 0:
        removal_rate = dedup_stats["total_removed"] / dedup_stats["total_original_articles"] * 100
        print(f"  Removal rate: {removal_rate:.2f}%")

    if dedup_stats["top_events"]:
        print("\n  Top events with most duplicates:")
        for evt in dedup_stats["top_events"]:
            print(f"    - {evt['date']} ({evt['topic']}): {evt['removed']} removed ({evt['original']} → {evt['unique']})")

    # =========================================================================
    # Save Result
    # =========================================================================
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
