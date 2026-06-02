#!/usr/bin/env python3
"""
Wikipedia Articles Scraper
==========================
Scrape article content from URLs collected by wikipedia_summaries_scraper.

This script reads a summaries JSON file and uses newspaper3k to fetch
the full article text for each URL in the 'articles' field.

Input: summaries_{dir_id}.json (from wikipedia_summaries_scraper.py)
Output: articles_scraped_{dir_id}.json

Usage:
    python wikipedia_articles_scraper.py --dir-id us_elect
    python wikipedia_articles_scraper.py --dir-id sk_martial --delay 1.0
"""

import argparse
import json
import time
from typing import Dict, List

from newspaper import Article


# =============================================================================
# Article Scraper
# =============================================================================

def scrape_articles(data: List[Dict], delay: float = 0.5) -> Dict:
    """
    Scrape article content for all URLs in the data.
    
    Args:
        data: List of summary dicts with 'articles' field containing {source: url}
        delay: Delay between requests in seconds
    
    Returns:
        Statistics dict with counts
    """
    stats = {
        "total_articles": 0,
        "success": 0,
        "errors": 0,
        "skipped": 0,
    }

    for summary in data:
        articles = summary.get("articles", {})

        for source, url in list(articles.items()):
            # Skip if already processed (url is already a dict)
            if isinstance(url, dict):
                stats["skipped"] += 1
                continue

            stats["total_articles"] += 1

            try:
                article = Article(url)
                article.download()
                article.parse()

                # Update structure: replace URL string with dict
                articles[source] = {
                    "url": url,
                    "title": article.title,
                    "text": article.text,
                }
                stats["success"] += 1

                time.sleep(delay)  # Be polite to servers

            except Exception as e:
                stats["errors"] += 1
                print(f"Error [{source}]: {url[:50]}... - {str(e)[:50]}")
                # Keep original URL on error

    return stats


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape article content from Wikipedia summary URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape articles for US election summaries
  python wikipedia_articles_scraper.py --dir-id us_elect

  # With custom delay
  python wikipedia_articles_scraper.py --dir-id sk_martial --delay 1.0
        """
    )
    parser.add_argument("--dir-id", required=True,
                        help="Directory identifier (reads summaries_{dir_id}.json)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds (default: 0.5)")
    return parser.parse_args()


def main():
    args = parse_args()

    # File paths based on dir_id
    input_file = f"summaries_{args.dir_id}.json"
    output_file = f"articles_scraped_{args.dir_id}.json"

    print("=" * 60)
    print("Wikipedia Articles Scraper")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Delay: {args.delay}s")
    print("=" * 60)

    # Load data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} summaries")
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
        return

    # Scrape articles
    print("-" * 60)
    stats = scrape_articles(data, delay=args.delay)

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Scraping Complete!")
    print("=" * 60)
    print(f"Total articles: {stats['total_articles']}")
    print(f"Success: {stats['success']}")
    print(f"Errors: {stats['errors']}")
    if stats["skipped"] > 0:
        print(f"Skipped (already processed): {stats['skipped']}")
    print(f"Saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
