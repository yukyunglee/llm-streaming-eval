#!/usr/bin/env python3
"""
Wikipedia Summary Scraper
=========================
Scrape event summaries from Wikipedia Portal:Current_events pages for a specific topic.

Features:
- Fetch raw HTML from Wikipedia REST API
- Extract date-wise summaries for a specified topic
- Save monthly JSON files with summaries and article URLs
- Optional: Merge and deduplicate monthly files into a single output

Output JSON structure:
    {
        "topic": str,
        "category": str,
        "date": str (YYYY_MM_DD),
        "subtopic_list": List[str],
        "summary": str,
        "articles": Dict[str, str]  # {outlet_name: url}
    }

Usage:
    python wikipedia_summaries_scraper.py --topic "2024 United States presidential election" \\
        --dir-id us_elect --year 2024 --months 1 2 3 4 5 6 7 8 9 10 11 12
    
    # With merge
    python wikipedia_summaries_scraper.py --topic "..." --dir-id us_elect --year 2024 --months 1 2 3 --merge
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import quote


# =============================================================================
# I/O Helpers
# =============================================================================

def fetch_wikipedia_html(page_title: str) -> Optional[str]:
    """Fetch HTML from Wikipedia REST API."""
    encoded_title = quote(page_title, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{encoded_title}"
    headers = {"User-Agent": "KUCS (research@example.com)"}
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 200:
        print(f"[OK] Successfully fetched HTML ({len(resp.text):,} characters)")
        return resp.text
    print(f"[Error] HTTP {resp.status_code} while fetching HTML")
    return None


def save_raw_html(html: str, output_path: str) -> None:
    """Save raw HTML to JSON file."""
    data = {"html": html, "length": len(html)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"* Raw HTML saved: {output_path}")


def save_summaries(rows: List[Dict], output_path: str) -> None:
    """Write summaries list to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# =============================================================================
# Date/Month Utilities
# =============================================================================

def convert_month_to_number(month_name: str) -> str:
    """Convert month name to number string. e.g., 'February' -> '02'"""
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    return months.get(month_name.lower(), "00")


def month_num_to_name(month_num: int) -> str:
    """Convert month number to name. e.g., 1 -> 'January'"""
    names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    return names[month_num - 1]


def build_title(year: str, month_num: int) -> str:
    """Build Wikipedia page title for Portal:Current_events."""
    return f"Portal:Current_events/{month_num_to_name(month_num)}_{year}"


def parse_date_from_id(date_id: str) -> Optional[str]:
    """Parse date from HTML id. e.g., '2024_March_7' -> '2024_03_07'"""
    m = re.match(r"(\d{4})_([A-Za-z]+)_(\d{1,2})$", date_id or "")
    if not m:
        return None
    y, mon, d = m.groups()
    mm = convert_month_to_number(mon)
    return f"{y}_{mm}_{int(d):02d}"


# =============================================================================
# HTML Text Extraction Helpers
# =============================================================================

_WS = re.compile(r"\s+")


def normalize(text: Optional[str]) -> str:
    """Normalize whitespace and lowercase text."""
    return _WS.sub(" ", (text or "")).strip().lower()


def text_until_first_ul(li: Tag) -> str:
    """Concatenate text up to the first nested <ul> inside an <li>."""
    parts: List[str] = []
    for node in li.children:
        if isinstance(node, Tag) and node.name == "ul":
            break
        if isinstance(node, Tag):
            parts.append(node.get_text(" ", strip=True))
        elif isinstance(node, NavigableString):
            s = str(node).strip()
            if s:
                parts.append(s)
    # Drop citation markers like [1]
    txt = re.sub(r"\s*\[\d+\]", "", " ".join(parts).strip())
    return _WS.sub(" ", txt).strip()


def extract_article_citations(leaf_li: Tag) -> Dict[str, str]:
    """Collect external article links shown as '(Outlet)'."""
    articles: Dict[str, str] = {}
    for a in leaf_li.find_all("a", class_=lambda c: c and "external" in c):
        name = a.get_text(strip=True).strip("()")
        href = a.get("href") or ""
        if name and href:
            articles[name] = href
    return articles


def collect_summary_and_articles(leaf_li: Tag) -> Tuple[str, Dict[str, str]]:
    """
    Extract summary text and article citations from a leaf <li>.
    Summary = text before the first nested <ul>, excluding external link texts.
    Articles = dict of outlet->url from external links.
    """
    text_parts: List[str] = []
    for node in leaf_li.contents:
        if isinstance(node, Tag) and node.name == "ul":
            break
        if isinstance(node, Tag):
            # Skip external link label in summary
            if node.name == "a" and any("external" in (node.get("class") or []) for _ in [0]):
                continue
            text_parts.append(node.get_text(" ", strip=True))
        elif isinstance(node, NavigableString):
            s = str(node).strip()
            if s:
                text_parts.append(s)
    summary = _WS.sub(" ", " ".join(text_parts)).strip()
    return summary, extract_article_citations(leaf_li)


def ancestor_topic_chain(leaf_li: Tag) -> List[str]:
    """Return [level1, level2, ...] by climbing ancestor <li> nodes."""
    chain: List[str] = []
    cur = leaf_li
    while True:
        parent_li = cur.find_parent("li")
        if not parent_li:
            break
        chain.append(text_until_first_ul(parent_li))
        cur = parent_li
    chain.reverse()
    return chain


# =============================================================================
# Tree Traversal
# =============================================================================

def find_deepest_summaries(root_li: Tag) -> List[Tuple[Tag, List[str]]]:
    """
    Collect leaf <li> under root_li.
    Return list of (leaf_li, trail) where trail excludes the leaf itself.
    """
    results: List[Tuple[Tag, List[str]]] = []

    def dfs(li: Tag, trail: List[str]) -> None:
        sub_ul = li.find("ul", recursive=False)
        if not sub_ul:
            results.append((li, trail))
            return
        for child in sub_ul.find_all("li", recursive=False):
            head = text_until_first_ul(child)
            if child.find("ul", recursive=False):
                dfs(child, trail + ([head] if head else []))
            else:
                # Child is leaf - do not include its own head into trail
                results.append((child, trail))

    root_head = text_until_first_ul(root_li)
    dfs(root_li, [root_head] if root_head else [])
    return results


# =============================================================================
# Category Iterator
# =============================================================================

def iter_category_blocks(content_div: Tag):
    """
    Yield (category, ul) pairs within a single date region.
    Category appears as <p><b>Category</b></p> followed by a sibling <ul>.
    """
    p = content_div.find("p")
    while p:
        b = p.find("b")
        ul = p.find_next_sibling("ul")
        if b and ul:
            yield b.get_text(strip=True), ul
            p = ul.find_next_sibling("p")
        else:
            p = p.find_next_sibling("p") if p else None


# =============================================================================
# Core Extractor
# =============================================================================

def extract_summaries_for_topic(html: str, target_topic: str, month_number: int) -> List[Dict]:
    """
    Traverse each date region. For each category list, find the topic root <li>
    whose head text contains target_topic, then collect all leaf summaries.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict] = []

    # Each date region: <div class="current-events-main vevent" id="YYYY_Month_DD">
    for region in soup.select("div.current-events-main.vevent"):
        content_div = region.select_one("div.current-events-content.description")
        if not content_div:
            continue
        date = parse_date_from_id(region.get("id", "")) or ""

        # Iterate categories within the date region
        for category, ul in iter_category_blocks(content_div):
            # Restrict to the root <li> nodes of this category
            for root_li in ul.find_all("li", recursive=False):
                if normalize(target_topic) not in normalize(text_until_first_ul(root_li)):
                    continue

                # Collect all deepest summaries under the matched topic tree
                for leaf_li, trail in find_deepest_summaries(root_li):
                    topic_text = trail[0] if trail else ""
                    subtopics = trail[1:] if len(trail) > 1 else []
                    summary, articles = collect_summary_and_articles(leaf_li)
                    if not summary:
                        continue
                    rows.append({
                        "topic": topic_text,
                        "category": category,
                        "date": date,
                        "subtopic_list": subtopics,
                        "summary": summary,
                        "articles": articles,
                    })
    return rows


# =============================================================================
# Batch Runner
# =============================================================================

def run_batch(year: str, months: List[int], topic: str, raw_dir: str, out_dir: str) -> Dict:
    """
    Run scraping for specified year and months.
    Returns statistics dict.
    """
    stats = {
        "months_processed": 0,
        "months_failed": 0,
        "total_summaries": 0,
        "total_articles": 0,
    }

    for m in months:
        month_name = month_num_to_name(m)
        month_number = f"{m:02d}"
        page_title = build_title(year, m)

        # 1) Fetch HTML
        html = fetch_wikipedia_html(page_title)
        if not html:
            print(f"[Skip] Failed to fetch: {page_title}")
            stats["months_failed"] += 1
            continue

        raw_path = os.path.join(raw_dir, f"raw_{year}_{month_number}.json")
        save_raw_html(html, raw_path)

        # 2) Extract summaries for the month
        summaries = extract_summaries_for_topic(html, topic, m)

        # Count articles
        article_count = sum(len(s.get("articles", {})) for s in summaries)

        # Save month-level summaries
        month_out = os.path.join(out_dir, f"summaries_{year}_{month_number}.json")
        save_summaries(summaries, month_out)

        stats["months_processed"] += 1
        stats["total_summaries"] += len(summaries)
        stats["total_articles"] += article_count

        print(f"[Done] {month_name} {year}: {len(summaries)} summaries, {article_count} articles")
        print("-" * 60)

    return stats


# =============================================================================
# Merge Function
# =============================================================================

def merge_summaries(out_dir: str, dir_id: str, topic: str) -> Dict:
    """
    Merge all monthly summary files and deduplicate.
    Returns statistics dict.
    """
    out_dir_path = Path(out_dir)

    # Load all JSON files
    all_data = []
    file_count = 0
    
    for json_file in sorted(out_dir_path.glob("summaries_*.json")):
        # Skip merged output file if exists
        if json_file.name == f"summaries_{dir_id}.json":
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    file_count += 1
                else:
                    print(f"Warning: {json_file} is not a list")
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    print(f"\nTotal files processed: {file_count}")
    print(f"Total items loaded: {len(all_data)}")

    # Group by summary text for deduplication
    summary_dict = {}
    for item in all_data:
        summary_text = item.get("summary", "")
        if summary_text not in summary_dict:
            summary_dict[summary_text] = []
        summary_dict[summary_text].append(item)

    # Find duplicates
    duplicate_groups = {s: items for s, items in summary_dict.items() if len(items) > 1}

    # Deduplicate
    final_data = []
    total_duplicates = 0

    for summary_text, items in summary_dict.items():
        if len(items) == 1:
            final_data.append(items[0])
        else:
            total_duplicates += len(items) - 1
            # Prefer items matching the target topic
            topic_items = [item for item in items if item.get("topic") == topic]
            if topic_items:
                final_data.append(topic_items[0])
            else:
                final_data.append(items[0])

    # Save merged file
    output_file = os.path.join(out_dir, f"summaries_{dir_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    # Count total articles
    total_articles = sum(len(item.get("articles", {})) for item in final_data)

    print(f"\n{'='*60}")
    print("Merge Summary:")
    print(f"  Duplicate groups: {len(duplicate_groups)}")
    print(f"  Duplicates removed: {total_duplicates}")
    print(f"  Original total: {len(all_data)}")
    print(f"  After deduplication: {len(final_data)}")
    print(f"  Total articles: {total_articles}")
    print(f"  Saved to: {output_file}")
    print(f"{'='*60}")

    return {
        "files_processed": file_count,
        "original_count": len(all_data),
        "final_count": len(final_data),
        "duplicates_removed": total_duplicates,
        "total_articles": total_articles,
        "output_file": output_file,
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape Wikipedia Portal:Current_events for topic summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape 2024 US election summaries for all months
  python wikipedia_summaries_scraper.py \\
      --topic "2024 United States presidential election" \\
      --dir-id us_elect --year 2024 --months 1 2 3 4 5 6 7 8 9 10 11 12

  # Scrape and merge
  python wikipedia_summaries_scraper.py \\
      --topic "2024 South Korean martial law" \\
      --dir-id sk_martial --year 2024 --months 12 --merge
        """
    )
    parser.add_argument("--topic", required=True, help="Topic string to search for")
    parser.add_argument("--dir-id", required=True, help="Directory identifier for output folders")
    parser.add_argument("--year", required=True, help="Year to scrape (e.g., 2024)")
    parser.add_argument("--months", type=int, nargs="+", required=True,
                        help="Month numbers to scrape (e.g., 1 2 3 or 12)")
    parser.add_argument("--raw-dir", default=None,
                        help="Raw HTML output directory (default: raw_html_{dir_id})")
    parser.add_argument("--out-dir", default=None,
                        help="Summaries output directory (default: summaries_{dir_id})")
    parser.add_argument("--merge", action="store_true",
                        help="Merge monthly files into single output after scraping")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set directory paths
    raw_dir = args.raw_dir or f"raw_html_{args.dir_id}"
    out_dir = args.out_dir or f"summaries_{args.dir_id}"

    # Create directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Wikipedia Summary Scraper")
    print("=" * 60)
    print(f"Topic: {args.topic}")
    print(f"Year: {args.year}")
    print(f"Months: {args.months}")
    print(f"Raw HTML dir: {raw_dir}")
    print(f"Output dir: {out_dir}")
    print("=" * 60)

    # Run scraping
    stats = run_batch(args.year, args.months, args.topic, raw_dir, out_dir)

    print("\n" + "=" * 60)
    print("Scraping Complete!")
    print("=" * 60)
    print(f"Months processed: {stats['months_processed']}")
    print(f"Months failed: {stats['months_failed']}")
    print(f"Total summaries: {stats['total_summaries']}")
    print(f"Total articles: {stats['total_articles']}")
    print("=" * 60)

    # Optional merge
    if args.merge:
        print("\nStarting merge...")
        merge_stats = merge_summaries(out_dir, args.dir_id, args.topic)


if __name__ == "__main__":
    main()
