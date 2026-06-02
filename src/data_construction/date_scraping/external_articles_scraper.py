#!/usr/bin/env python3
"""
External Article Scraper
This script scrapes external articles from Wikipedia event pages.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import requests
from urllib.parse import quote
from bs4 import BeautifulSoup, Tag, NavigableString
from newspaper import Article


# Default configuration
DEFAULT_KEYWORD = "Aftermath"
DEFAULT_WIKI_URL = "https://en.wikipedia.org/wiki/2024_United_States_presidential_election"
DEFAULT_TOPIC = "2024_united_states_presidential_election"


MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

ISO_NUMERIC = re.compile(r"\b(19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2}\b")
MON_NAME = re.compile(
    r"\b(?:(\d{1,2})\s+)?"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|October|November|December)"
    r"\.?,?\s*(\d{1,2})?,?\s*((?:19|20)\d{2})\b",
    re.I,
)


def _normalize_iso(date_str: str) -> Optional[str]:
    if not date_str:
        return None

    # Try numeric format first (2024-12-03, 2024.12.3, 2024/12/3 ...)
    m = ISO_NUMERIC.search(date_str)
    if m:
        parts = re.split(r"[-./]", m.group(0))
        y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
        try:
            return datetime(y, mo, d).date().isoformat()
        except ValueError:
            pass

    # Try month name format (Dec 3, 2024 / 3 December 2024, etc.)
    m = MON_NAME.search(date_str)
    if m:
        day1, mon_name, day2, year = m.groups()
        mon = MONTHS[mon_name.lower()]
        day = int(day2 or day1 or 1)
        try:
            return datetime(int(year), mon, day).date().isoformat()
        except ValueError:
            pass

    return None


def extract_reference_date(ref_li: Tag) -> Optional[str]:
    """Extract date candidate from reference <li> and return as ISO (YYYY-MM-DD)."""
    for sel in ["span.reference-date", "span.date", "time[datetime]", "time"]:
        el = ref_li.select_one(sel)
        if el:
            text = el.get("datetime") or el.get_text(" ", strip=True)
            iso = _normalize_iso(text)
            if iso:
                return iso
            if re.match(r"^\d{4}-\d{2}-\d{2}", text):
                return text[:10]

    text = ref_li.get_text(" ", strip=True)
    iso = _normalize_iso(text)
    return iso


def extract_reference_link(ref_li: Tag) -> Optional[str]:
    """Extract one representative external link from reference <li>."""
    a = ref_li.select_one("a.external")
    if a and a.has_attr("href"):
        return a["href"]

    for a in ref_li.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http://") or href.startswith("https://"):
            return href

    return None


@dataclass
class EventReferenceRow:
    source_article_url: str
    event_title: str
    section_anchor: str
    summary: str
    reference_number: int
    reference_id: str
    reference_date: Optional[str]
    reference_link: Optional[str]


def fetch_html(url: str) -> str:
    headers = {"User-Agent": "KUCS-LLM-Research (contact@example.com)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def build_reference_map(soup: BeautifulSoup) -> Dict[str, Tag]:
    """
    Map <li id="cite_note-..."> from References section to a dictionary.
    key: 'cite_note-55'
    """
    ref_map: Dict[str, Tag] = {}

    # Find span#References or h2#References
    refs_span = soup.select_one("span#References")
    if refs_span:
        refs_h2 = refs_span.find_parent("h2")
    else:
        # Case where h2 has id="References" directly
        refs_h2 = soup.select_one("h2#References")
    
    if not refs_h2:
        # References section not found, but search for cite_note elements directly
        for li in soup.select("li[id^=cite_note-]"):
            li_id = li.get("id")
            if li_id:
                ref_map[li_id] = li
        return ref_map

    # Find all cite_note elements from References h2 to next h2
    next_h2 = refs_h2.find_next_sibling("h2")
    if not next_h2:
        current = refs_h2.find_next()
        while current:
            if isinstance(current, Tag) and current.name == "h2":
                next_h2 = current
                break
            current = current.find_next()
    
    current = refs_h2.find_next()
    while current and current != next_h2:
        if isinstance(current, Tag):
            for li in current.select("li[id^=cite_note-]"):
                li_id = li.get("id")
                if li_id:
                    ref_map[li_id] = li
        current = current.find_next()

    return ref_map


def parse_events_references(url: str, keyword: str = None) -> List[EventReferenceRow]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    base_url = url.split("#")[0]
    ref_map = build_reference_map(soup)

    # Extract keyword from URL anchor if not provided
    if keyword is None:
        if "#" in url:
            keyword = url.split("#")[1]
        else:
            keyword = "Events"  # Default fallback
    
    # Find Events section - handle both span#Events and h2#Events
    keyword_capitalized = keyword[0].upper() + keyword[1:] if keyword else keyword
    events_span = soup.select_one(f"span#{keyword_capitalized}")
    if events_span:
        events_h2 = events_span.find_parent("h2")
    else:
        # Case where h2 has id="Events" directly
        events_h2 = soup.select_one(f"h2#{keyword_capitalized}")
        if not events_h2:
            raise RuntimeError(f"Cannot find #{keyword_capitalized} section")

    # Events area (until next h2)
    # Use find_next since next_siblings alone may miss actual content
    section_nodes: List[Tag] = []
    next_h2 = events_h2.find_next_sibling("h2")
    if not next_h2:
        # Find next h2 using find_next (may not be a sibling)
        current = events_h2.find_next()
        while current:
            if isinstance(current, Tag) and current.name == "h2":
                next_h2 = current
                break
            current = current.find_next()
    
    # Collect all elements from Events h2 to next_h2
    current = events_h2.find_next()
    while current and current != next_h2:
        if isinstance(current, Tag):
            section_nodes.append(current)
        current = current.find_next()

    def is_paragraph(tag: Tag) -> bool:
        return isinstance(tag, Tag) and tag.name == "p" and tag.get_text(strip=True)

    def paragraph_starts_with_bold(p: Tag) -> bool:
        if not p.contents:
            return False
        first = p.contents[0]
        return isinstance(first, Tag) and first.name in ("b", "strong")

    def bold_title(p: Tag) -> str:
        if not p.contents:
            return ""
        first = p.contents[0]
        if isinstance(first, Tag) and first.name in ("b", "strong"):
            return first.get_text(" ", strip=True)
        return ""

    rows: List[EventReferenceRow] = []
    current_event_title = ""
    current_anchor = ""

    for node in section_nodes:
        if not isinstance(node, Tag):
            continue

        # h3/h4 heading → update event title/anchor
        if node.name in ("h3", "h4"):
            span = node.find("span", {"class": "mw-headline"})
            current_anchor = span.get("id") if span and span.has_attr("id") else ""
            current_event_title = node.get_text(" ", strip=True)
            continue

        # Paragraph starting with bold → use as event title
        if is_paragraph(node) and paragraph_starts_with_bold(node):
            t = bold_title(node)
            current_event_title = t or current_event_title
            if not current_anchor and t:
                current_anchor = t.strip().replace(" ", "_")

        # Actual paragraph: summary + sup.reference elements
        if is_paragraph(node):
            summary_text = node.get_text(" ", strip=True)
            if not summary_text:
                continue

            for sup in node.find_all("sup", class_="reference"):
                a = sup.find("a", href=True)
                if not a:
                    continue
                display = sup.get_text("", strip=True)  # [55]
                m = re.search(r"\[(\d+)\]", display)
                if not m:
                    continue
                ref_num = int(m.group(1))

                href = a["href"]  # "#cite_note-55"
                target_id = href[1:] if href.startswith("#") else href
                li = ref_map.get(target_id)
                if not li:
                    # prefix matching fallback
                    for k, v in ref_map.items():
                        if k.startswith(target_id):
                            li = v
                            target_id = k
                            break
                if not li:
                    continue

                ref_date = extract_reference_date(li)
                ref_link = extract_reference_link(li)

                rows.append(
                    EventReferenceRow(
                        source_article_url=base_url,
                        event_title=current_event_title,
                        section_anchor=current_anchor,
                        summary=summary_text,
                        reference_number=ref_num,
                        reference_id=target_id,
                        reference_date=ref_date,
                        reference_link=ref_link,
                    )
                )

    return rows


def normalize_summary(text):
    """Normalize summary text for comparison by removing reference markers."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove all reference markers [55], [ 56 ], [57] etc.
    text = re.sub(r'\s*\[\s*\d+\s*\]\s*', ' ', text)
    return text.strip()


def extract_and_format(wiki_url: str, topic: str, output_json: str, keyword: str = None):
    """Extract events and format them into the output structure."""
    rows = parse_events_references(wiki_url, keyword)
    print(f"Extracted {len(rows)} reference-level rows")

    # Group rows by normalized summary text
    summary_groups = defaultdict(list)
    for r in rows:
        normalized_summary = normalize_summary(r.summary)
        summary_groups[normalized_summary].append(r)

    # Transform to events structure
    events = []
    for summary_text, row_list in summary_groups.items():
        # Get event_title from first row (should be same for all)
        event_title = row_list[0].event_title if row_list else ""
        
        # Use original summary text from first row
        original_summary = row_list[0].summary
        
        # Collect all articles with their dates, removing duplicates by URL
        articles_dict = {}  # url -> article dict
        for r in row_list:
            if r.reference_link:
                # If URL already exists, keep the one with more information (non-None date)
                if r.reference_link not in articles_dict:
                    articles_dict[r.reference_link] = {
                        "reference_number": r.reference_number,
                        "url": r.reference_link,
                        "date": r.reference_date,
                    }
                else:
                    # Update if current row has date and existing doesn't
                    if r.reference_date and not articles_dict[r.reference_link]["date"]:
                        articles_dict[r.reference_link]["date"] = r.reference_date
                        articles_dict[r.reference_link]["reference_number"] = r.reference_number
        
        # Convert to list and sort by reference_number
        articles = sorted(articles_dict.values(), key=lambda x: x["reference_number"])
        
        # Create event structure
        events.append({
            "event_title": event_title,
            "summary": original_summary,
            "articles": articles,
        })

    # Save final data
    summaries_payload = {
        "topic": topic,
        "source_article_url": wiki_url.split("#")[0],
        "events": events,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summaries_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(events)} events to {output_json}")
    return summaries_payload


def scrape_articles(output_json: str):
    """Scrape articles from URLs and update the data structure."""
    # Load existing file
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all unique URLs that need scraping (skip if title and text already exist)
    urls_to_fetch = set()
    url_to_article_refs = {}  # url -> list of article dicts that need updating

    for event in data.get("events", []):
        for article in event.get("articles", []):
            url = article.get("url")
            if url and (not article.get("title") or not article.get("text")):
                urls_to_fetch.add(url)
                if url not in url_to_article_refs:
                    url_to_article_refs[url] = []
                url_to_article_refs[url].append(article)

    print(f"Found {len(urls_to_fetch)} URLs that need scraping")
    print(f"Total articles in file: {sum(len(e.get('articles', [])) for e in data.get('events', []))}")

    error_count = 0
    success_count = 0

    for i, url in enumerate(sorted(urls_to_fetch), 1):
        print(f"[{i}/{len(urls_to_fetch)}] Fetching: {url}")
        try:
            art = Article(url)
            art.download()
            art.parse()

            for article_ref in url_to_article_refs[url]:
                if not article_ref.get("title"):
                    article_ref["title"] = art.title
                if not article_ref.get("text"):
                    article_ref["text"] = art.text
            
            success_count += 1

        except Exception as e:
            error_count += 1
            print(f"  -> FAILED: {e}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved updated data to {output_json}")
    print(f"Success: {success_count}")
    print(f"Errors: {error_count}")


def filter_articles(output_json: str):
    """Filter articles to keep only English text articles."""
    # Korean Unicode range
    korean_pattern = re.compile(r'[가-힣]')

    def is_english_text(text):
        """Check if text is English (no Korean characters)."""
        if not text:
            return False
        return not korean_pattern.search(text)

    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_articles_before = 0
    empty_articles = 0
    english_articles = 0
    korean_articles = 0

    # Filter articles to keep only English text
    for event in data.get("events", []):
        articles = event.get("articles", [])
        filtered_articles = []
        
        for article in articles:
            total_articles_before += 1
            text = article.get("text", "")
            
            # Keep article if text is empty or English (no Korean)
            if not text:
                empty_articles += 1
            elif is_english_text(text):
                filtered_articles.append(article)
                english_articles += 1
            else:
                korean_articles += 1
        
        # Update event with filtered articles
        event["articles"] = filtered_articles

    # Save filtered data
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Calculate unique articles by URL
    unique_urls = set()
    for event in data.get("events", []):
        for article in event.get("articles", []):
            url = article.get("url")
            if url:
                unique_urls.add(url)

    print(f"Total articles before: {total_articles_before}")
    print(f"Articles removed (Empty text): {empty_articles}")
    print(f"Articles removed (Korean text): {korean_articles}")
    print(f"Articles kept (English text): {english_articles}")
    print(f"Unique articles (by URL): {len(unique_urls)}")
    print(f"\nSaved filtered data to {output_json}")


def main():
    """Main execution function."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Scrape external articles from Wikipedia event pages")
    parser.add_argument(
        "--keyword",
        type=str,
        default=DEFAULT_KEYWORD,
        help=f"Section keyword to scrape (default: {DEFAULT_KEYWORD})"
    )
    parser.add_argument(
        "--wiki-url",
        type=str,
        default=DEFAULT_WIKI_URL,
        help=f"Wikipedia page URL (default: {DEFAULT_WIKI_URL})"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help=f"Topic identifier (default: {DEFAULT_TOPIC})"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: external_articles_{topic}_{keyword}.json)"
    )
    
    args = parser.parse_args()
    
    # Construct wiki URL with anchor
    wiki_url = f"{args.wiki_url}#{args.keyword}"
    
    # Generate output filename
    if args.output:
        output_json = args.output
    else:
        dir_id = f"{args.topic}_{args.keyword}"
        output_json = f"external_articles_{dir_id}.json"
    
    # Use relative path from script directory if not absolute
    script_dir = Path(__file__).parent
    if not Path(output_json).is_absolute():
        output_json = str(script_dir / output_json)
    
    # Step 1: Extract and format events
    extract_and_format(wiki_url, args.topic, output_json, args.keyword)
    
    # Step 2: Scrape articles
    scrape_articles(output_json)
    
    # Step 3: Filter articles
    filter_articles(output_json)


if __name__ == "__main__":
    main()

