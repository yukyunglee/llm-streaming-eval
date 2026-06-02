# Article Scraping and Processing Pipeline

This directory contains a comprehensive pipeline for scraping, processing, and organizing articles from Wikipedia event pages and external sources.

## Overview

The pipeline consists of several scripts that work together to:
1. Scrape event summaries from Wikipedia Event Portal
2. Scrape articles linked to each summary URL
3. Scrape external articles from Wikipedia event pages
4. Convert and organize articles into structured formats
5. Merge and integrate articles using semantic similarity
6. Clean and filter articles
7. Enrich with additional related articles from NewsAPI
8. Collect full articles based on retrieved URLs
9. Filter articles by semantic similarity

## File Structure

```
scraping/
├── wikipedia_summaries_scraper.py
├── wikipedia_articles_scraper.py
├── external_articles_scraper.py
├── convert_external_to_processed.py
├── merge_empty_topics_by_similarity.py
├── integrate_original_to_topics.py
├── clean_integrated_topics.py
├── newsapi_enricher.py
├── truncated_article_updater.py
├── filter_by_similarity.py
├── requirements.txt
├── README.md
└── result/
    ├── external/          # External article files
    ├── original/          # Original article files
    └── processed/         # Processed event files
```

## Pipeline

### 1. `wikipedia_summaries_scraper.py`

Scrape event summaries from Wikipedia's Portal:Current_events.

**Features:**
- Fetch raw HTML from Wikipedia REST API
- Extract summaries matching a specific topic
- Save monthly JSON files
- Optional merge and deduplication of monthly files

**Usage:**
```bash
python wikipedia_summaries_scraper.py \
    --topic "2024 United States presidential election" \
    --dir-id us_elect \
    --year 2024 \
    --months 1 2 3 4 5 6 7 8 9 10 11 12 \
    --merge
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--topic` | Yes | - | Topic to search for |
| `--dir-id` | Yes | - | Directory identifier |
| `--year` | Yes | - | Year to scrape |
| `--months` | Yes | - | Months to scrape |
| `--merge` | No | False | Merge monthly files |

**Output:** `summaries_{dir_id}/summaries_{dir_id}.json`

---

### 2. `wikipedia_articles_scraper.py`

Scrape full article content from URLs in summaries file using newspaper3k.

**Features:**
- Fetch article title and full text using newspaper3k
- Skip already-processed articles (idempotent)
- Configurable delay between requests
- Preserve original URL on scraping failure

**Usage:**
```bash
python wikipedia_articles_scraper.py --dir-id us_elect --delay 0.5
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dir-id` | Yes | - | Directory identifier |
| `--delay` | No | 0.5 | Delay between requests (seconds) |

**Input:** `summaries_{dir_id}.json`  
**Output:** `articles_scraped_{dir_id}.json`

---

### 3. `external_articles_scraper.py`

Scrapes external articles from Wikipedia event pages.

**Features:**
- Parses Wikipedia event pages to extract references
- Downloads and extracts article content using the `newspaper` library
- Filters articles to keep only English text
- Groups events by summary text

**Usage:**
```bash
python external_articles_scraper.py --keyword "Aftermath" --wiki-url "https://en.wikipedia.org/wiki/2024_United_States_presidential_election" --topic "2024_united_states_presidential_election"
```

**Arguments:**
- `--keyword`: Section keyword to scrape (default: "Aftermath")
- `--wiki-url`: Wikipedia page URL without anchor (default: "https://en.wikipedia.org/wiki/2024_United_States_presidential_election")
- `--topic`: Topic identifier (default: "2024_united_states_presidential_election")
- `--output`: Output JSON file path (default: `external_articles_{topic}_{keyword}.json`)

**Output:**
Creates a JSON file with structure:
```json
{
  "topic": "...",
  "source_article_url": "...",
  "events": [
    {
      "event_title": "...",
      "summary": "...",
      "articles": [
        {
          "reference_number": 1,
          "url": "...",
          "date": "2024-01-01",
          "title": "...",
          "text": "..."
        }
      ]
    }
  ]
}
```

---

### 4. `convert_external_to_processed.py`

Converts external article files to processed events format.

**Features:**
- Groups events by `event_title` (becomes `topic_title`)
- Calculates event dates from article dates
- Filters articles by date matching
- Creates structured topic format

**Usage:**
```bash
python convert_external_to_processed.py --pattern "external_articles_*.json" --output "result/processed/processed_events.json"
```

**Arguments:**
- `--external-dir`: Directory containing external article JSON files (default: "result/external")
- `--pattern`: File pattern to match (required, e.g., "external_articles_*.json")
- `--output`: Output file path (required)

**Output:**
Creates a JSON file with structure:
```json
{
  "keywords": [],
  "topics": [
    {
      "topic_title": "...",
      "topic_category": "",
      "topic_url": "",
      "start_date": "2024-01-01",
      "last_date": "2024-12-31",
      "events": [
        {
          "event_date": "2024-01-01",
          "event_sum": "...",
          "event_urls": ["..."],
          "event_text": ["..."]
        }
      ]
    }
  ]
}
```

---

### 5. `merge_empty_topics_by_similarity.py`

Merges events from topics with empty `topic_title` into topics with non-empty titles using semantic similarity.

**Features:**
- Uses sentence transformers for semantic similarity
- Assigns orphan events to most similar topics
- Updates topic date ranges
- Removes empty topics

**Usage:**
```bash
python merge_empty_topics_by_similarity.py --input "result/processed/processed_events.json" --output "result/processed/processed_events_merged.json"
```

**Arguments:**
- `--input`: Input JSON file path (required)
- `--output`: Output JSON file path (required)
- `--device`: Device to use: 'cuda' for GPU, 'cpu' for CPU, 'auto' for auto-detection (default: auto)
- `--model`: Sentence transformer model name (default: Alibaba-NLP/gte-large-en-v1.5)

**Output:**
Creates a merged JSON file with events assigned to topics based on similarity.

---

### 6. `integrate_original_to_topics.py`

Integrates original articles from the `original` folder into merged processed events using semantic similarity.

**Features:**
- Matches original articles to topics by semantic similarity
- Converts date formats
- Updates topic date ranges
- Preserves article structure

**Usage:**
```bash
python integrate_original_to_topics.py --merged "result/processed/processed_events_merged.json" --original "result/original/articles.json" --output "result/processed/processed_events_integrated.json"
```

**Arguments:**
- `--merged`: Path to merged processed events JSON file (required)
- `--original`: Path to original articles JSON file (required)
- `--output`: Path to output integrated JSON file (required)
- `--device`: Device to use: 'cuda' for GPU, 'cpu' for CPU, 'auto' for auto-detection (default: auto)
- `--model`: Sentence transformer model name (default: Alibaba-NLP/gte-large-en-v1.5)

**Output:**
Creates an integrated JSON file with original articles added to appropriate topics.

---

### 7. `clean_integrated_topics.py`

Cleans integrated events files by filtering and sorting.

**Features:**
- Sorts events by date within each topic
- Filters out events with dates before 2024-01-01
- Removes events with empty/null fields
- Removes events with Korean text
- Updates topic date ranges

**Usage:**
```bash
# Clean a single file
python clean_integrated_topics.py --input "result/processed/processed_events_integrated.json" --output "result/processed/processed_events_cleaned.json"

# Clean multiple files using pattern
python clean_integrated_topics.py --batch "result/processed/*_integrated.json"
```

**Arguments:**
- `--input`: Input JSON file path (required if not using --batch)
- `--output`: Output JSON file path (default: overwrites input file)
- `--batch`: Process multiple files matching pattern (e.g., "result/processed/*_integrated.json")

**Output:**
Creates cleaned JSON files with filtered and sorted events.

---

### 8. `newsapi_enricher.py`

Enrich event data with articles from NewsAPI (EventRegistry) and deduplicate.

**Features:**
- Extract proper nouns from event summaries using spaCy
- Fetch articles from NewsAPI (EventRegistry) by keywords and date
- Fallback keywords support for better coverage
- Jaccard similarity-based deduplication

**Usage:**
```bash
python newsapi_enricher.py \
    --input processed_events.json \
    --api-key "YOUR_API_KEY" \
    --fallback "California AND (wildfire OR fire)"
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Input JSON file |
| `--api-key` | Yes | - | NewsAPI (EventRegistry) key |
| `--fallback` | No | None | Fallback search keywords |
| `--top-n-keywords` | No | 5 | Keywords to extract |
| `--max-articles` | No | 30 | Max articles per event |
| `--threshold` | No | 0.9 | Dedup similarity threshold |

**Output:** `{input}_enriched_dedup.json`

---

### 9. `truncated_article_updater.py`

Replace truncated article texts (ending with `'...'`) with full content using newspaper3k.

This script is designed to be used **after `newsapi_enricher.py`**, which adds articles with truncated body text (first 200 characters + `'...'`).

**Features:**
- Fetch full article content from URLs
- Configurable timeout and retry logic
- Preserve original text on failure

**Usage:**
```bash
python truncated_article_updater.py --input events_enriched_dedup.json
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | Yes | - | Input JSON file |
| `--timeout` | No | 10 | Request timeout (seconds) |
| `--retry-attempts` | No | 1 | Retry attempts |
| `--delay` | No | 2 | Delay after success (seconds) |

**Output:** `{input}_final.json`

---

### 10. `filter_by_similarity.py`

Filters events based on semantic similarity between `event_text` and `event_sum`.

**Features:**
- Uses batch processing for efficiency
- Filters out URLs and texts that don't match event summaries
- Configurable similarity threshold
- Detailed statistics

**Usage:**
```bash
# Filter a single file
python filter_by_similarity.py --input "result/processed/processed_events.json" --output "result/processed/processed_events_filtered.json" --threshold 0.6

# Filter multiple files using pattern
python filter_by_similarity.py --pattern "result/processed/*_integrated_topic_final.json" --threshold 0.6
```

**Arguments:**
- `--input`: Input JSON file path (required if not using --pattern)
- `--output`: Output JSON file path (default: adds `_filtered_{threshold}` suffix)
- `--pattern`: File pattern to match (default: "result/processed/*_integrated_topic_final.json")
- `--threshold`: Similarity threshold (default: 0.6)
- `--device`: Device to use: 'cuda' for GPU, 'cpu' for CPU, 'auto' for auto-detection (default: auto)
- `--model`: Sentence transformer model name (default: Alibaba-NLP/gte-large-en-v1.5)

**Output:**
Creates filtered JSON files with `_filtered_{threshold}.json` suffix.

---

## Data Formats

### Wikipedia Summaries Format

```json
[
  {
    "topic": "2024 United States presidential election",
    "category": "Politics and elections",
    "date": "2024_03_05",
    "subtopic_list": ["Primary elections"],
    "summary": "Event summary text...",
    "articles": {
      "CNN": "https://cnn.com/...",
      "BBC": "https://bbc.com/..."
    }
  }
]
```

### Processed Events Format

```json
{
  "keywords": [],
  "topics": [
    {
      "topic_title": "Topic Name",
      "topic_category": "",
      "topic_url": "",
      "start_date": "2024-01-01",
      "last_date": "2024-12-31",
      "events": [
        {
          "event_date": "2024-01-15",
          "event_sum": "Event summary",
          "event_urls": ["url1", "url2"],
          "event_text": ["text1", "text2"]
        }
      ]
    }
  ]
}
```

## Notes

- Some news sites block automated scraping (Reuters, Fox News, etc.)
- Use appropriate delays to avoid rate limiting
- NewsAPI requires an API key from [EventRegistry](https://eventregistry.org/)
- Typical scraping success rate is 60-80%
- Sentence transformer models require GPU for faster processing (CPU works but slower)
