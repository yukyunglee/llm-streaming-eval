# Streaming Unified Pipeline

A unified evaluation pipeline for streaming text understanding with LLMs. Evaluates how well models handle incrementally arriving information through sliding window processing.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with mock engine (no GPU needed, for testing)
python unified_pipeline.py \
  --task temporal_qa \
  --data-file data/unified_events_218.json \
  --llm-engine mock

# Run with vLLM (GPU required)
python unified_pipeline.py \
  --task temporal_qa \
  --data-file data/unified_events_218.json \
  --llm-engine vllm \
  --model google/gemma-2-2b-it \
  --texts-per-event 5 \
  --use-structured-data
```

## Tasks

| Task | Description | Metrics |
|------|-------------|---------|
| `clustering` | Assign articles to events | ARI, NMI |
| `summarization` | Generate summaries from accumulated info | ROUGE-L, BERTScore |
| `temporal_qa` | Answer questions at specific time points | Exact Match, F1 |

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | (required) | `clustering`, `summarization`, `temporal_qa` |
| `--data-file` | (required) | Path to input JSON |
| `--llm-engine` | `mock` | `mock`, `vllm`, `together`, `openai` |
| `--model` | engine default | HuggingFace model name |
| `--texts-per-event` | `1` | Number of texts sampled per event (1, 3, 5, 10) |
| `--use-structured-data` | `False` | Include structured metadata |
| `--window-size` | `5` | Sliding window size |
| `--step-size` | `1` | Window step size |
| `--output-dir` | `results` | Output directory |
| `--save-results` | `False` | Save detailed results to JSON |

## Running Batch Experiments

```bash
# All tasks across models and settings
bash run_experiments.sh

# Ablation experiments
bash run_ablation_experiments.sh
```

Configure environment variables before running batch scripts:
```bash
export HF_TOKEN="your_hf_token"
export HF_HOME="/path/to/hf_cache"
export VLLM_CACHE_ROOT="/path/to/vllm_cache"
```

## Project Structure

```
├── unified_pipeline.py           # Main CLI entry point
├── config.py                     # Configuration management
├── requirements.txt
├── core/
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── llm_client.py             # Unified LLM client (vLLM, OpenAI, Together)
│   └── window_manager.py         # Sliding window logic
├── tasks/
│   ├── base_task.py              # Base task class
│   ├── task1_clustering.py       # Clustering task
│   ├── task2_summarization.py    # Summarization task
│   └── task3_temporal_qa.py      # Temporal QA task
└── evaluation/
    ├── clustering_metrics.py     # ARI, NMI
    ├── summarization_metrics.py  # ROUGE-L, BERTScore
    └── qa_metrics.py             # Exact Match, F1
```

## Data Format

Input JSON follows a unified event structure:
```json
{
  "events": [
    {
      "event_date": "2016-04-21",
      "event_text": ["text1", "text2", ...],
      "People": [...],
      "Location": [...],
      "Result": [...],
      "qa_pairs": [...]
    }
  ]
}
```
