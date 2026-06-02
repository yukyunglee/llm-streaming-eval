<div align="center">

# Can Structural Cues Save LLMs? Evaluating Language Models in Massive Document Streams (KDD 2026)

[![Paper](https://img.shields.io/badge/Paper-KDD%202026-blue)](https://kdd2026.kdd.org/) [![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2603.19250)

**Yukyung Lee**¹, **Yebin Lim**²*, **Woojun Jung**²*, **Wonjun Choi**², **Susik Yoon**¹†

¹Boston University, ²Korea University

*Equal contribution, †Corresponding author

</div>

## Repository Structure

```
├── src/
│   ├── data_construction/       # Dataset building tools
│   │   ├── date_scraping/       # Event date extraction
│   │   └── temporal_qa/         # QA pair generation
│   └── streaming_unified_pipeline/  # Main evaluation pipeline
│       ├── unified_pipeline.py  # CLI entry point
│       ├── core/                # Data loader, LLM client, window manager
│       ├── tasks/               # Clustering, summarization, temporal QA
│       └── evaluation/          # Metrics (F1, ROUGE, Acc, etc.)
└── dataset/                     # Data files
```

## Quick Start

```bash
cd src/streaming_unified_pipeline
pip install -r requirements.txt

# Full run with vLLM (GPU required)
python unified_pipeline.py \
  --task temporal_qa \
  --data-file data/unified_events_218.json \
  --llm-engine vllm \
  --model google/gemma-2-2b-it \
  --texts-per-event 5 \
  --use-structured-data
```

See `src/streaming_unified_pipeline/README.md` for detailed usage and configuration.