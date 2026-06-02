#!/bin/bash
# Unified export script for experiment results by year
# 
# Year mapping:
#   2025: story 1,2,3 from experiments_structured_multistage
#   2016: story 25,218,227 from experiments_structured
#
# Usage:
#   ./run_export_multistage.sh                          # Export all years, all tasks (no BERTScore)
#   ./run_export_multistage.sh --year 2025              # Export 2025 only
#   ./run_export_multistage.sh --year 2016              # Export 2016 only
#   ./run_export_multistage.sh --bertscore              # With BERTScore (GPU needed)
#   ./run_export_multistage.sh --task clustering        # Clustering only
#   ./run_export_multistage.sh --task summarization --bertscore --year 2025
module load miniconda
conda activate /projectnb/tin-lab/yukyung/streaming-bench/envs

set -e

cd "$(dirname "$0")"

# HuggingFace cache settings
export HF_HOME="/projectnb/tin-lab/yukyung/emnlp-rebuttal/Models/hub"
export TRANSFORMERS_CACHE="/projectnb/tin-lab/yukyung/emnlp-rebuttal/Models/hub"
export HF_DATASETS_CACHE="/projectnb/tin-lab/yukyung/emnlp-rebuttal/Models/datasets"

# Parse arguments
TASK="all"
YEAR="all"
USE_BERTSCORE=""
DEVICE="cuda:0"
OUTPUT_DIR="./results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --year)
            YEAR="$2"
            shift 2
            ;;
        --bertscore)
            USE_BERTSCORE="--use-bertscore"
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Exporting experiment results"
echo "========================================"
echo "Year: $YEAR"
echo "  - 2025: story 1,2,3 (experiments_structured_multistage)"
echo "  - 2016: story 25,218,227 (experiments_structured)"
echo "Task: $TASK"
echo "BERTScore: ${USE_BERTSCORE:-disabled}"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo ""

python export_multistage_results.py \
    --task "$TASK" \
    --year "$YEAR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    $USE_BERTSCORE

echo ""
echo "========================================"
echo "Output files:"
echo "========================================"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "No CSV files found yet"
