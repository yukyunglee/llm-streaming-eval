#!/bin/bash

# CheckEval for Multi-Document Summarization
# Uses GPT-4o-mini for binary (Yes/No) evaluation
# Based on TAC/DUC criteria
#
# Usage:
#   bash run_checkeval_summarization.sh                     # All years
#   bash run_checkeval_summarization.sh --year 2025         # 2025 only
#   bash run_checkeval_summarization.sh --limit 5           # Test with 5 files

echo "========================================"
echo "📊 CheckEval: Multi-Document Summarization"
echo "========================================"
echo ""

# Default values
YEAR="all"
MODEL="gpt-4o-mini"
OUTPUT_DIR="./results"
LIMIT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --year)
            YEAR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "  - Year: $YEAR"
echo "  - Model: $MODEL"
echo "  - Output: $OUTPUT_DIR"
if [ -n "$LIMIT" ]; then
    echo "  - Limit: $LIMIT"
fi
echo ""

python3 checkeval_summarization.py \
    --year "$YEAR" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    $LIMIT

echo ""
echo "========================================"
echo "📁 Output files:"
echo "========================================"
ls -la "$OUTPUT_DIR"/*/checkeval_results.csv 2>/dev/null || echo "No CheckEval results found"
echo ""
