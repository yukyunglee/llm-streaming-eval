#!/bin/bash

set -e

# ============================================
# ABLATION STUDY: Structured Data Components
# ============================================
# This script runs leave-one-out ablation experiments
# to measure the contribution of each structured data field.
#
# Conditions:
#   - no_people: Full - People
#   - no_location: Full - Location  
#   - no_result: Full - Result
#   - no_event_attrs: Full - Event Attributes너
#
# Stories: 1, 2, 3 only
# ============================================

export TF_CPP_MIN_LOG_LEVEL=2  
export TF_ENABLE_ONEDNN_OPTS=0
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DATA_DIR="./data"
OUTPUT_DIR="./ablation"  # Ablation results go here
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TORCH_HOME"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$HF_DATASETS_CACHE"

# ============================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================

# Select model (uncomment one)
# MODEL="google/gemma-2-2b-it"
MODEL="Qwen/Qwen2.5-72B-Instruct"

# Select tasks to run (comment out to skip)
RUN_CLUSTERING=true
RUN_QA=true
RUN_SUMMARIZATION=true

# Data files for stories 1, 2, 3 only
DATA_FILES=(
    "unified_events_1_multistage.json"
    "unified_events_2_multistage.json"
    "unified_events_3_multistage.json"
)

# Ablation conditions: field to exclude
declare -A ABLATION_CONDITIONS
ABLATION_CONDITIONS["no_people"]="People"
ABLATION_CONDITIONS["no_location"]="Location"
ABLATION_CONDITIONS["no_result"]="Result"
ABLATION_CONDITIONS["no_event_attrs"]="Event Attributes"

# Experiment parameters
WINDOW_SIZE=7
STEP_SIZE=1
LLM_ENGINE="vllm"
MAX_INPUT_TOKENS=8000
RANDOM_SEED=42
TEXTS_PER_EVENT_VALUES=(1 3 5 10)  # Multiple values to test
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
BATCH_SIZE=1

# ============================================
# END CONFIGURATION
# ============================================

echo "=============================================="
echo "ABLATION STUDY: Structured Data Components"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Output Directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Stories: 1, 2, 3"
echo "Ablation conditions: ${!ABLATION_CONDITIONS[@]}"
echo "=============================================="

# Extract model name for folder
MODEL_NAME=$(basename "$MODEL" | sed 's/[^a-zA-Z0-9-]/_/g')

# Function to run a single ablation experiment
run_ablation_experiment() {
    local task=$1
    local data_file=$2
    local ablation_name=$3
    local exclude_field=$4
    local texts_per_event=$5
    
    # Extract story number
    base_name=$(basename "$data_file" .json)
    story_num=$(echo "$base_name" | grep -oP '(?<=_)\d+(?=_|$)' | head -1)
    
    local experiment_name="${task}_story${story_num}_t${texts_per_event}_${ablation_name}"
    local output_subdir="$OUTPUT_DIR/$MODEL_NAME/$task/${experiment_name}_${TIMESTAMP}"
    mkdir -p "$output_subdir"
    
    echo "Running: $experiment_name"
    echo "  Task: $task"
    echo "  Story: $story_num"
    echo "  Texts per event: $texts_per_event"
    echo "  Ablation: $ablation_name (excluding: $exclude_field)"
    
    # Build command
    local cmd="python unified_pipeline.py \
        --task $task \
        --data-file $DATA_DIR/$data_file \
        --window-size $WINDOW_SIZE \
        --step-size $STEP_SIZE \
        --mode sliding \
        --llm-engine $LLM_ENGINE \
        --max-input-tokens $MAX_INPUT_TOKENS \
        --random-seed $RANDOM_SEED \
        --texts-per-event $texts_per_event \
        --batch-size $BATCH_SIZE \
        --output-dir $output_subdir \
        --save-results \
        --log-level INFO \
        --use-structured-data \
        --exclude-fields \"$exclude_field\" \
        --model $MODEL \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --hf-cache-dir $HF_HOME"
    
    # Add task-specific parameters
    if [ "$task" = "summarization" ]; then
        cmd="$cmd --summary-max-tokens 500 --temperature 0.0 --label-type abstract"
    elif [ "$task" = "temporal_qa" ]; then
        cmd="$cmd --qa-max-tokens 10 --temperature 0.0"
    fi
    
    # Execute
    echo "=== COMMAND START ===" | tee -a "$output_subdir/experiment.log"
    echo "Command: $cmd" | tee -a "$output_subdir/experiment.log"
    echo "=== EXECUTION START ===" | tee -a "$output_subdir/experiment.log"
    
    eval "$cmd" 2>&1 | tee -a "$output_subdir/experiment.log"
    exit_code=${PIPESTATUS[0]}
    
    echo "=== EXECUTION END (Exit Code: $exit_code) ===" | tee -a "$output_subdir/experiment.log"
    echo "  ✓ Completed: $experiment_name"
}

# ==============================
# RUN ABLATION EXPERIMENTS
# ==============================

for TEXTS_PER_EVENT in "${TEXTS_PER_EVENT_VALUES[@]}"; do
    echo ""
    echo "=============================================="
    echo "TEXTS_PER_EVENT = $TEXTS_PER_EVENT"
    echo "=============================================="
    
    for data_file in "${DATA_FILES[@]}"; do
        if [[ ! -f "$DATA_DIR/$data_file" ]]; then
            echo "Warning: $DATA_DIR/$data_file not found, skipping..."
            continue
        fi
        
        story_num=$(echo "$data_file" | grep -oP '(?<=_)\d+(?=_|$)' | head -1)
        echo ""
        echo "=============================="
        echo "Processing Story $story_num (t=$TEXTS_PER_EVENT)"
        echo "=============================="
        
        for ablation_name in "${!ABLATION_CONDITIONS[@]}"; do
            exclude_field="${ABLATION_CONDITIONS[$ablation_name]}"
            
            # Clustering
            if [ "$RUN_CLUSTERING" = true ]; then
                run_ablation_experiment "clustering" "$data_file" "$ablation_name" "$exclude_field" "$TEXTS_PER_EVENT"
            fi
            
            # Temporal QA
            if [ "$RUN_QA" = true ]; then
                run_ablation_experiment "temporal_qa" "$data_file" "$ablation_name" "$exclude_field" "$TEXTS_PER_EVENT"
            fi
            
            # Summarization
            if [ "$RUN_SUMMARIZATION" = true ]; then
                run_ablation_experiment "summarization" "$data_file" "$ablation_name" "$exclude_field" "$TEXTS_PER_EVENT"
            fi
        done
    done
done

echo ""
echo "=============================================="
echo "ABLATION EXPERIMENTS COMPLETED"
echo "=============================================="
echo "Results saved in: $OUTPUT_DIR"
