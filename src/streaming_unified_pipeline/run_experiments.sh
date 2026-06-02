#!/bin/bash

set -e

export TF_CPP_MIN_LOG_LEVEL=2  
export TF_ENABLE_ONEDNN_OPTS=0
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=ERROR
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DATA_DIR="./data"
OUTPUT_DIR="./experiments_structured_multistage"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TORCH_HOME"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$HF_HUB_CACHE"
mkdir -p "$HF_DATASETS_CACHE"
echo "vLLM cache directory: $VLLM_CACHE_DIR"
echo "Torch compile cache directory: $TORCHINDUCTOR_CACHE_DIR"

# Data files (all tasks use unified format)
DATA_FILES=(
    "unified_events_25_multistage.json"
    "unified_events_218_multistage.json"
    "unified_events_227_multistage.json" 
    "processed_events_1_multistage.json"
    "processed_events_2_multistage.json"
    "processed_events_3_multistage.json"
)

QA_DATA_FILES=("${DATA_FILES[@]}")
SUMMARIZATION_DATA_FILES=("${DATA_FILES[@]}")
CLUSTERING_DATA_FILES=("${DATA_FILES[@]}")

# Default experiment parameters
WINDOW_SIZE=7
STEP_SIZE=1
LLM_ENGINE="vllm"  # Change to "together" or "vllm" for real experiments
MAX_INPUT_TOKENS=8000  # Fallback for unknown models (actual limit is auto-calculated per model)
RANDOM_SEED=42
TEXTS_PER_EVENT_VALUES=(1 3 5 10)

# vLLM specific settings
VLLM_MODELS=(
    "google/gemma-2-2b-it"
    "google/gemma-3-4b-it"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct" 
    "mistralai/Mistral-Large-Instruct-2411"
)
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
BATCH_SIZE=1

echo "=============================================="
echo "STRUCTURED DATA COMPARISON EXPERIMENTS"
echo "=============================================="
echo "Timestamp: $TIMESTAMP"
echo "Output Directory: $OUTPUT_DIR"
echo "LLM Engine: $LLM_ENGINE"

if [ "$LLM_ENGINE" = "vllm" ]; then
    echo "vLLM Model: $VLLM_MODEL"
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
    echo "HF Cache Directory: $HF_CACHE_DIR"
fi

echo "Window Size: $WINDOW_SIZE"
echo "Max Input Tokens: $MAX_INPUT_TOKENS"
echo "=============================================="

# Function to run a single experiment
run_experiment() {
    local task=$1
    local data_file=$2
    local structured_flag=$3
    local experiment_name=$4
    local current_model=$5
    
    echo "Running: $experiment_name"
    echo "  Task: $task"
    echo "  Data: $data_file"
    echo "  Structured Data: $structured_flag"
    echo "  Texts per event: $TEXTS_PER_EVENT"
    
    # Use the current model being processed
    # Extract model name for folder organization (remove slashes and special chars)
    local model_name=$(basename "$current_model" | sed 's/[^a-zA-Z0-9-]/_/g')
    
    # Improved folder structure: model / task / experiment
    local output_subdir="$OUTPUT_DIR/$model_name/$task/${experiment_name}_${TIMESTAMP}"
    mkdir -p "$output_subdir"
    
    # Build unified command
    local cmd="python unified_pipeline.py \
        --task $task \
        --data-file $DATA_DIR/$data_file \
        --window-size $WINDOW_SIZE \
        --step-size $STEP_SIZE \
        --mode sliding \
        --llm-engine $LLM_ENGINE \
        --max-input-tokens $MAX_INPUT_TOKENS \
        --random-seed $RANDOM_SEED \
        --texts-per-event $TEXTS_PER_EVENT \
        --batch-size $BATCH_SIZE \
        --output-dir $output_subdir \
        --save-results \
        --log-level INFO \
        $structured_flag"
    
    # Add engine-specific parameters
    if [ "$LLM_ENGINE" = "vllm" ]; then
        cmd="$cmd --model $current_model"
        cmd="$cmd --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
        cmd="$cmd --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
        cmd="$cmd --hf-cache-dir $HF_HOME"
    fi
    
    # Execute command with detailed logging
    echo "=== COMMAND START ===" | tee -a "$output_subdir/experiment.log"
    echo "Command: $cmd" | tee -a "$output_subdir/experiment.log" 
    echo "=== EXECUTION START ===" | tee -a "$output_subdir/experiment.log"
    
    eval "$cmd" 2>&1 | tee -a "$output_subdir/experiment.log"
    exit_code=${PIPESTATUS[0]}
    
    echo "=== EXECUTION END (Exit Code: $exit_code) ===" | tee -a "$output_subdir/experiment.log"
    
    echo "  ✓ Completed: $experiment_name"
}

# ==============================
# TASK 1: CLUSTERING EXPERIMENTS
# ==============================
echo "1. CLUSTERING EXPERIMENTS"
echo "------------------------------"

for MODEL in "${VLLM_MODELS[@]}"; do
    echo "Testing with MODEL=$MODEL"
    for TEXTS_PER_EVENT in "${TEXTS_PER_EVENT_VALUES[@]}"; do
        echo "  Testing with TEXTS_PER_EVENT=$TEXTS_PER_EVENT"
        for data_file in "${CLUSTERING_DATA_FILES[@]}"; do
            if [[ -f "$DATA_DIR/$data_file" ]]; then
                # Extract story number from filename (e.g., processed_events_25.json -> 25)
                base_name=$(basename "$data_file" .json)
                story_num=$(echo "$base_name" | grep -oP '(?<=_)\d+(?=_|$)' | head -1)
                
                # Without structured data
                run_experiment "clustering" "$data_file" "" "cluster_${story_num}_t${TEXTS_PER_EVENT}_no_struct" "$MODEL"
                
                # With structured data  
                run_experiment "clustering" "$data_file" "--use-structured-data" "cluster_${story_num}_t${TEXTS_PER_EVENT}_with_struct" "$MODEL"
            else
                echo "Warning: $DATA_DIR/$data_file not found, skipping..."
            fi
        done
    done
done

# ===============================
# TASK 2: SUMMARIZATION EXPERIMENTS
# ===============================
echo "2. SUMMARIZATION EXPERIMENTS"
echo "------------------------------"

for data_file in "${SUMMARIZATION_DATA_FILES[@]}"; do
    if [[ -f "$DATA_DIR/$data_file" ]]; then
        base_name=$(basename "$data_file" .json)
        
        # Without structured data
        run_experiment "summarization" "$data_file" "" "${base_name}_no_struct"
        
        # With structured data
        run_experiment "summarization" "$data_file" "--use-structured-data" "${base_name}_with_struct"
    else
        echo "Warning: $DATA_DIR/$data_file not found, skipping..."
    fi
done

# ==============================
# TASK 3: TEMPORAL QA EXPERIMENTS
# ==============================
echo "3. TEMPORAL QA EXPERIMENTS"
echo "------------------------------"

for MODEL in "${VLLM_MODELS[@]}"; do
    echo "Testing with MODEL=$MODEL"
    for TEXTS_PER_EVENT in "${TEXTS_PER_EVENT_VALUES[@]}"; do
        echo "  Testing with TEXTS_PER_EVENT=$TEXTS_PER_EVENT"
        for data_file in "${QA_DATA_FILES[@]}"; do
            if [[ -f "$DATA_DIR/$data_file" ]]; then
                # Extract just the story number from filename
                base_name=$(basename "$data_file" .json)
                story_num=$(echo "$base_name" | grep -oP '(?<=_)\d+(?=_|$)' | head -1)
                
                # Without structured data
                run_experiment "temporal_qa" "$data_file" "" "qa_${story_num}_t${TEXTS_PER_EVENT}_no_struct" "$MODEL"
                
                # With structured data  
                run_experiment "temporal_qa" "$data_file" "--use-structured-data" "qa_${story_num}_t${TEXTS_PER_EVENT}_with_struct" "$MODEL"
            else
                echo "Warning: $DATA_DIR/$data_file not found, skipping..."
            fi
        done
    done
done
# ==============================
# EXPERIMENT SUMMARY
# ==============================
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "=============================================="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python analyze_structured_data_results.py $OUTPUT_DIR"
echo ""

# Create experiment summary
cat > "$OUTPUT_DIR/experiment_summary_${TIMESTAMP}.txt" << EOF
Structured Data Comparison Experiments
======================================
Date: $(date)
Timestamp: $TIMESTAMP

Configuration:
- Window Size: $WINDOW_SIZE
- Step Size: $STEP_SIZE  
- LLM Engine: $LLM_ENGINE
- Max Input Tokens: $MAX_INPUT_TOKENS
- Random Seed: $RANDOM_SEED

Experiments Run:
EOF

# List all experiment directories
find "$OUTPUT_DIR" -maxdepth 1 -type d -name "*${TIMESTAMP}" | sort >> "$OUTPUT_DIR/experiment_summary_${TIMESTAMP}.txt"

echo "Experiment summary saved: $OUTPUT_DIR/experiment_summary_${TIMESTAMP}.txt"
