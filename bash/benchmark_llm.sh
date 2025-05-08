#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate llm


# Benchmark settings
NUM_WARMUPS=5
NUM_TRIALS=100
PRECISION="bf16"
CONTEXT_LENGTH=128

# Array of model configurations with labels
declare -a MODELS=(
    "tiny 512 1344 4 16"
    "small 768 3072 12 12"
    "medium 1024 4096 24 16"
    "large 1280 5120 36 20"
    "xl 1600 6400 48 25"
    "2.7B 2560 10240 32 32"
)

# Output directory
OUTPUT_DIR="/media/bryan/ssd01/expr/llm_from_scratch/benchmarking/question1.3"
mkdir -p $OUTPUT_DIR

# Run benchmark for each configuration
for MODEL in "${MODELS[@]}"; do
    read -r LABEL D_MODEL D_FF NUM_LAYERS NUM_HEADS <<< "$MODEL"
    OUTPUT_FILE="$OUTPUT_DIR/${LABEL}_benchmark.log"
    
    echo "Running benchmark for $LABEL model..." | tee "$OUTPUT_FILE"
    
    python llm/tools/benchmark_llm.py \
        --context-length $CONTEXT_LENGTH \
        --precision $PRECISION \
        --num-warmups $NUM_WARMUPS \
        --num-trials $NUM_TRIALS \
        --d-model $D_MODEL \
        --d-ff $D_FF \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --compile \
        > "$OUTPUT_FILE" 2>&1
    
    NSYS_OUTPUT="$OUTPUT_DIR/${LABEL}_benchmark"
    nsys profile -o $NSYS_OUTPUT \
    --trace=cuda,osrt,nvtx \
    --force-overwrite=true \
    python llm/tools/benchmark_llm.py \
        --context-length $CONTEXT_LENGTH \
        --precision $PRECISION \
        --num-warmups $NUM_WARMUPS \
        --num-trials $NUM_TRIALS \
        --d-model $D_MODEL \
        --d-ff $D_FF \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS
done

read -r LABEL D_MODEL D_FF NUM_LAYERS NUM_HEADS <<< "${MODELS[5]}"
python llm/tools/benchmark_llm.py \
    --save-memory-profile "${OUTPUT_DIR}/${LABEL}_CTX${CONTEXT_LENGTH}_memory_snapshot.pickle" \
    --context-length $CONTEXT_LENGTH \
    --precision $PRECISION \
    --num-warmups $NUM_WARMUPS \
    --num-trials $NUM_TRIALS \
    --d-model $D_MODEL \
    --d-ff $D_FF \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS