#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate llm

python llm/tools/benchmark_attention.py \
--batch-size 8 \
--d-models 16 32 64 128 \
--seq-lens 256 1024 4096 8192 16384 \
--precision bf16 \
--num-warmups 10 \
--num-trials 100 \
--compile