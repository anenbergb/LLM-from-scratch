#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate llm

python llm/tools/train_llm.py \
--overwrite-output-dir \
--lr-warmup-iters 1000 \
--max-train-iters 1000 \
--limit-val-iters 100 \
--evaluation-iters 100