#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate llm


python llm/tools/train_llm.py \
--overwrite-output-dir \
--lr-warmup-iters 500 \
--max-train-iters 10000


# Vary learning rate

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-lr/2e-4 \
# --overwrite-output-dir \
# --lr-warmup-iters 500 \
# --max-train-iters 10000 \
# --evaluation-iters 1000 \
# --checkpoint-iters 1000 \
# --max-lr 2e-4 \
# --min-lr 2e-6

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-lr/5e-4 \
# --overwrite-output-dir \
# --lr-warmup-iters 500 \
# --max-train-iters 10000 \
# --evaluation-iters 1000 \
# --checkpoint-iters 1000 \
# --max-lr 5e-4 \
# --min-lr 5e-6

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-lr/1e-3 \
# --overwrite-output-dir \
# --lr-warmup-iters 500 \
# --max-train-iters 10000 \
# --evaluation-iters 1000 \
# --checkpoint-iters 1000 \
# --max-lr 1e-3 \
# --min-lr 1e-5

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-lr/1e-2 \
# --overwrite-output-dir \
# --lr-warmup-iters 500 \
# --max-train-iters 10000 \
# --evaluation-iters 1000 \
# --checkpoint-iters 1000 \
# --max-lr 1e-2 \
# --min-lr 1e-4

# Vary batch size

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-batch-size/lr1e-3-bs256 \
# --overwrite-output-dir \
# --lr-warmup-iters 500 \
# --max-train-iters 10000 \
# --evaluation-iters 1000 \
# --checkpoint-iters 1000 \
# --max-lr 1e-3 \
# --min-lr 1e-5 \
# --batch-size 256

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-batch-size/lr1e-3-bs256-iters20k \
# --overwrite-output-dir \
# --lr-warmup-iters 1000 \
# --max-train-iters 20000 \
# --evaluation-iters 1000 \
# --checkpoint-iters 1000 \
# --max-lr 1e-3 \
# --min-lr 1e-5 \
# --batch-size 256

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-batch-size/lr1e-3-bs256-iters50k \
# --overwrite-output-dir \
# --lr-warmup-iters 1000 \
# --max-train-iters 50000 \
# --evaluation-iters 5000 \
# --checkpoint-iters 5000 \
# --max-lr 1e-3 \
# --min-lr 1e-5 \
# --batch-size 256

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-batch-size/lr5e-4-bs256-iters50k \
# --overwrite-output-dir \
# --lr-warmup-iters 1000 \
# --max-train-iters 50000 \
# --evaluation-iters 5000 \
# --checkpoint-iters 5000 \
# --max-lr 5e-4 \
# --min-lr 5e-6 \
# --batch-size 256

# python llm/tools/train_llm.py \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/tune-batch-size/lr2e-4-bs256-iters50k \
# --overwrite-output-dir \
# --lr-warmup-iters 1000 \
# --max-train-iters 50000 \
# --evaluation-iters 5000 \
# --checkpoint-iters 5000 \
# --max-lr 2e-4 \
# --min-lr 2e-6 \
# --batch-size 256