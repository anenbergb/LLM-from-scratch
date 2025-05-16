#!/bin/bash

eval "$(conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=0
conda activate llm


# python llm/tools/train_llm.py \
# --overwrite-output-dir \
# --lr-warmup-iters 500 \
# --max-train-iters 10000


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

# python llm/tools/train_llm.py \
# --train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
# --val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
# --tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-weight-sharing/tiny-50k \
# --overwrite-output-dir \
# --lr-warmup-iters 1000 \
# --max-train-iters 50000 \
# --evaluation-iters 5000 \
# --checkpoint-iters 5000

# python llm/tools/train_llm.py \
# --train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
# --val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
# --tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-weight-sharing/tiny-50k-weight-sharing-lm-head \
# --overwrite-output-dir \
# --lr-warmup-iters 1000 \
# --max-train-iters 50000 \
# --evaluation-iters 5000 \
# --checkpoint-iters 5000 \
# --weight-sharing

# python llm/tools/train_llm.py \
# --train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
# --val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
# --tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-weight-sharing/tiny-50k-weight-sharing-embedding \
# --overwrite-output-dir \
# --lr-warmup-iters 1000 \
# --max-train-iters 50000 \
# --evaluation-iters 5000 \
# --checkpoint-iters 5000

# Small model

# python llm/tools/train_llm.py \
# --train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
# --val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
# --tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/small-100k \
# --overwrite-output-dir \
# --lr-warmup-iters 2000 \
# --max-train-iters 100000 \
# --evaluation-iters 10000 \
# --checkpoint-iters 10000 \
# --context-length 256 \
# --batch-size 64 --val-batch-size 64 \
# --weight-sharing \
# --d-model 768 --d-ff 3072 --num-layers 12 --num-heads 12

# python llm/tools/train_llm.py \
# --train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
# --val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
# --tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/tiny-100k \
# --overwrite-output-dir \
# --lr-warmup-iters 2000 \
# --max-train-iters 100000 \
# --evaluation-iters 10000 \
# --checkpoint-iters 10000 \
# --context-length 256 \
# --batch-size 64 --val-batch-size 64 \
# --weight-sharing

# python llm/tools/train_llm.py \
# --train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
# --val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
# --tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/medium-100k-lr3e-4 \
# --resume-from-checkpoint /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/medium-100k-lr3e-4/checkpoint_20000_crashed.pt \
# --lr-warmup-iters 5000 \
# --max-train-iters 100000 \
# --evaluation-iters 10000 \
# --checkpoint-iters 10000 --log-iters 100 \
# --context-length 256 \
# --batch-size 32 --val-batch-size 32 \
# --weight-sharing \
# --max-lr 3e-4 --min-lr 3e-6 \
# --d-model 1024 --d-ff 4096 --num-layers 24 --num-heads 16

# python llm/tools/train_llm.py \
# --train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
# --val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
# --tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
# --output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/large-100k-lr2e-4 \
# --resume-from-checkpoint /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/large-100k-lr2e-4/checkpoint_60000.pt \
# --lr-warmup-iters 5000 \
# --max-train-iters 100000 \
# --evaluation-iters 10000 \
# --checkpoint-iters 10000 --log-iters 100 \
# --context-length 256 \
# --batch-size 8 --val-batch-size 8 \
# --weight-sharing \
# --max-lr 2e-4 --min-lr 2e-6 \
# --d-model 1280 --d-ff 5120 --num-layers 36 --num-heads 20

export PYTORCH_CUDA_GRAPH_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python llm/tools/train_llm.py \
--train-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy \
--val-dataset /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy \
--tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
--output-dir /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/medium-500k-lr5e-4 \
--resume-from-checkpoint /media/bryan/ssd01/expr/llm_from_scratch/owl-model-size/medium-500k-lr5e-4/checkpoint_170000.pt \
--num-checkpoints 1 \
--lr-warmup-iters 10000 \
--max-train-iters 500000 \
--evaluation-iters 50000 \
--checkpoint-iters 50000 --log-iters 100 \
--context-length 256 \
--batch-size 32 --val-batch-size 32 \
--weight-sharing \
--max-lr 5e-4 --min-lr 5e-6 \
--d-model 1024 --d-ff 4096 --num-layers 24 --num-heads 16