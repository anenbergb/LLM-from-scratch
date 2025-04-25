#!/bin/bash

python llm/tools/bpe_encode_document.py \
--tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_10k_tinystories.pkl \
--file-path /media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-valid.txt \
--save-path  /media/bryan/ssd01/expr/llm_from_scratch/tokenization/TinyStoriesV2-GPT4-valid.npy

python llm/tools/bpe_encode_document.py \
--tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_10k_tinystories.pkl \
--file-path /media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-train.txt \
--save-path  /media/bryan/ssd01/expr/llm_from_scratch/tokenization/TinyStoriesV2-GPT4-train.npy

python llm/tools/bpe_encode_document.py \
--tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
--file-path /media/bryan/ssd01/data/cs336/owt_valid.txt  \
--save-path  /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_valid.npy

python llm/tools/bpe_encode_document.py \
--tokenized-dataset-pickle /media/bryan/ssd01/expr/llm_from_scratch/tokenization/bpe_32k_owt_train.pkl \
--file-path /media/bryan/ssd01/data/cs336/owt_train.txt  \
--save-path  /media/bryan/ssd01/expr/llm_from_scratch/tokenization/owt_train.npy