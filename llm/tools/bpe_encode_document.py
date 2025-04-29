import argparse
import os
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from llm.tokenization import Tokenizer


def get_args():
    parser = argparse.ArgumentParser(
        description="Given a previously-trained tokenizer, encode a document and save the token IDs as a Numpy array"
    )
    parser.add_argument("--tokenized-dataset-pickle", type=str, help="Path to a tokenized dataset .pkl file")
    parser.add_argument("--file-path", type=str, required=True, help="Path to the text file to encode")
    parser.add_argument("--special-tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the encoded token IDs .npy file")
    parser.add_argument("--num-workers", type=int, default=cpu_count(), help="Number of multiprocessing workers")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of lines per chunk to send to workers")
    return parser.parse_args()


def encode_chunk(tokenizer_pickle_path, special_tokens, lines):
    """Worker function: loads tokenizer, encodes a list of lines."""
    tokenizer = Tokenizer.from_pickle(tokenizer_pickle_path, special_tokens=special_tokens)
    token_ids = []
    for line in lines:
        token_ids.extend(tokenizer.encode(line))
    return token_ids


def chunked_iterable(iterable, chunk_size):
    """Yield successive chunks from iterable."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


if __name__ == "__main__":
    args = get_args()

    assert args.save_path.endswith(".npy")
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("BPE Tokenization Script with Multiprocessing")
    print(f"Pre-trained Tokenizer: {args.tokenized_dataset_pickle}")
    print(f"Text to tokenize: {args.file_path}")
    print(f"Saving token IDs to: {args.save_path}")

    start_time = time.time()

    token_ids = []

    with open(args.file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")

    # Create chunks
    line_chunks = list(chunked_iterable(lines, args.chunk_size))

    print(f"Total chunks: {len(line_chunks)}. Using {args.num_workers} worker processes.")

    # Define a partial function to avoid passing all arguments to each worker
    worker_fn = partial(encode_chunk, args.tokenized_dataset_pickle, args.special_tokens)

    # Parallel encoding
    with Pool(args.num_workers) as pool:
        for encoded_chunk in tqdm(pool.imap(worker_fn, line_chunks), total=len(line_chunks)):
            token_ids.extend(encoded_chunk)

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Encoding completed. Total tokens: {len(token_ids)}. Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s.")

    # Save
    token_array = np.array(token_ids, dtype=np.uint16)
    print(f"Saving token IDs to {args.save_path}")
    np.save(args.save_path, token_array)
