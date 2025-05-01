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


def encode_chunk(index_chunk_pair, tokenizer_pickle_path, special_tokens, output_dir):
    idx, lines = index_chunk_pair
    tokenizer = Tokenizer.from_pickle(tokenizer_pickle_path, special_tokens=special_tokens)

    chunk_path = os.path.join(output_dir, f"chunk_{idx:06d}.npy")
    if os.path.exists(chunk_path):
        return idx  # Already processed

    token_ids = []
    for line in lines:
        token_ids.extend(tokenizer.encode(line))

    np.save(chunk_path, np.array(token_ids, dtype=np.uint16))
    return idx


def chunked_iterable(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def load_done_chunks(output_dir):
    return {int(fname.split("_")[1].split(".")[0]) for fname in os.listdir(output_dir) if fname.endswith(".npy")}


if __name__ == "__main__":
    args = get_args()

    assert args.save_path.endswith(".npy")
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    temp_chunk_dir = os.path.splitext(args.save_path)[0] + "_chunks"
    os.makedirs(temp_chunk_dir, exist_ok=True)

    print("BPE Tokenization Script with Multiprocessing and Resume Support")
    print(f"Pre-trained Tokenizer: {args.tokenized_dataset_pickle}")
    print(f"Text to tokenize: {args.file_path}")
    print(f"Saving token IDs to: {args.save_path}")
    print(f"Temporary chunk directory: {temp_chunk_dir}")

    start_time = time.time()

    with open(args.file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total lines: {len(lines)}")
    line_chunks = list(enumerate(chunked_iterable(lines, args.chunk_size)))
    done_chunks = load_done_chunks(temp_chunk_dir)

    chunks_to_process = [(i, chunk) for i, chunk in line_chunks if i not in done_chunks]
    print(
        f"Chunks to process: {len(chunks_to_process)} / {len(line_chunks)}. Using {args.num_workers} worker processes."
    )

    worker_fn = partial(
        encode_chunk,
        tokenizer_pickle_path=args.tokenized_dataset_pickle,
        special_tokens=args.special_tokens,
        output_dir=temp_chunk_dir,
    )

    with Pool(args.num_workers) as pool:
        list(tqdm(pool.imap(worker_fn, chunks_to_process), total=len(chunks_to_process)))

    print("Reconstructing final token array...")
    token_ids = []
    for i in range(len(line_chunks)):
        chunk_path = os.path.join(temp_chunk_dir, f"chunk_{i:06d}.npy")
        if not os.path.exists(chunk_path):
            raise RuntimeError(f"Missing chunk: {chunk_path}")
        token_ids.extend(np.load(chunk_path).tolist())

    np.save(args.save_path, np.array(token_ids, dtype=np.uint16))

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Encoding completed. Total tokens: {len(token_ids)}. Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s.")
