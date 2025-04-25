import argparse
import os
import time  # Import time module
from tqdm import tqdm
from llm.tokenization import Tokenizer
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="Given a previously-trained tokenizer, encode a document and save the token IDs as a Numpy array"
    )
    parser.add_argument("--tokenized-dataset-pickle", type=str, help="Path to a tokenized dataset .pkl file")
    parser.add_argument(
        "--file-path",
        type=str,
        default="/media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the text file to use for training.",
    )
    parser.add_argument(
        "--special-tokens", type=str, nargs="+", default=["<|endoftext|>"], help="List of special tokens."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the encoded text document as a Numpy array of token IDs. Must end with .npy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    assert args.save_path.endswith(".npy")
    # Ensure the parent directory of save_path exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("BPE Tokenization Script")
    print(f"Pre-trained BPE Tokenizer: {args.tokenized_dataset_pickle}")
    print(f"Text to tokenize: {args.file_path}")
    print(f"Token IDs will be saved to: {args.save_path}")

    tokenizer = Tokenizer.from_pickle(args.tokenized_dataset_pickle, special_tokens=args.special_tokens)

    token_ids = []
    start_time = time.time()  # Start the timer

    with open(args.file_path, "r", encoding="utf-8") as f:
        for i, token_id in enumerate(tokenizer.encode_iterable(f), start=1):
            token_ids.append(token_id)

            # Print progress every 100,000 tokens
            if i % 100_000 == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Processed {i} tokens so far. Time elapsed: {elapsed_time:.2f} seconds. Token list length: {len(token_ids)}"
                )
    # Final log
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Encoding completed. Total tokens: {len(token_ids)}. Total time: {int(hours)}h {int(minutes)}m {seconds:.2f}s."
    )

    # Convert the token IDs to a Numpy array and save it
    token_array = np.array(token_ids, dtype=np.uint16)
    print(f"Saving token IDs to {args.save_path}")
    np.save(args.save_path, token_array)
