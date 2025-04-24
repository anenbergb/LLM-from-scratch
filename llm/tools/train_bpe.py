import argparse
import sys
import pickle
import os
import time  # Import the time module
from llm.tokenization import run_train_bpe


def get_args():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer and save vocabulary and merges to disk."
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="/media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the text file to use for training.",
    )
    parser.add_argument("--vocab-size", type=int, default=500, help="Size of the vocabulary to train.")
    parser.add_argument(
        "--special-tokens", type=str, nargs="+", default=["<|endoftext|>"], help="List of special tokens."
    )
    parser.add_argument("--num-processes", type=int, default=8, help="Number of processes to use.")
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the trained vocabulary and merges as a pickle file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Measure the time taken by run_train_bpe
    start_time = time.time()  # Start the timer
    vocab, merges = run_train_bpe(
        input_path=args.file_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )
    end_time = time.time()  # End the timer

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"run_train_bpe completed in {elapsed_time:.2f} seconds.")

    # Ensure the parent directory of save_path exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Save the vocab and merges to the specified save path
    with open(args.save_path, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    print(f"Vocabulary and merges saved to {args.save_path}")
