import argparse
import os
from tqdm import tqdm
from llm.tokenization import Tokenizer
import random


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens


def get_args():
    parser = argparse.ArgumentParser(
        description="Given a previously-trained tokenizer, encode a text and measure it's compression ratio"
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
        "--num-documents", type=int, default=10, help="Number of documents to sample from the text file."
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Set the random seed for reproducibility
    random.seed(args.random_seed)

    tokenizer = Tokenizer.from_pickle(args.tokenized_dataset_pickle, special_tokens=args.special_tokens)

    # Read the file and split it into documents using the <|endoftext|> delimiter
    with open(args.file_path, "r", encoding="utf-8") as f:
        text = f.read()
    documents = text.split("<|endoftext|>")

    # Remove empty documents and strip whitespace
    documents = [doc.strip() for doc in documents if doc.strip()]

    # Sample args.num_documents documents (or all if fewer are available)
    num_to_sample = min(args.num_documents, len(documents))
    sampled_documents = random.sample(documents, num_to_sample) if num_to_sample > 0 else []

    num_bytes = 0
    num_tokens = 0
    for document in tqdm(sampled_documents, desc="Processing documents"):
        num_bytes += len(bytes(document, encoding="utf-8"))
        encoded_document = tokenizer.encode(document)
        num_tokens += len(encoded_document)

    ratio = num_bytes / num_tokens
    print(f"Compression Ratio: {ratio:.2f}")
