import os
import pickle
import regex as re
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Any
from collections.abc import Iterable, Iterator

from llm.pretokenization import load_and_pretokenize_file, PRETOKENIZATION_REGEX

"""
Example output of the pre-tokenization step

Token: '.', Count: 421616
Token: ',', Count: 235432
Token: ' the', Count: 211031
Token: ' and', Count: 196057
Token: ' a', Count: 152161
Token: '\n', Count: 151989
Token: ' to', Count: 150493
Token: ' was', Count: 108019
Token: ' They', Count: 52425
Token: ' it', Count: 51670

"""


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,1
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # ref_vocab, ref_merges = load_fixtures()

    # Load and pretokenize the file
    # dict[str, int] -> int
    pre_token_counts = load_and_pretokenize_file(
        input_path,
        special_tokens=special_tokens,
        num_processes=kwargs.get("num_processes", 32),
    )
    merges_index: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    merges: list[tuple[bytes, bytes]] = []  # bytes @ index1, bytes @ index2
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    # Add special tokens to the vocabulary
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # ' They' converted to (32, 84, 104, 101, 121)
    pre_token_indices_counts = {
        tuple(map(int, string.encode("utf-8"))): count for string, count in pre_token_counts.items()
    }
    pair_counts, pair_to_words = get_pair_counts(pre_token_indices_counts)

    def cmp_function(pair_count):
        """
        break ties by choosing the lexicographically greater (e.g. alphabetically pair)
        which can be determined by the UTF-8 code point
        e.g. to compare b" c" and b"t" you compare the first character
        b" " (UTF-8 32) and b"t" (UTF-8 116) and select b"t".
        """
        index1, index2 = pair_count[0]
        count = pair_count[1]
        bytes1 = vocab[index1]
        bytes2 = vocab[index2]
        return (count, bytes1, bytes2)

    while len(vocab) < vocab_size:
        pair = max(pair_counts.items(), key=cmp_function)[0]
        index1, index2 = pair

        # Merge that pair.
        new_index = len(vocab)
        merges_index[pair] = new_index
        merges.append((vocab[index1], vocab[index2]))
        # e.g. 'T' + 'h' -> 'Th'
        vocab[new_index] = vocab[index1] + vocab[index2]

        words = pair_to_words.pop(pair)
        for indices in words:
            _indices = merge(indices, pair, new_index)
            word_count = pre_token_indices_counts.pop(indices)
            pre_token_indices_counts[_indices] = word_count
            # update the counts for the adjacent pairs to the merged pair
            update_pair_counts_sub(indices, word_count, pair_counts, pair_to_words)
            update_pair_counts(_indices, word_count, pair_counts, pair_to_words)

        pair_counts.pop(pair)

    return vocab, merges


def get_pair_counts(pre_token_indices_counts):
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for indices, counts in pre_token_indices_counts.items():
        update_pair_counts(indices, counts, pair_counts, pair_to_words)
    return pair_counts, pair_to_words


def update_pair_counts(indices, word_count, pair_counts, pair_to_words):
    for index_pair in zip(indices, indices[1:]):  # For each adjacent pair
        pair_counts[index_pair] += word_count
        pair_to_words[index_pair].add(indices)


def update_pair_counts_sub(indices, word_count, pair_counts, pair_to_words):
    for index_pair in zip(indices, indices[1:]):  # For each adjacent pair
        pair_counts[index_pair] -= word_count
        pair_to_words[index_pair] -= {indices}


def merge(indices: list[int], pair: tuple[int, int], new_index: int):
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return tuple(new_indices)


class Tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    construct a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    """

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        max_token_id = max(vocab.keys())
        assert max_token_id == len(vocab) - 1

        special_bytes = [s.encode("utf-8") for s in special_tokens]
        special_tokens_found = set()
        for vocab_bytes in vocab.values():
            if vocab_bytes in special_bytes:
                special_tokens_found.add(vocab_bytes.decode("utf-8"))
        if len(special_tokens) > len(special_tokens_found):
            missing_special_tokens = set(special_tokens) - special_tokens_found
            for special_token in missing_special_tokens:
                print(f"special_token '{special_token}' not found in vocabulary. Adding it now.")
                vocab[len(vocab)] = special_token.encode("utf-8")

        self.bytes2id = {v: k for k, v in vocab.items()}
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Class  method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens. This method should accept the following additional parameters:
        """
        return NotImplementedError

    @classmethod
    def from_pickle(cls, filepath: str, special_tokens: list[str] | None = None):
        with open(filepath, "rb") as f:
            bpe_data = pickle.load(f)
        assert "vocab" in bpe_data
        assert "merges" in bpe_data
        return cls(bpe_data["vocab"], bpe_data["merges"], special_tokens)

    def _split_on_special_tokens(self, text: str):
        # Escape special characters in tokens and join into a regex pattern
        pattern = "(" + "|".join(map(re.escape, self.special_tokens)) + ")"
        # Split the text while keeping the delimiters
        chunks = re.split(pattern, text)
        # Optionally remove empty strings (e.g., from split at the beginning)
        chunks = [chunk for chunk in chunks if chunk]
        return chunks

    def _pretokenize_iter(self, text: str):
        for token in re.finditer(PRETOKENIZATION_REGEX, text):
            # Extract the token from the match object
            token = token.group(0)
            token_bytes = token.encode("utf-8")
            yield token_bytes

    def _apply_merges(self, pretoken: bytes) -> list[int]:
        """
        Iterate through the self.merges list[tuple[bytes, bytes]] in order
        and apply any of those merges to pretoken bytes.
        Return the merged bytes.
        """
        # Convert the pretoken bytes into a list UTF-8 ints
        token = list(pretoken)

        # Iterate through the merges and apply them
        for merge in self.merges:
            byte1, byte2 = merge
            id1 = self.bytes2id[byte1]
            id2 = self.bytes2id[byte2]
            merged_id = self.bytes2id[byte1 + byte2]
            i = 0
            while i < len(token) - 1:
                # Check if the current pair matches the merge
                if token[i] == id1 and token[i + 1] == id2:
                    # Replace the pair with the merged token
                    token[i : i + 2] = [merged_id]
                else:
                    i += 1

        return token

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        # pre-tokenize
        documents = self._split_on_special_tokens(text)
        indices = []
        for document in documents:
            if document in self.special_tokens:
                indices.append(self.bytes2id[document.encode("utf-8")])
            else:
                # pre-tokenize. each pretoken is bytes
                for pretoken in self._pretokenize_iter(document):
                    token_ids = self._apply_merges(pretoken)
                    indices.extend(token_ids)
        return indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle),
        return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files
        that we cannot directly load into memory.
        """
        return NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        string_bytes = bytes(ids)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize a text file.")
    parser.add_argument(
        "--file-path",
        type=str,
        default="/media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the text file to tokenize.",
    )
    parser.add_argument("--vocab-size", type=int, default=500, help="Size of the vocabulary to train.")
    parser.add_argument(
        "--special-tokens", type=str, nargs="+", default=["<|endoftext|>"], help="List of special tokens."
    )
    parser.add_argument("--num-processes", type=int, default=8, help="Number of processes to use.")
    parser.add_argument("--tokenized-dataset-pickle", type=str, help="Path to a tokenized dataset .pkl file")

    args = parser.parse_args()

    if args.tokenized_dataset_pickle is None:
        vocab, merges = run_train_bpe(
            input_path=args.file_path,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            num_processes=args.num_processes,
        )
        for merge_item in merges[:10]:
            print(f"Merge: {repr(merge_item)}")
        tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
    else:
        tokenizer = Tokenizer.from_pickle(args.tokenized_dataset_pickle, special_tokens=args.special_tokens)

    endoftext = "<|endoftext|>"
    with open(args.file_path, "r", encoding="utf-8", errors="ignore") as f:
        corpus = f.read()

    # find the first index of endoftext
    index = 5000
    document = corpus[:index]

    print("Encoding document...")
    encoded_document = tokenizer.encode(document)
    import ipdb

    ipdb.set_trace()
    print("Decoding document...")
    decoded_document = tokenizer.decode(document)

    print("Comparing before and after")
