import os
from dataclasses import dataclass
from collections import Counter

from llm.pretokenization import load_and_pretokenize_file

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
    pair_counts = get_pair_counts(pre_token_indices_counts)
    while len(vocab) < vocab_size:
        # break ties by choosing the lexicographically greater (e.g. alphabetically pair)
        # which can be determined by the UTF-8 code point, aka the index
        # x[1] is the freq count of the index pair.
        pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        index1, index2 = pair
        # Merge that pair.
        new_index = len(vocab)
        merges_index[pair] = new_index
        merges.append((vocab[index1], vocab[index2]))
        # e.g. 'T' + 'h' -> 'Th'
        vocab[new_index] = vocab[index1] + vocab[index2]
        # update pre_token_indices_counts and pair_counts
        _pre_token_indices_counts = {}
        for indices, word_count in pre_token_indices_counts.items():
            indices = merge(indices, word_count, pair, new_index, pair_counts)
            _pre_token_indices_counts[tuple(indices)] = word_count
        pre_token_indices_counts = _pre_token_indices_counts
        pair_counts.pop(pair)

    return vocab, merges


def get_pair_counts(pre_token_indices_counts):
    pair_counts = Counter()
    for indices, counts in pre_token_indices_counts.items():
        for index_pair in zip(indices, indices[1:]):  # For each adjacent pair
            pair_counts[index_pair] += counts
    return pair_counts


def merge(
    indices: list[int], word_count: int, pair: tuple[int, int], new_index: int, pair_counts: dict[tuple[int, int], int]
) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`.

    When performing the merge, if there is a matching pair of indices,
    then the adjacent indices pairs are no longer valid. Need to decrement those pairs
    and add in 2 new pairs.
    indices = [w, i, d, e, s, t]
    pair = (d,e)
    merged_indices = [w, i, de, s, t]
    counts[(i,d)] -= 1
    counts[(e,s)] -= 1
    counts[(i,de)] += 1
    counts[(de,s)] += 1
    """
    new_indices = []
    i = 0
    just_merged = False
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)

            if just_merged:
                pair_counts[(new_index, new_index)] += word_count
                # the previous matching merge added this new pair (new_index, indices[i+2]),
                # that we need to correct now that we have another matching pair
                pair_counts[(new_index, indices[i])] -= word_count
            elif i > 0:
                pair_counts[(indices[i - 1], indices[i])] -= word_count
                pair_counts[(indices[i - 1], new_index)] += word_count

            if i + 2 < len(indices):
                pair_counts[(indices[i + 1], indices[i + 2])] -= word_count
                pair_counts[(new_index, indices[i + 2])] += word_count

            just_merged = True
            i += 2
        else:
            new_indices.append(indices[i])
            just_merged = False
            i += 1
    return new_indices


# @dataclass(frozen=True)
# class BPETokenizerParams:
#     """All you need to specify a BPETokenizer."""

#     vocab: dict[int, bytes]  # index -> bytes
#     merges: dict[tuple[int, int], int]  # index1,index2 -> new_index

# class ByteTokenizer(Tokenizer):
#     """Represent a string as a sequence of bytes."""

#     def encode(self, string: str) -> list[int]:
#         string_bytes = string.encode("utf-8")  # @inspect string_bytes
#         indices = list(map(int, string_bytes))  # @inspect indices
#         return indices

#     def decode(self, indices: list[int]) -> str:
#         string_bytes = bytes(indices)  # @inspect string_bytes
#         string = string_bytes.decode("utf-8")  # @inspect string
#         return string

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize a text file.")
    parser.add_argument(
        "--file_path",
        type=str,
        default="/media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the text file to pre-tokenize.",
    )
    parser.add_argument("--vocab-size", type=int, default=500, help="Size of the vocabulary to train.")
    parser.add_argument(
        "--special_tokens", type=str, nargs="+", default=["<|endoftext|>"], help="List of special tokens."
    )
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processes to use.")

    args = parser.parse_args()
    vocab, merges = run_train_bpe(
        input_path=args.file_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )

    for merge_item in merges[:10]:
        print(f"Merge: {repr(merge_item)}")
