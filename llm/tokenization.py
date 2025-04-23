import os
from dataclasses import dataclass
from collections import Counter, defaultdict

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


def load_fixtures():
    import sys, json
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from tests.common import gpt2_bytes_to_unicode
    FIXTURES_PATH = "/home/bryan/src/LLM-from-scratch/tests/fixtures"
    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = os.path.join(FIXTURES_PATH, "train-bpe-reference-vocab.json")
    reference_merges_path = os.path.join(FIXTURES_PATH, "train-bpe-reference-merges.txt")

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    return reference_vocab, reference_merges


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
    ref_vocab, ref_merges = load_fixtures()

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
        index1, index2 = pair_count[0]
        count = pair_count[1]
        bytes1 = vocab[index1]
        bytes2 = vocab[index2]
        return (count, bytes1, bytes2)
        

    while len(vocab) < vocab_size:
        # break ties by choosing the lexicographically greater (e.g. alphabetically pair)
        # which can be determined by the UTF-8 code point
        # e.g. to compare b" c" and b"t" you compare the first character 
        # b" " (UTF-8 32) and b"t" (UTF-8 116) and select b"t".
        # x[1] is the freq count of the index pair.
        pair = max(pair_counts.items(), key=cmp_function)[0]
        index1, index2 = pair
        # DEBUGGING
        my_merge = (vocab[index1], vocab[index2])
        merge_index = len(merges)
        ref_merge = ref_merges[merge_index]
        if my_merge != ref_merge:
            print(f"{my_merge} | {ref_merge}")
            reverse_vocab = {v:k for k,v in vocab.items()}

            ref_byte_index1, ref_byte_index2 = ref_merge
            ref_index1 = reverse_vocab[ref_byte_index1]
            ref_index2 = reverse_vocab[ref_byte_index2]
            ref_pair = (ref_index1, ref_index2)
            ref_pair_count = pair_counts[ref_pair]
            
            print(f"my_merge pair_count: {pair_counts[pair]}")
            print(f"ref_merge pair_count: {ref_pair_count}")
            sorted_pair_counts = sorted(pair_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
            print("Sorted Pairs:")
            for sorted_pair, counts in sorted_pair_counts[:5]:
                sorted_pair_bytes = (vocab[sorted_pair[0]], vocab[sorted_pair[1]])
                print(f"\t{sorted_pair} = {sorted_pair_bytes}\t: {counts}")

            import ipdb

            ipdb.set_trace()

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
        "--file-path",
        type=str,
        default="/media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-valid.txt",
        help="Path to the text file to pre-tokenize.",
    )
    parser.add_argument("--vocab-size", type=int, default=500, help="Size of the vocabulary to train.")
    parser.add_argument(
        "--special-tokens", type=str, nargs="+", default=["<|endoftext|>"], help="List of special tokens."
    )
    parser.add_argument("--num-processes", type=int, default=8, help="Number of processes to use.")

    args = parser.parse_args()
    vocab, merges = run_train_bpe(
        input_path=args.file_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        num_processes=args.num_processes,
    )

    for merge_item in merges[:10]:
        print(f"Merge: {repr(merge_item)}")
