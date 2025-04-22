# This file was adapted from https://github.com/stanford-cs336/assignment1-basics
# and is licensed under the MIT License.
import os
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

# regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
# from github.com/openai/tiktoken/pull/234/files:
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_text_on_special_tokens(text: str, special_tokens=tuple[str]) -> list[str]:
    delimiter = re.escape("|".join(special_tokens))
    return re.split(delimiter, text)


def pretokenize_chunk(text: str, special_tokens: tuple[str]) -> dict[str, int]:
    """
    Pre-tokenize a chunk of text and count the occurrences of each pre-token.
    """
    documents = split_text_on_special_tokens(text, special_tokens)
    pre_token_counts = Counter()
    for document in documents:
        for token in re.finditer(PAT, document):
            # Extract the token from the match object
            token = token.group(0)
            # Otherwise, add it to the pre-token counts
            pre_token_counts[token] += 1
    return pre_token_counts


def load_and_pretokenize_chunk(args):
    """
    Loads the chunk of the file and applies pretokenization.
    This function is run in a separate process.
    """
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return pretokenize_chunk(chunk, special_tokens)


def load_and_pretokenize_file(file_path: str, special_tokens: tuple[str], num_processes: int = 32) -> dict[str, int]:
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    chunk_args = [(text_file_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(processes=num_processes) as pool:
        results = pool.map(load_and_pretokenize_chunk, chunk_args)

    total_pre_token_counts = Counter()
    for result in results:
        total_pre_token_counts.update(result)
    return total_pre_token_counts


if __name__ == "__main__":
    text_file_path = "/media/bryan/ssd01/data/cs336/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ("<|endoftext|>",)
    num_processes = 8
    total_pre_token_counts = load_and_pretokenize_file(text_file_path, special_tokens, num_processes)
    for token, count in total_pre_token_counts.most_common(10):
        print(f"Token: {token}, Count: {count}")
