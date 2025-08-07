import os
import regex as re
from collections import defaultdict
import pickle


def remove_special_tokens() -> str:
    return ""

def pretokenize(doc: str) -> dict[tuple, int]:
    love_bytes = b'love'
    return {(love_bytes, love_bytes): 5}

def merge_pairs(word: tuple[bytes, ...], best_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    if len(word) < 2:
        return word
    
    merged_pair = best_pair[0] + best_pair[1]

    new_word: list[bytes] = []
    i = 0
    while i < len(word) - 1:
        if word[i:i+2] == best_pair:
            new_word.append(merged_pair)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    if word[-2:] != best_pair:
        new_word.append(word[-1])
    return tuple(new_word)

def train_bpe(
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
            kept as a single token. If these special tokens occur in the `input_path`,
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
    ################################################################################
    # Step 1: Initialize vocab with single bytes and special tokens
    ################################################################################
    # init vocab with single byte 0 - 255
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id: int = len(vocab)
    existing_tokens: set[bytes] = set(vocab.values())
    
    # add special tokens
    for word in special_tokens:
        if len(vocab) >= vocab_size or word in existing_tokens:
            break
        vocab[next_id] = word.encode('utf-8')
        existing_tokens.add(word.encode('utf-8'))
        next_id += 1

    ################################################################################
    # Step 2: Read text from training file
    ################################################################################
    try:
        with open(input_path, 'r+', encoding='utf-8', errors='ignore') as f:
            corpus = f.read()
    except FileNotFoundError:
        corpus = ''

    ################################################################################
    # Step 3: Pretokenize
    ################################################################################
    # 1. remove special tokens using regex, i.e., split text using special tokens
    chunks = re.split('|'.join(map(re.escape, special_tokens)), corpus)

    # 2. initialize word freqs
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    # for each chunk, split using gpt2 regex-based pre-tokenizer
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        words: list[str] = re.findall(PAT, chunk)
        # words = chunk.split(' ')
        for word in words:
            word_bytes = [bytes([x]) for x in word.encode('utf-8')]
            word_freqs[tuple(word_bytes)] += 1

    # 3. calculate freq of pair for all word
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[word[i], word[i+1]] += freq


    ################################################################################
    # Step 4: Train BPE Tokenizer
    ################################################################################
    # init merges
    merges: list[tuple[bytes, bytes]] = []

    # train loop
    while len(vocab) < vocab_size:
        if not pair_freqs:
            break

        # most freq pair
        max_freq = max(pair_freqs.values())
        candidates = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        best_pair = max(candidates)

        # add to vocab and merges
        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1

        # update pair_freqs
        affected_word_freqs: list[tuple[tuple[bytes, ...], int]] = []
        for word, freq in word_freqs.items():
            has_pair = any(word[i:i+2] == best_pair for i in range(len(word) - 1))

            # only token including best_pair will change
            if has_pair:
                affected_word_freqs.append((word, freq))

        for word, freq in affected_word_freqs:
            # 1. remove old word with best pair from pair_freqs 
            for i in range(len(word) - 1):
                pair_freqs[word[i], word[i+1]] -= freq
                if pair_freqs[word[i], word[i+1]] <= 0:
                    del pair_freqs[word[i], word[i+1]]

            # 2. merge word pieces: word -> new_word
            new_word: tuple[bytes, ...] = merge_pairs(word, best_pair)

            # 3. re-compute pair_freq
            for i in range(len(new_word) - 1):
                pair_freqs[new_word[i], new_word[i+1]] += freq

            # update word_freqs
            del word_freqs[word]
            word_freqs[new_word] += freq

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    return vocab, merges


if __name__ == '__main__':
    train_bpe('./data/test.txt', 1000, ['<|endoftext|>'])