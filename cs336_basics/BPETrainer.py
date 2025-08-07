import os
import regex as re
from collections import defaultdict
import pickle
import time
import cProfile

class BPETrainer(object):
    def __init__(self, *args, **kwargs) -> None:
        self.input_path: str | os.PathLike = ''
        self.vocab_size: int = 2000
        self.special_tokens: list[str] = []

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: list[tuple[bytes, bytes]] = []
        self.next_id = len(self.vocab)

        self.vocab_path: str = ''
        self.merges_path: str = ''

        self.word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int)
        self.pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)

    def train_bpe(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # add special tokens
        for token in self.special_tokens:
            if len(self.vocab) >= self.vocab_size:
                break
        self.vocab[self.next_id] = token.encode('utf-8')
        self.next_id += 1

        ################################################################################
        # Step 2: Read text from training file
        ################################################################################
        try:
            with open(self.input_path, 'r+', encoding='utf-8', errors='ignore') as f:
                corpus = f.read()
        except FileNotFoundError:
            corpus = ''

        ################################################################################
        # Step 3: Pretokenize
        ################################################################################
        # 1. remove special tokens using regex, i.e., split text using special tokens
        chunks = re.split('|'.join(map(re.escape, special_tokens)), corpus)

        # 2. initialize word freqs
        # for each chunk, split using gpt2 regex-based pre-tokenizer
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            words: list[str] = re.findall(PAT, chunk)
            for word in words:
                word_bytes = [bytes([x]) for x in word.encode('utf-8')]
                self.word_freqs[tuple(word_bytes)] += 1

        # 3. calculate freq of pair for all word
        for word, freq in self.word_freqs.items():
            for i in range(len(word) - 1):
                self.pair_freqs[word[i], word[i+1]] += freq


        ################################################################################
        # Step 4: Train BPE Tokenizer
        ################################################################################
        # train loop
        while len(self.vocab) < self.vocab_size:
            if not self.pair_freqs:
                break

            # most freq pair
            max_freq = max(self.pair_freqs.values())
            candidates = [pair for pair, freq in self.pair_freqs.items() if freq == max_freq]
            best_pair = max(candidates)

            # add to vocab and merges
            self.merges.append(best_pair)
            self.vocab[self.next_id] = best_pair[0] + best_pair[1]
            self.next_id += 1

            # find affected words
            affected_word_freqs: list[tuple[tuple[bytes, ...], int]] = []
            for word, freq in self.word_freqs.items():
                has_pair = any(word[i:i+2] == best_pair for i in range(len(word) - 1))

                # only token including best_pair will change
                if has_pair:
                    affected_word_freqs.append((word, freq))

            # update 
            for word, freq in affected_word_freqs:
                # 1. remove old word with best pair from pair_freqs 
                for i in range(len(word) - 1):
                    self.pair_freqs[word[i], word[i+1]] -= freq
                    if self.pair_freqs[word[i], word[i+1]] <= 0:
                        del self.pair_freqs[word[i], word[i+1]]

                # 2. merge word pieces: word -> new_word
                new_word: tuple[bytes, ...] = self.merge_pairs(word, best_pair)

                # 3. re-compute pair_freq
                for i in range(len(new_word) - 1):
                    self.pair_freqs[new_word[i], new_word[i+1]] += freq

                # update word_freqs
                del self.word_freqs[word]
                self.word_freqs[new_word] += freq

        return self.vocab, self.merges
    
    def save(self, save_path: str):
        self.vocab_path = save_path + '-vocab.pkl'
        self.merges_path = save_path + '-merges.pkl'
        with open(self.vocab_path, "wb") as f:
            pickle.dump(self.vocab, f)
        
        with open(self.merges_path, "wb") as f:
            pickle.dump(self.merges, f)

    
    
    def merge_pairs(
        self, word: tuple[bytes, ...], 
        best_pair: tuple[bytes, bytes]
        ) -> tuple[bytes, ...]:
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


def main():
    input_path = './data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    dataset_name = os.path.split(input_path)[-1].split('.')[0]
    save_path = os.path.join('./results', dataset_name)
    

    bpe_trainer = BPETrainer()

    start_time = time.time()
    bpe_trainer.train_bpe(input_path, vocab_size, ['<|endoftext|>'])
    end_time = time.time()
    print(end_time - start_time, 's')

    bpe_trainer.save(save_path)

if __name__ == '__main__':
    # cProfile.run('main()')
    main()
