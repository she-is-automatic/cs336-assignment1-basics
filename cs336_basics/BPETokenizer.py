import pickle
from typing import Iterable, Iterator
import regex as re

class BPETokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens
                 ) -> None:
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # for O(1) token to id conversion
        self.token2id: dict[bytes, int] = {token: id for id, token in self.vocab.items()}

        # for O(1) find merge and compare merge order priority
        self.merges_map: dict[tuple[bytes, bytes], int] = {merge: i for i, merge in enumerate(merges)}

        # for split pretokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(cls, 
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens
                   ):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens."""
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int] :
        """
        Encode an input text into a sequence of token IDs
        """
        if not text:
            return []

        encoded_ids: list[int] = []

        pre_tokens = self._pretokenize(text)

        for pre_token in pre_tokens:
            if isinstance(pre_token, bytes):
                encoded_ids.append(self.token2id[pre_token])
            else:
                pre_token = self._merge_pretoken(pre_token)
                encoded_ids += [self.token2id[token] for token in pre_token]

        return encoded_ids

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        text_bytes: bytes = b''.join(self.vocab[id] for id in ids)
        return text_bytes.decode('utf-8', errors='replace')
    
    def _split_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = '|'.join(map(re.escape, sorted_tokens))
        chunks = re.split(f'({pattern})', text)
        return chunks


    def _pretokenize(self, text: str) -> list[bytes|tuple[bytes, ...]]:
        """
        return pre_tokens
        type(pre_tokens[i]) = tuple[bytes, ...] if it's NOT a special token;
        type(pre_tokens[i]) = bytes if it's a special token;
        eg.: 
            text: 'a <|endoftext|>bb<|endoftext|>'
            return: [(b'a',), (b' ',), b'<|endoftext|>', (b'b', b'b'), b'<|endoftext|>']
        """
        pre_tokens: list[bytes|tuple[bytes, ...]] = []

        chunks = self._split_special_tokens(text)
        for chunk in chunks:
            if not chunk:
                continue
            elif self.special_tokens and chunk in self.special_tokens:
                pre_tokens.append(chunk.encode('utf-8'))
            else:
                words = re.findall(self.PAT, chunk)
                
                for word in words:
                    word_bytes = tuple([bytes([x]) for x in word.encode('utf-8')])
                    pre_tokens.append(word_bytes)
                # pre_tokens += re.findall(self.PAT, chunk)
        return pre_tokens
    
    def _merge_pretoken(self, pretoken) -> tuple[bytes, ...]:

        while(True):

            # all pairs that can be merged
            pairs: set[tuple[bytes, bytes]] = set()
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i+1])
                if pair in self.merges_map:
                    pairs.add(pair)
            
            # no pairs to merge
            if not pairs:
                break

            # find the first merge pair in merges
            first_merge = min(pairs, key=lambda pair: self.merges_map[pair])

            # apply merge
            pretoken = self._merge_pairs(pretoken, first_merge)

        return pretoken

    
    def _merge_pairs(
        self, word: tuple[bytes, ...], 
        best_pair: tuple[bytes, bytes]
        ) -> tuple[bytes, ...]:
        if len(word) < 2:
            return word
        
        merged_pair = best_pair[0] + best_pair[1]

        new_word: list[bytes] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i:i+2] == best_pair:
                new_word.append(merged_pair)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

def test_from_files(tokenizer: BPETokenizer):
    print(len(tokenizer.vocab))
    print(max(tokenizer.vocab.values(), key=len))
    for id, token in tokenizer.vocab.items():
        if id >= 2000 and id < 2010:
            print(id, token)

    print([tokenizer.merges[i] for i in range(10)])

def test_split(tokenizer: BPETokenizer):
    text1 = 'a<|endoftext|>bb<|endoftext|>ccc<|endoftext|><|endoftext|><|endoftext|>'
    text2 = 'a bb ccc ddd'
    print(tokenizer._split_special_tokens(text1))
    print(tokenizer._split_special_tokens(text2))

def test_pretokenize(tokenizer):
    text1 = 'a <|endoftext|>bb<|endoftext|>ccc<|endoftext|><|endoftext|><|endoftext|> hello world'
    text2 = 'a bb ccc ddd'
    print(tokenizer._pretokenize(text1))
    print(tokenizer._pretokenize(text2))

def test_merge_pretoken(tokenizer):
    text = 'a <|endoftext|>bb<|endoftext|>ccc<|endoftext|><|endoftext|><|endoftext|> hello world'
    pretokens = tokenizer._pretokenize(text)
    for pretoken in pretokens:
        print(pretoken, tokenizer._merge_pretoken(pretoken))

def test_pretoken2id(tokenizer):
    text = 'a <|endoftext|>bb<|endoftext|>ccc<|endoftext|><|endoftext|><|endoftext|> hello world'
    pretokens = tokenizer._pretokenize(text)
    for pretoken in pretokens:
        print(pretoken)
        pretoken_ids = [tokenizer.token2id[token] for token in pretoken]
        print(pretoken_ids)

def test_encode(tokenizer):
    text1 = 'a <|endoftext|>bb<|endoftext|>ccc<|endoftext|><|endoftext|><|endoftext|> hello world'
    text2 = 'a bb ccc ddd'
    text3 = ''
    print(tokenizer.encode(text1))
    print(tokenizer.encode(text2))
    print(tokenizer.encode(text3))

def test_decode(tokenizer):
    text1 = 'a <|endoftext|>bb<|endoftext|>ccc<|endoftext|><|endoftext|><|endoftext|> hello world'
    text2 = 'a bb ccc ddd'
    text3 = ''
    text4 = 'a'

    ids1 = tokenizer.encode(text1)
    ids2 = tokenizer.encode(text2)
    ids3 = tokenizer.encode(text3)
    ids4 = tokenizer.encode(text4)
    print(tokenizer.decode(ids1) == text1)
    print(tokenizer.decode(ids2) == text2)
    print(tokenizer.decode(ids3) == text3)
    print(tokenizer.decode(ids4) == text4)


if __name__ == '__main__':
    vocab_filepath = 'results/TinyStoriesV2-GPT4-train-vocab.pkl'
    merges_filepath = 'results/TinyStoriesV2-GPT4-train-merges.pkl'
    special_tokens = ['<|endoftext|>']

    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # test_from_files(tokenizer)
    # test_split(tokenizer)
    # test_pretokenize(tokenizer)
    # test_merge_pretoken(tokenizer)
    # test_pretoken2id(tokenizer)
    # test_encode(tokenizer)
    test_decode(tokenizer)

