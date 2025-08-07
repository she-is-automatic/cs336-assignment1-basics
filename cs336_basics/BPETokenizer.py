import pickle
from typing import Iterable, Iterator

class BPETokenizer():
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None
                 ) -> None:
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, 
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None
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
        raise NotImplementedError
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.
        """
        raise NotImplementedError
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        raise NotImplementedError
    

def test_from_files(tokenizer: BPETokenizer):
    print(len(tokenizer.vocab))
    print(max(tokenizer.vocab.values(), key=len))
    for id, token in tokenizer.vocab.items():
        if id >= 2000 and id < 2010:
            print(id, token)

    print([tokenizer.merges[i] for i in range(10)])

if __name__ == '__main__':
    vocab_filepath = 'results/TinyStoriesV2-GPT4-train-vocab.pkl'
    merges_filepath = 'results/TinyStoriesV2-GPT4-train-merges.pkl'
    special_tokens = ['<|endoftext|>']

    tokenizer = BPETokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    test_from_files(tokenizer)

