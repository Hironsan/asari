from typing import List

from janome.tokenizer import Tokenizer

t = Tokenizer()


def tokenize(text: str) -> List[str]:
    return list(t.tokenize(text, wakati=True))
