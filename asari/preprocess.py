from janome.tokenizer import Tokenizer

t = Tokenizer(wakati=True)


def tokenize(text: str) -> str:
    return " ".join(t.tokenize(text))
