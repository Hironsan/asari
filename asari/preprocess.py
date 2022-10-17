from janome.tokenizer import Tokenizer

t = Tokenizer()


def tokenize(text):
    return list(t.tokenize(text, wakati=True))
