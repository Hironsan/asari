from janome.tokenizer import Tokenizer
t = Tokenizer()


def tokenize(text):
    return t.tokenize(text, wakati=True)
