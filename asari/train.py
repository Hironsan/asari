"""
Train a baseline model.
"""
import argparse
import json
import pathlib

import numpy as np
from skl2onnx import to_onnx
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from asari.api import Sonar
from asari.preprocess import tokenize


def load_jsonl(filename):
    texts, labels = [], []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
    return texts, labels


def main(args):
    print("Loading dataset...")
    X, y = load_jsonl(args.dataset)
    X = [tokenize(x) for x in X]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    print("Fitting...")
    pipe = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", CalibratedClassifierCV(LinearSVC())),
        ]
    )
    pipe.fit(x_train, y_train)

    print("Saving...")
    seps = {
        TfidfVectorizer: {
            "separators": [
                " ",
            ],
        }
    }
    onx = to_onnx(pipe, np.array(x_train)[1:], options=seps)
    with open(args.pipeline, "wb") as f:
        f.write(onx.SerializeToString())

    print("Predicting...")
    y_pred = pipe.predict(x_test)
    print(classification_report(y_test, y_pred, digits=4))
    print(pipe.predict_proba([tokenize("広告多すぎる♡")]))

    sonar = Sonar()
    y_pred = [sonar.ping(x)["top_class"] for x in x_test]
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
    SAVE_DIR = pathlib.Path(__file__).parent / "data"
    parser = argparse.ArgumentParser(description="Training a classifier")
    parser.add_argument("--dataset", default=DATA_DIR / "dataset.jsonl", help="dataset")
    parser.add_argument("--pipeline", default=SAVE_DIR / "pipeline.onnx", help="pipeline file")
    parser.add_argument("--test_size", type=float, default=0.1, help="test data size")
    args = parser.parse_args()
    main(args)
