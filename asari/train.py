"""
Train a baseline model.
"""
import argparse
import json
import pathlib

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from asari.preprocess import tokenize


def load_jsonl(filename):
    texts, labels = [], []
    label_map = {1: "negative", 5: "positive"}
    with open(filename) as f:
        for line in f:
            j = json.loads(line)
            label = int(j["rate"])
            if label in label_map:
                texts.append(j["text"])
                labels.append(label_map[label])

    return texts, labels


def main(args):
    print("Loading dataset...")
    X, y = load_jsonl(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    print("Fitting...")
    pipe = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
            ("classifier", CalibratedClassifierCV(LinearSVC())),
        ]
    )
    pipe.fit(x_train, y_train)

    print("Saving...")
    joblib.dump(pipe, args.pipeline)

    if args.test_size > 0.0:
        print("Predicting...")
        y_pred = pipe.predict(x_test)
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
    SAVE_DIR = pathlib.Path(__file__).parent / "data"
    parser = argparse.ArgumentParser(description="Training a classifier")
    parser.add_argument("--dataset", default=DATA_DIR / "reviews.jsonl", help="dataset")
    parser.add_argument("--pipeline", default=SAVE_DIR / "pipeline.joblib", help="pipeline file")
    parser.add_argument("--test_size", type=float, default=0.2, help="test data size")
    args = parser.parse_args()
    main(args)
