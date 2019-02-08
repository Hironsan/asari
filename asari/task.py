"""
Baseline model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from asari.utils import load_jsonl
from asari.preprocess import tokenize


def main(args):
    print('Loading dataset...')
    X, y = load_jsonl(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    print('Vectorizing...')
    vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(x_train)

    print('Fitting...')
    clf = CalibratedClassifierCV(LinearSVC())
    clf.fit(x_train, y_train)

    print('Saving...')
    joblib.dump(clf, args.model_file, protocol=2)
    joblib.dump(vectorizer, args.preprocessor, protocol=2)

    if args.test_size > 0.0:
        print('Predicting...')
        x_test = vectorizer.transform(x_test)
        y_pred = clf.predict(x_test)

        print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'data')
    parser = argparse.ArgumentParser(description='Training a classifier')
    parser.add_argument('--dataset', default=os.path.join(DATA_DIR, 'reviews.jsonl'), help='dataset')
    parser.add_argument('--model_file', default=os.path.join(SAVE_DIR, 'model.pkl'), help='model file')
    parser.add_argument('--preprocessor', default=os.path.join(SAVE_DIR, 'preprocess.pkl'), help='preprocessor')
    parser.add_argument('--test_size', type=float, default=0.2, help='test data size')
    args = parser.parse_args()
    main(args)
