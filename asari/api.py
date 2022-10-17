import pathlib

import joblib
import numpy as np


class Sonar:
    def __init__(self):
        pipeline_file = pathlib.Path(__file__).parent / "data" / "pipeline.joblib"
        self.pipe = joblib.load(pipeline_file)

    def ping(self, text: str):
        proba = self.pipe.predict_proba([text])[0]
        mapping = {0: "negative", 1: "positive"}
        res = {
            "text": text,
            "top_class": mapping[int(np.argmax(proba))],
            "classes": [{"class_name": mapping[k], "confidence": proba[k]} for k in sorted(mapping)],
        }

        return res
