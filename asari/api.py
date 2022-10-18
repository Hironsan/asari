import pathlib

import onnxruntime as rt

from asari.preprocess import tokenize


class Sonar:
    def __init__(self):
        pipeline_file = pathlib.Path(__file__).parent / "data" / "pipeline.onnx"
        self.sess = rt.InferenceSession(str(pipeline_file))
        self.input_name = self.sess.get_inputs()[0].name
        self.prob_name = self.sess.get_outputs()[1].name

    def ping(self, text: str):
        tokenized = tokenize(text)
        proba = self.sess.run([self.prob_name], {self.input_name: [tokenized]})[0][0]
        res = {
            "text": text,
            "top_class": max(proba, key=lambda k: proba[k]),
            "classes": [
                {"class_name": class_name, "confidence": confidence} for class_name, confidence in proba.items()
            ],
        }
        return res
