# asari

Asari is a Japanese sentiment analyzer implemented in Python.

## Usage

Behold, the power of asari:

```python
from asari.api import Sonar
sonar = Sonar()
sonar.ping(text="広告多すぎる♡")
{
  "text" : "広告多すぎる♡",
  "top_class" : "negative",
  "classes" : [ {
    "class_name" : "positive",
    "confidence" : 0.09130180181262026
  }, {
    "class_name" : "negative",
    "confidence" : 0.9086981981873797
  } ]
}
```

Asari allows you to classify text into positive/negative class, without the need for training. You have only to fed text into asari.

## Installation

To install asari, simply use `pip` with Python 3.8+:

```bash
pip install asari
```
