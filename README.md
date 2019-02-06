# asari
Asari is a Japanese sentiment analyzer implemented in Python.

![Image](./docs/asari.jpg)
Photo by [Andrew Buchanan](https://unsplash.com/@photoart2018) on [Unsplash](https://unsplash.com/)

Behold, the power of asari:

```python
>>> from asari import Sonar
>>> sonar = Sonar()
>>> sonar.ping(text="いい感じです！")
{
  "text" : "いい感じです！",
  "top_class" : "positive",
  "classes" : [ {
    "class_name" : "positive",
    "confidence" : 0.9965442572893405
  }, {
    "class_name" : "negative",
    "confidence" : 0.003455742710659515
  } ]
}
```

Asari allows you to classify text into positive/negative class, without the need for training. You have only to fed text into asari.

## Installation
To install asari, simply use `pip`:

```bash
$ pip install asari
```
