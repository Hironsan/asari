import json


def load_jsonl(filename):
    texts, labels = [], []
    # label_map = {1: 'negative', 2: 'negative', 4: 'positive', 5: 'positive'}
    label_map = {1: 'negative', 5: 'positive'}
    with open(filename) as f:
        for line in f:
            j = json.loads(line)
            label = int(j['rate'])
            if label in label_map:
                texts.append(j['text'])
                labels.append(label_map[label])

    return texts, labels
