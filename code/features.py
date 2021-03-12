import unicodedata
from collections import Counter
from functools import reduce

import numpy as np


def normalize(s):
    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("ASCII")
    s = s.lower()
    return s


def create_n_grams(words, n):
    text = "_".join(words)
    n_grams = [text[i:i + n] for i in range(0, len(text) - n)]
    return n_grams


def mfw(X, n, z_score=False):
    counters = [Counter(xi) for xi in X]
    total = reduce(lambda x, y: x + y, counters)
    mfw = dict(total.most_common(n))
    features = [[c[k] / v for k, v in mfw.items()] for c in counters]
    features = np.array(features)
    if z_score:
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        features = (features - means) / stds
    return features
