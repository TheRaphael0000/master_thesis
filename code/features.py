import unicodedata
from collections import Counter
from functools import reduce


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


def mfw(X, n, skip_n=0):
    counters = [Counter(xi) for xi in X]
    total = reduce(lambda x, y: x + y, counters)
    mfw = dict(total.most_common(n))
    normalized_mfw_counters = [
        {k: c[k] / v for k, v in mfw.items()} for c in counters]
    features = [c.values() for c in normalized_mfw_counters]
    return features
