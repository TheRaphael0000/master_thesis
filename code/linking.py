from pprint import pprint
import itertools
import unicodedata
from collections import Counter
from functools import reduce

import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from corpus import brunet, oxquarry, st_jean
from misc import zipflaw
import distances
from evaluate import hprec, precision_at_k, rprec, ap, rank_list_from_distances_matrix


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


def compute_links(X, n_grams, n_mfw, z_score, distance_func):
    # Normalization words
    X = [[normalize(w) for w in xi] for xi in X]
    # Convert text to n_grams
    if n_grams > 0:
        X = [create_n_grams(xi, n_grams) for xi in X]
    # Create features
    features = mfw(X, n_mfw, z_score)
    # Compute link distances
    distances_matrix = squareform(pdist(features, metric=distance_func))
    return distances_matrix


def experiment(X, Y, n_grams, n_mfw, z_score, distance_func):
    distances_matrix = compute_links(X, n_grams, n_mfw, z_score, distance_func)
    rank_list = rank_list_from_distances_matrix(distances_matrix)
    mesures = ap(rank_list, Y), rprec(rank_list, Y), hprec(rank_list, Y),
    return distances_matrix, rank_list, mesures


if __name__ == '__main__':
    # Loading dataset
    # id, x, y = oxquarry.parse()
    id, x_lemma, x_token, y = brunet.parse()

    # Select data
    X = x_lemma
    Y = y

    print(f"AP RPrec HPrec")
    print(experiment(X, Y, 5, 500, False, distances.tanimoto)[-1])
