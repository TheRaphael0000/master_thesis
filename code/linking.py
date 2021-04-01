"""Linking module"""

import unicodedata
from collections import Counter
from functools import reduce

import numpy as np
from scipy.spatial.distance import pdist, squareform
from corpus import brunet, oxquarry, st_jean, pan16
import distances
from misc import rank_list_from_distances_matrix
from evaluate import evaluate_linking


def normalize(s):
    # Removing accents
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("ASCII")
    # To lowercase
    s = s.lower()
    return s


def create_n_grams(words, ns):
    if type(ns) is int:
        ns = [ns]
    # if it's a list of numbers creating a long text of joined single character
    if type(words[0]) == int:
        text = "".join([chr(i) for i in words])
    # if it's a list of string create a text by joining them with a _
    if type(words[0]) == str:
        text = "_".join(words)
    n_grams = []
    # Creating n-grams from the long text
    for i in range(len(text)):
        for n in ns:
            n_gram = text[i:i + n]
            if len(n_gram) == n:
                n_grams.append(n_gram)
    return n_grams


def most_frequent_word(X, n, z_score=False, lidstone_lambda=0.1):
    # Counting the terms in each documents
    counters = [Counter(xi) for xi in X]
    # Remove hapax legomenon for each texts
    counters = [Counter({w: n for w, n in dict(c).items() if n > 1})
                for c in counters]
    # Computing the corpus word frequency
    total = reduce(lambda x, y: x + y, counters)
    # Selecting n mfw
    if n > 0:
        mfw = dict(total.most_common(n))
    else:
        mfw = dict(total)
    # tf to rtf or MLE probablity + lidstone smoothing
    features = [[(c[k] + lidstone_lambda) / (v + lidstone_lambda * len(total))
                 for k, v in mfw.items()] for c in counters]
    # Transforming the rtf to a 2D numpy array
    features = np.array(features)
    # Zscoring if needed
    if z_score:
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        features = (features - means) / stds
    return features, mfw


def compute_links(X, n_grams, n_mfw, z_score, lidstone_lambda, distance_func):
    # Tokens normalization
    if type(X[0]) == str:
        X = [[normalize(t) for t in xi] for xi in X]
    # Convert text to n_grams
    if type(n_grams) == list or type(n_grams) == tuple or n_grams > 0:
        X = [create_n_grams(xi, n_grams) for xi in X]
    # Create features
    features, mfw = most_frequent_word(X, n_mfw, z_score, lidstone_lambda)
    # Compute link distances into a 2D matrix
    distances_matrix = squareform(pdist(features, metric=distance_func))
    # Computing the rank list of for this distance matrix
    rank_list = rank_list_from_distances_matrix(distances_matrix)
    return rank_list


if __name__ == '__main__':
    # Simple test script for the module
    _, _, X, Y = brunet.parse()

    print("AP RPrec P@10 HPrec")
    rank_list = compute_links(X, 0, 500, False, 0.1, distances.manhattan)
    mesures = evaluate_linking(rank_list, Y)
    print(*mesures)
