"""Linking module"""

from collections import Counter
from functools import reduce
import itertools

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from corpus import oxquarry
from corpus import brunet
from corpus import st_jean
from corpus import pan16

import distances
import compressions

from misc import rank_list_from_distances_matrix
from misc import normalize_text
from evaluate import evaluate_linking


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


def most_frequent_word(X, n, z_score=False, lidstone_lambda=0.1, remove_hapax=True):
    # Counting the terms in each documents
    counters = [Counter(xi) for xi in X]
    # Remove hapax legomenon for each texts
    if remove_hapax:
        counters = [Counter({w: n for w, n in dict(c).items() if n > 1})
                    for c in counters]
    # Computing the corpus word frequency
    total = reduce(lambda x, y: x + y, counters)
    # n bounding
    if n > len(total):
        n = len(total)
    # Selecting n mfw
    mfw = dict(total.most_common(len(total)))
    # keep only the mfw in each counter
    features = [[c[k] for k in mfw.keys()] for c in counters]
    # Transforming the tf to a 2D numpy array
    features = np.array(features)
    # normalize tf to rtf or MLE probablity + lidstone smoothing
    num = features + lidstone_lambda
    den = np.sum(features, axis=1) + lidstone_lambda * features.shape[1]
    features = (num.T / den).T
    # Zscoring if needed
    if z_score:
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        features = (features - means) / stds
    return features, mfw


def compute_links_mfw(X, n_grams, n_mfw, z_score, lidstone_lambda, distance_func, remove_hapax=True):
    # Tokens normalization
    if type(X[0]) == str:
        X = [[normalize_text(t) for t in xi] for xi in X]
    # Convert text to n_grams
    if type(n_grams) == list or type(n_grams) == tuple or n_grams > 0:
        X = [create_n_grams(xi, n_grams) for xi in X]
    # Create features
    features, mfw = most_frequent_word(X, n_mfw, z_score, lidstone_lambda, remove_hapax)
    # Compute link distances into a 2D matrix
    distances_matrix = squareform(pdist(features, metric=distance_func))
    # Computing the rank list of for this distance matrix
    rank_list = rank_list_from_distances_matrix(distances_matrix)
    return rank_list


def compute_links_compress(X, compression_method, distance_func):
    X = ["_".join(Xi).encode("utf8") for Xi in X]

    X_sizes = [compression_method(Xi) for Xi in X]

    pairs_indices = list(itertools.combinations(range(len(X)), 2))

    X_sizes_pairs = {(a, b): compression_method(
        X[a] + X[b]) for a, b in pairs_indices}

    rank_list = [((a, b), distance_func(X_sizes[a], X_sizes[b],
                                        X_sizes_pairs[(a, b)])) for a, b in pairs_indices]
    rank_list.sort(key=lambda x: x[-1])
    return rank_list


def compute_links(text_representation):
    if type(text_representation) == list:
        return compute_links_mfw(*text_representation)
    elif type(text_representation) == tuple:
        return compute_links_compress(*text_representation)
    else:
        return None


if __name__ == '__main__':
    # Simple test script for the module
    # _, _, X, Y = brunet.parse()
    _, X, Y = oxquarry.parse()
    # _, _, _, X, Y = st_jean.parse_B()

    print("AP RPrec P@10 HPrec")
    rank_list = compute_links([X, 3, 500, False, 0.1, distances.manhattan])
    print(*evaluate_linking(rank_list, Y))
    rank_list = compute_links((X, compressions.bz2, distances.ncd))
    print(*evaluate_linking(rank_list, Y))
