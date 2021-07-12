"""Linking module

This module provide function to create rank list for authorship verification.
It either use the most frequent approach (MF) or the compression based methods.
For the MF, it can be tokens, lemma, letters n-grams or POS sequences.
"""

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
    """Create a list of n-grams from a list words.

    Arguments:
        words -- It can be with strings or integers. String are for list of tokens / lemmas. Integers are for POS sequences
        ns -- n-grams size

    Return:
        list -- A list containing every in n-grams of size ns with the list words.
    """
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


def most_frequent(X, n, z_score=False, lidstone_lambda=0.1, remove_hapax=True):
    """Create the feature vector using the most frequent (MF) method

    Arguments:
        X -- A 2D array, items list for each document.
        n -- The number of n-MF items to keep, this correspond to the feature vector size.
        z_score -- If the vectors should by Z-Score normalized or not (default: False)
        lidstone_lambda -- The Lidstone smoothing technique parameter (default: 0.1)
        remove_hapax -- If the Hapax legomenon should be removed from the item count (default: True)

    Return:
        np.array -- A 2D array, a feature vector for each document in X
        mfw -- A dict of the n most frequent items in X
    """
    # Counting the terms in each documents
    counters = [Counter(xi) for xi in X]
    # Remove hapax legomenon for each texts
    if remove_hapax:
        counters = [Counter({w: n for w, n in dict(c).items() if n > 1}) for c in counters]
    # Computing the corpus word frequency
    total = reduce(lambda x, y: x + y, counters)
    # n bounding
    if n > len(total):
        n = len(total)
    # Selecting n mfw
    mfw = dict(total.most_common(n))
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


def compute_links_mf(X, n_grams, n_mf, z_score, lidstone_lambda, distance_func, remove_hapax=True):
    """Create the rank list based on the MF method

    Arguments:
        X -- The documents list
        n_grams -- The n-grams size, 0 correspond to no n-grams (whole tokens)
        n_mf -- The number of n-MF items to keep, this correspond to the feature vector size.
        z_score -- If the vectors should by Z-Score normalized or not
        lidstone_lambda -- The Lidstone smoothing technique parameter (default: 0.1)
        distance_func -- The distance function to use to compute the distance between vectors pairs (see distances module)
        remove_hapax -- If the Hapax legomenon should be removed from the item count (default: True)

    Return:
        rank_list -- A list containg every pairwise document pair and their distance according to the parameters
    """
    # Tokens normalization
    if type(X[0]) == str:
        X = [[normalize_text(t) for t in xi] for xi in X]
    # Convert text to n_grams
    if type(n_grams) == list or type(n_grams) == tuple or n_grams > 0:
        X = [create_n_grams(xi, n_grams) for xi in X]
    # Create features
    features, mfw = most_frequent(X, n_mf, z_score, lidstone_lambda, remove_hapax)
    # Compute link distances into a 2D matrix
    distances_matrix = squareform(pdist(features, metric=distance_func))
    # Computing the rank list of for this distance matrix
    rank_list = rank_list_from_distances_matrix(distances_matrix)
    return rank_list


def compute_links_compress(X, compression_method, distance_func):
    """Create the rank list based on the compression method

    Arguments:
        X -- The documents list
        compression_method -- The compression algorithm to use (see compressions module)
        distance_func -- The distance function to use to compute the distance between vectors pairs (see distances module)
    Return:
        rank_list -- A list containg every pairwise document pair and their distance according to the parameters
    """
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
    """Function to help the computation of rank lists

    Compute either the MF based rank list or the Compression based rank list depending on the type of container used for the arguments.

    Arguments:
        List -- for the MF-based rank lists
        Tuples -- for the compression-based rank lists

    Return:
        list -- The rank list
    """
    if type(text_representation) == list:
        return compute_links_mf(*text_representation)
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
