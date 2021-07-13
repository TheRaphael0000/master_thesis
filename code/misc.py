"""Misc module

This module contains divers functions used in the project.
"""

from collections import Counter
import itertools
import unicodedata

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom_test
from scipy.stats import beta
from scipy.stats import weightedtau
from scipy.stats import wilcoxon


def compute_r(Y):
    """Find the number of true links in a dataset

    Arguments:
        Y -- The labels of a dataset

    Returns:
        int -- The number of true links
    """
    return int(sum([(n * (n - 1)) / 2 for n in dict(Counter(Y)).values()]))


def dataset_infos(x, y):
    """Different basic metrics used to evaluate a dataset

    Arguments:
        x -- The documents in a dataset
        y -- The dataset labels

    Returns:
        int -- The number of authors
        int -- The number of texts
        float -- The ratio author / texts
        int -- The number of links
        float -- The true link ratio
        float -- The mean length of the texts
        float -- The mean length of the words
    """
    authors = len(set(y))
    texts = len(y)
    r = round(authors / texts, 3)
    true_links = compute_r(y)
    links = (texts * (texts - 1)) // 2
    true_links_ratio = round(true_links / links, 3)
    mean_length = round(np.mean([len(xi) for xi in x]))
    mean_token_length = round(np.mean(
        list(itertools.chain.from_iterable([[len(w) for w in xi] for xi in x]))), 3)
    return authors, texts, r, true_links, links, true_links_ratio, mean_length, mean_token_length


def zipf_law(total, n=21):
    """Plot the zipf law of a dictionary of words frequencies

    Arguments:
        total -- The words frequency dictionary
        n -- The number of words to keep

    Returns:
        void -- Create an image
    """
    distribution = dict(Counter(total.values()))
    keys = distribution.keys()
    A = list(range(1, n))
    B = [distribution[i] if i in distribution else 0 for i in A]
    plt.figure(figsize=(4, 3), dpi=200)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot(A, B)
    plt.tight_layout()
    plt.savefig("img/zipf.png")


def rank_list_from_distances_matrix(distances_matrix):
    """Convert distances' matrix into a rank list

    Arguments:
        distances_matrix -- The distences' matrix

    Return:
        rank_list -- The rank list corresponding to the provided distances' matrix
    """
    rank_list = []
    for a, b in itertools.combinations(range(distances_matrix.shape[0]), 2):
        link = a, b
        dist = distances_matrix[link]
        rank_list.append((link, dist))
    rank_list.sort(key=lambda x: x[-1])
    return rank_list


def distances_matrix_from_rank_list(rank_list):
    """Convert rank list into a distances' matrix

    Arguments:
        rank_list -- The rank list

    Return:
        distances_matrix -- The distences' matrix corresponding to the provided rank list
    """
    indices = list(itertools.chain.from_iterable([i[0] for i in rank_list]))
    w = len(np.unique(indices))
    # print(w)
    distances_matrix = np.full((w, w,), np.nan)
    # distances_matrix = np.zeros((w, w,))
    for (a, b), dist in rank_list:
        distances_matrix[a, b] = dist
        distances_matrix[b, a] = dist
    return distances_matrix


def features_from_rank_list(rank_list):
    """Create a feature vector for each rank list sample.
    Feature used :
        - log of the relative rank
        - score

    Arguments:
        rank_list -- A rank list

    Returns:
        array -- An array with a feature vectore for each sample in a rank list
    """
    return [[np.log((i + 1) / len(rank_list)), score]
            for i, (link, score) in enumerate(rank_list)]


def labels_from_rank_list(rank_list, Y):
    """Create a label vector corresponding to each sample in a rank list

    Arguments:
        rank_list -- A rank list
        Y -- Its labels

    Returns:
        array -- The label vector
    """
    return [1 if Y[a] == Y[b] else 0 for (a, b), score in rank_list]


def division(A, B):
    """Shorthand for the numpy division replacing division by 0 with 0

    Arguments:
        A -- numerator
        B -- denominator

    Return:
        float -- quotient
    """
    return np.divide(A, B, where=B != 0, out=np.zeros(A.shape))


def log(A):
    """Shorthand for the numpy logarithm replacing 0 or negative log by 0

    Arguments:
        A -- input

    Return:
        falot -- output
    """
    return np.log(A, where=A > 0, out=np.full(A.shape, np.inf))


def normalize(x, a, b):
    """Normalize between a-b an array like

    Arguments:
        x -- Vector
        a -- Lower bound
        b -- Upper bound

    Return:
        array -- the vector x normalized between a and b
    """
    input_range = np.max(x) - np.min(x)
    output_range = b - a
    return (x - np.min(x)) / input_range * output_range + a


def sigmoid(x):
    """The sigmoid function

    Arguments:
        x -- input / argument

    Return:
        y -- output / value of the function
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_r(x):
    """The sigmoid inverse function

    Arguments:
        x -- input / argument

    Return:
        y -- output / value of the function
    """
    return - np.log((1 - x) / x)


def sign_test(A, B):
    """Sign test between A and B

    Arguments:
        A -- Popluation A
        B -- Popluation B

    Return:
        list -- Lines for the positive, equal, negatives tests
        list -- The binomial test
    """
    pos = np.sum(A > B, axis=0)
    eq = np.sum(A == B, axis=0)
    neg = np.sum(A < B, axis=0)

    binom_tests = [binom_test([pos[i], neg[i]]) for i in range(pos.shape[0])]
    binom_tests = np.array(binom_tests)

    return np.array([pos, eq, neg]).T.tolist(), list(binom_tests)


def print_side_by_side_clusters(Y_true, Y_pred):
    """Print the clusters side by side

    Arguments:
        Y_true -- The clusters grountruth labels
        Y_pred -- The predicted clusters

    Return:
        void
    """
    for a in sorted(zip(Y_true, Y_pred), key=lambda x: x[-1]):
        print(a)


def rank_list_to_txt(rank_list, Y):
    """Create a text file to save a rank list

    Arguments:
        rank_list -- The rank list to serialize
        Y -- The labels for this rank list
    """
    with open("rank_list.txt", "w") as f:
        for rank, ((a, b), score) in enumerate(rank_list):
            f.write(f"{rank}, {'1' if Y[a] == Y[b] else '0'}, {str(score)}\n")


def normalize_text(s):
    """Normalize a string s, keep only lowercase ASCII characters

    Arguments:
        s -- A string

    Return:
        str -- The string s normalize
    """
    # Removing accents
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ASCII", "ignore").decode("ASCII")
    # To lowercase
    s = s.lower()
    return s


def first_letters_cut(X, n):
    """Cut the words to only keep the n first characters

    Arguments:
        X -- The documents
        n -- The size of the cut, the n first characters are kept

    Returns:
        array -- The documents with only the n first characters
    """
    return [[x[:n] for x in Xi] for Xi in X]


def last_letters_cut(X, n):
    """Cut the words to only keep the n last characters

    Arguments:
        X -- The documents
        n -- The size of the cut, the n last characters are kept

    Returns:
        array -- The documents with only the n last characters
    """
    return [[x[-n:] for x in Xi] for Xi in X]


def create_word_n_gram(word, n):
    """Create a n-grams for a word

    Arguments:
        word -- A word
        n -- The size of the cut, the n last characters are kept

    Returns:
        array -- The n-grams correspond to a word
    """
    return [word[i:i + n] for i in range(len(word) - n + 1)]


def word_n_grams(X, n):
    """Create a n-grams for every word in a multiple documents

    Arguments:
        X -- The documents
        n -- The size of the n-grams

    Returns:
        array -- The documents n-grams
    """
    texts = []
    for Xi in X:
        words = []
        for x in Xi:
            n_grams = create_word_n_gram(x, n)
            words.extend(n_grams)
        texts.append(words)
    return texts


def rank_list_distance(A, B):
    """Distance between two rank lists using the Kendall’s weighted tau.

    Agruments:
        A -- Rank list A
        B -- Rank list B

    Returns:
        float -- Kendall's weighted tau correlation coefficient
    """
    def rl_to_rank_dict(rl):
        return {link: i for i, (link, score) in enumerate(rl)}

    dA = rl_to_rank_dict(A)
    dB = rl_to_rank_dict(B)
    assert set(dA.keys()) == set(dB.keys())
    keys = dA.keys()

    vA, vB = [dA[k] for k in keys], [dB[k] for k in keys]
    correlation, pvalue = weightedtau(vA, vB)
    return correlation


def sort_Y_and_distance_matrix(Y, distances_matrix):
    """Sort a distance matrix and its label alphabetically

    Arguments:
        Y -- The labels
        distances_matrix -- The distance matrix

    Returns:
        Y -- The labels sorted
        distances_matrix -- The distance matrix sorted
    """
    Y = np.array(Y)
    Y_arg = np.argsort(Y)
    Y = Y[Y_arg]
    distances_matrix = distances_matrix[Y_arg, :]
    distances_matrix = distances_matrix[:, Y_arg]
    return Y, distances_matrix


def subset_Y_and_distance_matrix(Y, distances_matrix, subset):
    """Extract from a distance matrix a subset of labels

    Arguments:
        Y -- The labels
        distances_matrix -- The distance matrix
        subset -- The subset to extact

    Returns:
        Y -- The subset of labels
        distances_matrix -- The distances' matrix subset
    """
    Y = np.array(Y)
    ids = np.array(list(itertools.chain(
        *[np.argwhere(Y == s_)[:, 0] for s_ in subset]))).flatten()
    Y = Y[ids]
    distances_matrix = distances_matrix[ids, :]
    distances_matrix = distances_matrix[:, ids]
    return Y, distances_matrix


def fit_beta(Xi):
    """Find the beta distribution corresponding to the provided values

    Arguments:
        Xi -- The poplutation of fit

    Return:
        rv_continuous -- A continuous scipy function, the beta distribution corresponding to Xi
    """
    mean = np.mean(Xi)
    var = np.std(Xi)**2
    _ = ((mean * (1 - mean)) / var - 1)
    a = mean * _
    b = (1 - mean) * _
    return beta(a, b)


def find_two_beta_same_area(b1, b2):
    """Use a binary search to find where the two beta distribution provided have the same area under the curve

    Arguments:
        b1 -- First beta distribution
        b2 -- Second beta distribution

    Return:
        float -- A value on the x axis, this position both function have the same area under the curve
    """
    if b1.mean() > b2.mean():
        beta_left, beta_right = b2, b1
    else:
        beta_left, beta_right = b1, b2

    i_max = 1000
    alpha = 1e-15

    dt_l = 0
    dt_r = 1
    for _ in range(i_max):
        dt = (dt_r - dt_l) / 2 + dt_l
        s_l = beta_left.cdf(dt)
        s_r = 1 - beta_right.cdf(dt)

        delta_s = s_r - s_l

        if np.abs(delta_s) < alpha:
            break

        if delta_s > 0:
            dt_l = dt
        else:
            dt_r = dt

    return dt
